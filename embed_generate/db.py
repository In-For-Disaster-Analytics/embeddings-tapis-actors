"""
embeddingsdb client for the embed-generate Actor -- REAL psycopg2 writes.

Grounded directly in embeddings-tapis-actors/schema/embeddingsdb.sql's own
CREATE TABLE statements for `tile_observations`, `embeddings`, and
`covariates` -- not memory -- and mirrors
coreplugins/embeddings/embeddings_client.py's connection/error-handling
style (the WebODM-side client for the same database, see Decision 34), so
both sides of this system talk to embeddingsdb the same way.

This module is deliberately narrow: it implements ONLY the three functions
this increment scopes as real (`write_tile_observation`, `write_embedding`,
`write_covariates` -- see design spec Decision 35). Everything upstream of
these calls (tile enumeration, pixel fetching, Clay v1.5 inference, and
therefore `embed_generate/main.py`'s own `run()`) is still a documented
`NotImplementedError` stub and never actually reaches this module yet.

Reads `EMBEDDINGSDB_URL` directly from the process environment (this
Actor's own env var, matching `.env.example` -- NOT `WO_EMBEDDINGS_DB_URL`,
which is the WebODM Django-settings name for the same connection string used
by `embeddings_client.py` on the other side).

Structured-result note, not used here: a real Tapis Actor can also return
results via a Unix socket contract (`/_abaco_results.sock`, tapipy's
`send_bytes_result()`/`send_python_result()`, retrievable through Tapis's own
`/results` endpoint) -- this system's real state lives in embeddingsdb
itself (this module), not in Abaco's results queue. Documented here so a
future implementer doesn't have to rediscover it, not because this
increment uses it.
"""

import logging
import os

import psycopg2

logger = logging.getLogger(__name__)

# embeddingsdb is a small Tapis Pod handling low-volume Actor traffic -- a
# short connect timeout is enough to fail fast rather than hang an Actor
# execution. Matches embeddings_client.py's identical rationale/value.
DEFAULT_CONNECT_TIMEOUT = 10  # seconds


class EmbeddingsDBConfigError(RuntimeError):
    """Raised when EMBEDDINGSDB_URL isn't configured for this Actor."""


class EmbeddingsDBError(RuntimeError):
    """Raised on a real connection/query failure against embeddingsdb.
    Mirrors embeddings_client.EmbeddingsDBError's "raise a clear exception,
    don't fail silently" style.
    """


def _connect():
    """Opens a new connection to embeddingsdb, one short-lived connection
    per top-level call unless a caller passes its own `conn` through (see
    the `conn=` parameter on each write function below, for batching a
    tile's writes into one transaction)."""
    url = (os.environ.get('EMBEDDINGSDB_URL') or '').strip()
    if not url:
        raise EmbeddingsDBConfigError(
            "EMBEDDINGSDB_URL is not set -- this Actor's embeddingsdb "
            "writes cannot run without it. See .env.example."
        )
    try:
        return psycopg2.connect(url, connect_timeout=DEFAULT_CONNECT_TIMEOUT)
    except psycopg2.Error as e:
        logger.exception('Could not connect to embeddingsdb')
        raise EmbeddingsDBError(f'Could not connect to embeddingsdb: {e}') from e


def _execute(conn, query, params=None, fetch=None):
    """Shared cursor/error-handling helper for every query below. fetch:
    None | 'one' | 'all'. Raises EmbeddingsDBError on any psycopg2 failure."""
    with conn.cursor() as cur:
        try:
            cur.execute(query, params or ())
        except psycopg2.Error as e:
            conn.rollback()
            logger.exception('embeddingsdb query failed: %s', query)
            raise EmbeddingsDBError(f'embeddingsdb query failed: {e}') from e
        if fetch == 'one':
            return cur.fetchone()
        if fetch == 'all':
            return cur.fetchall()
        return None


def write_tile_observation(tile_grid_id, visit_id, pixel_size, conn=None):
    """
    Real upsert of one `tile_observations` row, keyed on the table's own
    `UNIQUE (tile_grid_id, visit_id)` constraint (schema/embeddingsdb.sql)
    -- this is what makes "ensure a tile_observations row exists"
    (embed_generate/main.py's module docstring, step 3) an actual upsert
    rather than a plain INSERT that would violate the constraint on re-run.

    `conn`: an existing psycopg2 connection to reuse (e.g. so a caller can
    wrap a whole tile's writes -- tile_observation + embedding + covariates
    -- in one transaction). A new short-lived connection is opened and
    committed/closed here if not given, matching embeddings_client.py's
    per-call simplicity.

    Returns the row's id (str, uuid).
    """
    owns_conn = conn is None
    conn = conn or _connect()
    try:
        row = _execute(
            conn,
            "INSERT INTO tile_observations (tile_grid_id, visit_id, pixel_size) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (tile_grid_id, visit_id) "
            "DO UPDATE SET pixel_size = EXCLUDED.pixel_size "
            "RETURNING id;",
            (tile_grid_id, visit_id, pixel_size),
            fetch='one',
        )
        if owns_conn:
            conn.commit()
        return str(row[0])
    finally:
        if owns_conn:
            conn.close()


def write_embedding(tile_observation_id, encoder_id, vector, conn=None):
    """
    Real INSERT into `embeddings` (schema/embeddingsdb.sql: id,
    tile_observation_id FK, encoder_id FK, `vector vector(1024) NOT NULL`).

    `vector`: an iterable of floats. There is no `pgvector` python package
    in requirements.txt (not added this increment -- this is the only call
    site) -- the vector is formatted as pgvector's own text input format
    (`'[v1,v2,...]'`) and cast with `%s::vector` in the query, which is
    pgvector's documented plain-SQL input syntax, not a workaround.

    No upsert here: unlike `tile_observations`, `embeddings` has no
    UNIQUE(tile_observation_id, encoder_id) constraint in the current schema
    (schema/embeddingsdb.sql) -- calling this twice for the same
    (tile_observation_id, encoder_id) creates two rows, not an error. That
    matches the schema exactly as written today; flagged here rather than
    silently assumed away, in case a re-run policy needs revisiting later.

    Returns the row's id (str, uuid).
    """
    vector_literal = '[' + ','.join(repr(float(v)) for v in vector) + ']'
    owns_conn = conn is None
    conn = conn or _connect()
    try:
        row = _execute(
            conn,
            "INSERT INTO embeddings (tile_observation_id, encoder_id, vector) "
            "VALUES (%s, %s, %s::vector) RETURNING id;",
            (tile_observation_id, encoder_id, vector_literal),
            fetch='one',
        )
        if owns_conn:
            conn.commit()
        return str(row[0])
    finally:
        if owns_conn:
            conn.close()


def write_covariates(tile_observation_id, covariates, conn=None):
    """
    Real INSERT into `covariates` (schema/embeddingsdb.sql: elevation,
    slope, aspect, chm, ndvi, ndwi -- all nullable `double precision`),
    matching Decision 25: "Returns None (no covariates row written) when
    the required source bands are absent -- never backfilled or faked"
    (embed_generate/main.py's compute_covariates() docstring).

    `covariates`: a dict with any subset of the keys above, or None/empty --
    if falsy, this function is a real no-op (returns None, writes nothing),
    matching that Decision precisely rather than writing an all-NULL row.

    Returns the row's id (str, uuid), or None if `covariates` was falsy.
    """
    if not covariates:
        return None
    owns_conn = conn is None
    conn = conn or _connect()
    try:
        row = _execute(
            conn,
            "INSERT INTO covariates "
            "(tile_observation_id, elevation, slope, aspect, chm, ndvi, ndwi) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;",
            (
                tile_observation_id,
                covariates.get('elevation'),
                covariates.get('slope'),
                covariates.get('aspect'),
                covariates.get('chm'),
                covariates.get('ndvi'),
                covariates.get('ndwi'),
            ),
            fetch='one',
        )
        if owns_conn:
            conn.commit()
        return str(row[0])
    finally:
        if owns_conn:
            conn.close()
