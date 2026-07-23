"""
embed-generate Tapis Actor -- entrypoint.

MOSTLY SKELETON, with two real pieces as of this increment (see Decision 35
in the design spec): the Abaco message contract (`read_actor_message()`) and
the three embeddingsdb write functions (`write_tile_observation`,
`write_embedding`, `write_covariates`, delegating to the real `embed_generate.db`
module). Every other function below still has a docstring describing what it
will do once the remaining supporting infrastructure (a deployed Clay v1.5
checkpoint, real WebODM-tiler/STAC-asset pixel access) exists, and a
`NotImplementedError` body -- no fabricated logic. See design spec:
WebODM/docs/design/2026-07-22-geospatial-embeddings-classification.md
(in the odm-suite monorepo-of-repos), Decisions 9, 19, 20, 23, 24, 25, 27, 35,
and the "New Infrastructure" / "Tile Coverage" / "Raster Source Independence"
sections.

Real Abaco message contract (confirmed against Tapis's own Actors docs, not
re-derived here): Abaco injects the invocation payload as the `MSG`
environment variable -- a JSON string -- and injects context values
(`_abaco_execution_id`, `_abaco_username`, etc.) as their own env vars.
`read_actor_message()` below reads `os.environ['MSG']` directly and
`json.loads()`s it -- the simplest, most portable mechanism, preferred over
adding a dependency on tapipy's actor-side helpers (`tapipy.actors`) for this.
Structured-result note, not used here: a real Actor can also return results
via a Unix socket contract (`/_abaco_results.sock`, tapipy's
`send_bytes_result()`/`send_python_result()`, retrievable through Tapis's own
`/results` endpoint) -- this system's real state lives in embeddingsdb, not
Abaco's results queue, so that mechanism is documented here for completeness
but genuinely not used.

What this Actor is meant to do, end to end, once implemented
--------------------------------------------------------------
Given an invocation payload identifying one `visit` (a WebODM task's
orthophoto, or an already-imported STAC item -- see the `visits` table in the
design spec's Embeddings DB Schema) and a zoom level:

1. Resolve the `site_id` the visit belongs to (user-chosen at trigger time,
   never inferred -- Decision 27) and that site's LOCKED zoom, if one is
   already set by an earlier visit (Decision 24). Honor `zoom_override` if
   the caller explicitly requested a different zoom than the site's existing
   `tile_grid` rows use.
2. Enumerate every valid `(z, x, y)` tile at that zoom for the visit's
   raster -- NOT a hand-picked subset (Decision 9):
   - If the visit has no `stac_item_id`: enumerate via WebODM's own tiler
     coverage logic (mirrors `app/api/tiler.py`'s `tile_exists(z, x, y)`
     checks against the orthophoto's real bounds) and fetch each tile's
     pixels from WebODM's own `/tiles/{z}/{x}/{y}` endpoint.
   - If the visit HAS a `stac_item_id` (a genuinely STAC-sourced visit, or a
     `webodm`-origin visit already published to the DSO STAC API -- Decision
     20's refinement of the branch condition): resolve the STAC item's COG
     asset href and enumerate/fetch tiles via `rio_tiler` directly against
     that asset -- the same library WebODM's own tiler already depends on.
3. For each tile: ensure a `tile_observations` row exists (keyed by
   `tile_grid_id`, `visit_id`), run the Clay v1.5 encoder (RGB-only, per the
   approved Phase 1 recommendation) over the tile's pixels, and write an
   `embeddings` row (`tile_observation_id`, `encoder_id`, `vector`).
4. Separately, where DSM/DTM/multispectral bands exist for this visit's
   source, compute `covariates` (elevation, slope, aspect, CHM, NDVI, NDWI,
   ...) for each `tile_observation` -- inside this same Actor run, not a
   separate Actor (Decision 25). Absent where the source bands don't exist;
   never backfilled or faked.
5. Report status back (however `.../task/{pk}/embed-status` polling is wired
   up to observe it -- not designed in this increment).

Explicitly NOT implemented in this increment
---------------------------------------------
- No Clay v1.5 checkpoint loading / no claymodel encoder instantiation.
- No WebODM tiler HTTP calls, no rio_tiler STAC-asset tiling.
- No Tapis service-token handling (Decision 30) -- credential plumbing for
  this Actor's async invocation is unresolved, see this repo's README.
- `run()` itself still raises NotImplementedError before ever reaching the
  now-real write_* functions below -- see the module docstring's "Real
  Abaco message contract" note above for what IS real this increment.
"""

import json
import os

from embed_generate import db


def main():
    """
    Actor entrypoint. Reads the real Abaco invocation message (`MSG` env
    var, see module docstring) and dispatches to `run()`.
    """
    message = read_actor_message()
    run(message)


def read_actor_message():
    """
    Parse this invocation's message payload from the `MSG` environment
    variable -- confirmed against Tapis's own Actors docs (Abaco injects the
    message as `MSG`, a JSON string; see module docstring). Raises a clear,
    specific error rather than a bare KeyError/JSONDecodeError if `MSG` is
    unset or malformed, so a misconfigured invocation fails loudly.

    Expected shape (per the design spec's `POST .../task/{task_pk}/embed`
    endpoint, which is what queues this Actor):
        {
            "visit_id": "...",       # embeddingsdb visits.id
            "site_id": "...",        # user-chosen, Decision 27
            "zoom": 19,
            "zoom_override": False,  # Decision 24/27
            "encoder": "clay-v1.5-large-rgb",
        }
    """
    raw = os.environ.get('MSG')
    if raw is None:
        raise RuntimeError(
            "MSG environment variable is not set. Abaco injects the "
            "invocation payload as MSG (confirmed against Tapis's own "
            "Actors docs) -- this Actor cannot determine what to run "
            "without it. Check how it was invoked (e.g. "
            "t.actors.send_message(actor_id=..., request_body={'message': "
            "...}))."
        )
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"MSG environment variable is not valid JSON: {e}. "
            f"Raw MSG value: {raw!r}"
        ) from e


def resolve_site_zoom(site_id, requested_zoom, zoom_override):
    """
    Decision 24/27: look up whether `site_id` already has `tile_grid` rows at
    a different zoom than `requested_zoom`. If so and `zoom_override` is not
    set, this should fail loudly (the caller -- WebODM's plugin -- is
    responsible for surfacing the UI confirmation warning before ever
    reaching this Actor). If no `tile_grid` rows exist yet for this site,
    `requested_zoom` becomes the site's locked zoom.
    """
    raise NotImplementedError(
        "Site zoom resolution is not implemented yet -- depends on the "
        "embeddingsdb Pod's tile_grid table, which does not exist yet."
    )


def enumerate_tiles(visit, zoom):
    """
    Decision 9: return every valid (z, x, y) at `zoom` for `visit`'s raster --
    the full coverage set, never a hand-picked subset.

    Branches per Decision 19/20:
    - visit.stac_item_id is None -> enumerate via WebODM's own tiler coverage
      logic (mirrors app/api/tiler.py's tile_exists(z, x, y) against the
      orthophoto's real bounds).
    - visit.stac_item_id is set -> resolve the STAC item's asset href and
      enumerate via rio_tiler directly against that asset.
    """
    raise NotImplementedError(
        "Tile enumeration is not implemented yet -- depends on WebODM's own "
        "tiler (webodm-sourced visits) or a real STAC item asset href "
        "(stac_item_id-bearing visits). See design spec 'Tile Coverage' and "
        "'Raster Source Independence' sections, Decisions 9, 19, 20."
    )


def fetch_tile_pixels(visit, z, x, y):
    """
    Fetch the actual pixel data for one (z, x, y) tile, via whichever source
    `enumerate_tiles` determined applies to this visit (WebODM's own tiler
    endpoint, or rio_tiler against a STAC asset href).
    """
    raise NotImplementedError(
        "Tile pixel fetching is not implemented yet. See design spec "
        "'Raster Source Independence' and Decisions 19/20/23."
    )


def embed_tile(pixels, sensor_metadata, capture_date, center_lat, center_lon):
    """
    Run the Clay v1.5 encoder (RGB-only) over one tile's pixels, producing an
    embedding vector.

    The real implementation should follow
    embeddings-research/scripts/clay_embed_sized.py's conditioning logic
    (band wavelength/mean/std normalization from Clay's own
    configs/metadata.yaml, sin/cos time-of-year and lat/lon encoding, a
    256x256 input tile) rather than reinventing it -- that script is the
    validated reference for how this repo's Phase 1 research actually called
    Clay. Not implemented here: no checkpoint is loaded, no `claymodel`
    encoder is instantiated (see requirements.txt for why `claymodel` isn't a
    normal pip dependency yet).
    """
    raise NotImplementedError(
        "Clay v1.5 inference is not implemented yet -- no checkpoint is "
        "wired into this repo. See CLAY_CHECKPOINT_PATH/CLAY_MODEL_DIR in "
        ".env.example and README 'Next steps' item 3."
    )


def compute_covariates(visit, tile_observation):
    """
    Decision 25: compute elevation/slope/aspect/CHM/NDVI/NDWI covariates for
    one tile_observation from DSM/DTM/multispectral bands, when they exist
    for this visit's source. Returns None (no covariates row written) when
    the required source bands are absent -- never backfilled or faked.
    """
    raise NotImplementedError(
        "Covariate computation is not implemented yet -- depends on real "
        "DSM/DTM/multispectral raster access for the visit's source, which "
        "is not wired in yet."
    )


def write_tile_observation(tile_grid_id, visit_id, pixel_size):
    """
    Real, as of this increment (Decision 35): upsert one `tile_observations`
    row in embeddingsdb via `embed_generate.db.write_tile_observation()`.
    Requires `EMBEDDINGSDB_URL` to be set in this Actor's environment (see
    .env.example) -- raises `db.EmbeddingsDBConfigError`/`EmbeddingsDBError`
    otherwise. See `db.py` for the real query and its grounding in
    `schema/embeddingsdb.sql`.
    """
    return db.write_tile_observation(tile_grid_id, visit_id, pixel_size)


def write_embedding(tile_observation_id, encoder_id, vector):
    """
    Real, as of this increment (Decision 35): write one `embeddings` row
    (pgvector column) via `embed_generate.db.write_embedding()`. See `db.py`
    for how `vector` is formatted for pgvector's text input syntax.
    """
    return db.write_embedding(tile_observation_id, encoder_id, vector)


def write_covariates(tile_observation_id, covariates):
    """
    Real, as of this increment (Decision 35): write one `covariates` row via
    `embed_generate.db.write_covariates()`, which is a real no-op (returns
    None, writes nothing) when `covariates` is falsy -- matching Decision 25
    ("never backfilled or faked").
    """
    return db.write_covariates(tile_observation_id, covariates)


def run(message):
    """
    Orchestrates the full embed-generate flow described in the module
    docstring above, for one invocation message.
    """
    raise NotImplementedError(
        "embed-generate's end-to-end flow is not implemented yet -- see this "
        "module's docstring for the intended sequence, and this repo's "
        "README 'Next steps' for what has to exist first."
    )


if __name__ == "__main__":
    main()
