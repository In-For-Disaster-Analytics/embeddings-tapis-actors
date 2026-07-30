"""
embed-generate Tapis Actor -- entrypoint.

Real as of this increment (Decision 44, following Decisions 35/37/40/41):
the full `run()` pipeline for a WebODM-sourced visit (`visits.stac_item_id
IS NULL`) -- site-zoom resolution, tile enumeration via WebODM's own tiler,
per-tile pixel fetch, real Clay v1.5 inference (`embed_generate.clay`), and
embeddingsdb writes (`embed_generate.db`). See design spec:
WebODM/docs/design/2026-07-22-geospatial-embeddings-classification.md
(in the odm-suite monorepo-of-repos), Decisions 9, 19, 20, 23, 24, 25, 27,
35, 37, 39, 40, 41, 44, and the "New Infrastructure" / "Tile Coverage" /
"Raster Source Independence" sections.

Explicitly NOT implemented in this increment (Decision 40's own scope cut):
- The STAC-asset tiling branch (Decisions 19/20/23) -- `enumerate_tiles()`
  raises `NotImplementedError` for a `stac_item_id`-bearing visit rather
  than silently mishandling it.
- Real DSM/DTM/multispectral-derived `covariates` -- `compute_covariates()`
  always returns None this increment; see its own docstring for exactly
  why (not simply "not gotten to yet" -- a real, stated technical gap).

Real Abaco message contract (confirmed against Tapis's own Actors docs, not
re-derived here): Abaco injects the invocation payload as the `MSG`
environment variable, and injects context values (`_abaco_execution_id`,
`_abaco_username`, etc.) as their own env vars. `read_actor_message()`
below reads `os.environ['MSG']` -- the simplest, most portable mechanism,
preferred over adding a dependency on tapipy's actor-side helpers
(`tapipy.actors`) for this.

`MSG` is base64-encoded JSON, not plain JSON (Decision 45 follow-up) --
this Actor no longer runs under Abaco at all (see Decision 45: it runs as
a Tapis Job on ls6 instead), and that execution path's SINGULARITY runtime
joins every env var into one comma-separated `apptainer run --env
k1=v1,k2=v2,...` argument, which a plain-JSON value (full of commas and
quotes) breaks. Base64 survives that untouched. `read_actor_message()`
decodes it before `json.loads()`.
Structured-result note, not used here: a real Actor can also return results
via a Unix socket contract (`/_abaco_results.sock`, tapipy's
`send_bytes_result()`/`send_python_result()`, retrievable through Tapis's own
`/results` endpoint) -- this system's real state lives in embeddingsdb, not
Abaco's results queue, so that mechanism is documented here for completeness
but genuinely not used.
"""

import base64
import json
import logging
import os
import time

from embed_generate import clay, db, webodm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    endpoint, which is what queues this Actor -- extended per Decision 41
    with `project_pk`/`webodm_jwt`):
        {
            "visit_id": "...",       # embeddingsdb visits.id
            "site_id": "...",        # user-chosen, Decision 27
            "zoom": 19,
            "zoom_override": False,  # Decision 24/27
            "encoder": "clay-v1.5-large-rgb",
            "project_pk": 1,         # Decision 41 -- WebODM Project.id
            "webodm_jwt": "...",     # Decision 44 (correcting Decision 41):
                                     #   a WebODM-NATIVE JWT minted for the
                                     #   requesting user (rest_framework_jwt),
                                     #   NOT the per-user Tapis access token
                                     #   used to invoke this Actor itself --
                                     #   those are different tokens with
                                     #   different signing secrets, confirmed
                                     #   by testing the Tapis token directly
                                     #   against WebODM's tiler endpoint (it
                                     #   fails). Forwarded to WebODM's tiler
                                     #   endpoint as ?jwt=.
        }
    """
    raw = os.environ.get('MSG')
    if raw is None:
        raise RuntimeError(
            "MSG environment variable is not set. Abaco injects the "
            "invocation payload as MSG (confirmed against Tapis's own "
            "Actors docs) -- this Actor cannot determine what to run "
            "without it. Check how it was invoked (e.g. "
            "t.actors.sendMessage(actor_id=..., message=...))."
        )
    # Decision 45 follow-up: MSG is base64-encoded JSON, not plain JSON.
    # Found via a real, failed ls6 Job run: Tapis's SINGULARITY runtime
    # joins EVERY env var (its own _tapisXxx ones plus ours) into a single
    # comma-separated `apptainer run --env k1=v1,k2=v2,...` argument. A
    # plain-JSON MSG value (commas between keys, quoted strings) breaks
    # that naive join -- confirmed from the job's own tapisjob.out:
    # "parse error ... bare \" in non-quoted-field". Abaco never had this
    # problem (it sets MSG as a real, standalone process env var, no
    # joining) -- this is specific to the SINGULARITY/Tapis-Job delivery
    # path. Base64 output has no commas/quotes, so it survives untouched;
    # apply_embed_generate() (WebODM repo) base64-encodes the same way.
    try:
        raw = base64.b64decode(raw).decode('utf-8')
    except Exception as e:
        raise ValueError(
            f"MSG environment variable is not valid base64: {e}. "
            f"Raw MSG value: {raw!r}"
        ) from e
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"MSG environment variable (base64-decoded) is not valid JSON: {e}. "
            f"Decoded MSG value: {raw!r}"
        ) from e


def resolve_site_zoom(site_id, requested_zoom, zoom_override):
    """
    Decision 24/27: a site's zoom is locked by its first `tile_grid` row.
    WebODM's own `TaskEmbedView.post()` already runs this exact check
    (`embeddings_client.get_site_zoom()`) and returns 409 before ever
    queuing this Actor if it would mismatch without `zoom_override` -- so
    reaching this Actor with a real mismatch should not normally happen.
    This is a defense-in-depth re-check (e.g. against a race between two
    concurrent embed calls for the same site), not a duplicate of the UI
    flow -- there is no user watching this Actor's own failure, so it fails
    loudly (raises) rather than trying to surface a 409-shaped response.

    Returns the effective zoom to use -- always `requested_zoom` itself
    (an override doesn't retroactively change already-embedded tiles' zoom,
    it just permits this run to proceed at a new one).
    """
    existing_zoom = db.get_site_zoom(site_id)
    if existing_zoom is not None and existing_zoom != requested_zoom and not zoom_override:
        raise RuntimeError(
            f"Site {site_id} already has tile_grid rows at zoom "
            f"{existing_zoom}, but this invocation requested zoom "
            f"{requested_zoom} without zoom_override. WebODM's own "
            f"TaskEmbedView should have already blocked this with a 409 "
            f"before queuing this Actor (Decision 24/27) -- failing here "
            f"as defense-in-depth, not as the primary check."
        )
    return requested_zoom


def enumerate_tiles(visit, zoom):
    """
    Decision 9: return every candidate (x, y) at `zoom` for `visit`'s
    raster's bounding box -- NOT yet the final coverage set (that's
    `fetch_tile_pixels()`'s own 404 handling in `run()`, mirroring WebODM's
    own `tile_exists()` check, which this function does not reimplement).

    Decision 40: only the `stac_item_id is None` (WebODM-sourced) branch is
    implemented this increment. A `stac_item_id`-bearing visit would need
    the STAC-asset/`rio_tiler` tiling path (Decisions 19/20/23) instead --
    explicitly not built yet, so this raises rather than silently
    mishandling it.
    """
    if visit.get('stac_item_id'):
        raise NotImplementedError(
            "STAC-asset tiling (Decisions 19/20/23) is not implemented in "
            "this increment (Decision 40) -- this visit has a stac_item_id "
            "and needs rio_tiler-against-asset-href tiling, not WebODM's "
            "own tiler endpoint."
        )
    _minzoom, _maxzoom, bounds = webodm_client.get_tile_coverage(
        visit['webodm_url'], visit['project_pk'], visit['webodm_task_id'],
        visit['webodm_jwt'],
    )
    return list(webodm_client.candidate_tiles(bounds, zoom))


def fetch_tile_pixels(visit, z, x, y):
    """
    Fetch one (z, x, y) tile's real pixels from WebODM's own tiler endpoint
    (Decision 40's WebODM-sourced branch; Decision 41's jwt-authenticated
    request). Raises `webodm_client.TileNotFound` if this tile genuinely
    isn't covered (WebODM's own `tile_exists()` said no) -- callers should
    treat that as "skip," not a fatal error.
    """
    return webodm_client.fetch_tile(
        visit['webodm_url'], visit['project_pk'], visit['webodm_task_id'],
        z, x, y, visit['webodm_jwt'],
    )


def embed_tile(pixels, capture_date, center_lat, center_lon, size, model_dir, checkpoint_path):
    """
    Run the Clay v1.5 encoder (RGB-only, per the approved Phase 1
    recommendation) over one tile's pixels, producing an embedding vector.
    Delegates to `embed_generate.clay.embed_tile()` -- see that module for
    the real conditioning-input construction (band wavelength/mean/std,
    sin/cos time-of-year, sin/cos lat/lon, GSD) mirrored from
    `embeddings-research/scripts/clay_embed_sized.py`.
    """
    return clay.embed_tile(pixels, capture_date, center_lat, center_lon, size, model_dir, checkpoint_path)


def compute_covariates(visit, z, x, y):
    """
    Decision 25: compute elevation/slope/aspect/CHM/NDVI/NDWI covariates for
    one tile_observation from DSM/DTM/multispectral bands, when they exist
    for this visit's source. Returns None (no covariates row written) when
    the required source bands are absent -- never backfilled or faked.

    Always returns None in this increment -- an honest, stated scope cut,
    not a claim that DSM/DTM is universally absent:
    - Real elevation/slope/aspect needs the RAW float DSM values, but
      WebODM's tiler renders DSM/DTM tiles as colormapped 8-bit PNGs for map
      display (`app/api/tiler.py`), not a raw-value format. Reconstructing
      true elevation would mean also fetching `/metadata`'s rescale range
      and treating an approximate reconstruction as real data -- not done
      here.
    - NDVI/NDWI need multispectral bands, which Phase 1's own research
      found absent in every WebODM task checked (design spec "Phase 1
      Research Findings: own task history has a real multispectral-
      processing gap") -- these would be None for WebODM-sourced visits
      regardless of this function's own implementation state.

    A real implementation is future work, not this increment's scope
    (Decision 40).
    """
    return None


def run(message):
    """
    Orchestrates the full embed-generate flow for one invocation message,
    for a WebODM-sourced visit (Decision 40's scope): resolve the effective
    zoom, enumerate candidate tiles, fetch + embed each real tile, write
    `tile_observations`/`embeddings` (and `covariates`, when
    `compute_covariates()` returns something -- currently never, this
    increment).
    """
    visit_id = message['visit_id']
    site_id = message['site_id']
    requested_zoom = message['zoom']
    zoom_override = bool(message.get('zoom_override', False))
    encoder_key = message['encoder']
    webodm_jwt = message.get('webodm_jwt')

    webodm_url = os.environ.get('WEBODM_URL')
    if not webodm_url:
        raise RuntimeError(
            "WEBODM_URL is not set -- this Actor cannot fetch tiles from "
            "WebODM without it. See .env.example."
        )

    visit_row = db.get_visit(visit_id)
    if visit_row is None:
        raise RuntimeError(f"No visits row found for visit_id={visit_id!r}.")

    # project_pk is on both the message (Decision 41) and the visits row
    # (Decision 41's schema addition) -- the visits row is authoritative if
    # they ever disagree (it's what get_or_create_visit() actually persisted).
    project_pk = visit_row['project_pk'] if visit_row['project_pk'] is not None else message.get('project_pk')
    if project_pk is None:
        raise RuntimeError(
            f"visit_id={visit_id!r} has no project_pk (neither the visits "
            f"row nor the invocation message carries one) -- cannot "
            f"construct WebODM's project-nested tiler URL (Decision 41)."
        )
    if not webodm_jwt:
        raise RuntimeError(
            "No webodm_jwt in the invocation message -- cannot authenticate "
            "tile fetches against WebODM (Decision 41/44)."
        )

    visit = {
        'webodm_url': webodm_url,
        'project_pk': project_pk,
        'webodm_task_id': visit_row['webodm_task_id'],
        'stac_item_id': visit_row['stac_item_id'],
        'webodm_jwt': webodm_jwt,
    }

    zoom = resolve_site_zoom(site_id, requested_zoom, zoom_override)

    name, version, size, band_config = clay.parse_encoder_key(encoder_key)
    encoder_id = db.get_or_create_encoder(name, version, size, band_config)

    checkpoint_path = os.environ.get('CLAY_CHECKPOINT_PATH')
    model_dir = os.environ.get('CLAY_MODEL_DIR')
    if not checkpoint_path or not model_dir:
        raise RuntimeError(
            "CLAY_CHECKPOINT_PATH/CLAY_MODEL_DIR must both be set -- "
            "cannot load the Clay v1.5 encoder without them. See .env.example."
        )

    candidates = enumerate_tiles(visit, zoom)
    logger.info(
        "embed-generate: visit=%s site=%s zoom=%d encoder=%s -- %d candidate tiles",
        visit_id, site_id, zoom, encoder_key, len(candidates),
    )

    embedded = 0
    skipped = 0
    total = len(candidates)
    start = time.monotonic()
    # Progress-only log line -- no other signal exists between the initial
    # candidate count and the final "complete" line otherwise, which made a
    # genuinely slow (CPU-only, large-encoder) run indistinguishable from a
    # hang on a real live run (~1764 tiles, 12+ minutes with zero output).
    log_every = 25

    for i, (x, y) in enumerate(candidates, start=1):
        try:
            image = fetch_tile_pixels(visit, zoom, x, y)
        except webodm_client.TileNotFound:
            # Real coverage gap (e.g. non-rectangular flight footprint) --
            # WebODM's own tile_exists() said no. Expected, not an error.
            skipped += 1
            continue

        bounds_wkt = webodm_client.tile_bounds_wkt(x, y, zoom)
        tile_grid_id = db.get_or_create_tile_grid(site_id, zoom, x, y, bounds_wkt)

        center_lat, center_lon = webodm_client.tile_center_lonlat(x, y, zoom)
        vector = embed_tile(
            image, visit_row['capture_date'], center_lat, center_lon,
            size, model_dir, checkpoint_path,
        )

        pixel_size = webodm_client.meters_per_pixel(zoom, center_lat)
        tile_observation_id = db.write_tile_observation(tile_grid_id, visit_id, pixel_size)
        db.write_embedding(tile_observation_id, encoder_id, vector.tolist())

        covariates = compute_covariates(visit, zoom, x, y)
        db.write_covariates(tile_observation_id, covariates)

        embedded += 1

        if i % log_every == 0 or i == total:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else float('inf')
            logger.info(
                "embed-generate progress: visit=%s %d/%d tiles processed "
                "(embedded=%d skipped=%d) -- %.1fs elapsed, %.1f tiles/s, "
                "~%.0fs remaining",
                visit_id, i, total, embedded, skipped, elapsed, rate, remaining,
            )

    logger.info(
        "embed-generate complete: visit=%s embedded=%d skipped=%d (no coverage)",
        visit_id, embedded, skipped,
    )
    return {'embedded': embedded, 'skipped': skipped}


if __name__ == "__main__":
    main()
