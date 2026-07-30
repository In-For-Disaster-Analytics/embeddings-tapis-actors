"""
HTTP client for WebODM's own tiler endpoints (`app/api/tiler.py` in the
WebODM repo) -- the Decision 40 scope for this increment: only the
`visits.stac_item_id IS NULL` branch (a genuine WebODM-sourced visit). The
STAC-asset/`rio_tiler` branch (Decisions 19/20/23) is a real, designed,
still-unimplemented code path -- see `main.py`'s explicit `NotImplementedError`
for a `stac_item_id`-bearing visit.

Grounded directly in the real WebODM routes (confirmed by reading
`app/api/urls.py`/`app/api/tiler.py` in the WebODM repo, not guessed):
  GET {WO_URL}/api/projects/{project_pk}/tasks/{task_id}/orthophoto/tiles.json
      -> {"minzoom": int, "maxzoom": int, "bounds": [west, south, east, north]}
  GET {WO_URL}/api/projects/{project_pk}/tasks/{task_id}/orthophoto/tiles/{z}/{x}/{y}.png
      -> PNG/JPEG bytes, or 404 if rio_tiler's own COGReader.tile_exists(z, x, y)
         is False for this raster (i.e. this tile isn't real coverage).

Auth: WebODM's `Tiles`/`TileJson` views require an authenticated request
unless `task.public or task.project.public` (confirmed via
`app/api/tasks.py`'s `get_and_check_task()`). Decision 41: this Actor
receives the same per-user Tapis access token `apply_embed_generate()`
already resolved to invoke the Actor itself, and passes it as a `?jwt=`
query parameter -- WebODM's `JSONWebTokenAuthenticationQS` (`app/api/
authentication.py`) reads exactly that query parameter name, confirmed by
reading its `get_jwt_value()` directly.
"""

import math

import requests
from requests.adapters import HTTPAdapter
from PIL import Image
import io


class TileNotFound(Exception):
    """Raised when WebODM's tiler reports 404 for a (z, x, y) -- this tile
    is genuinely not covered by the raster (rio_tiler's own tile_exists()
    check), not a transient error. Callers should treat this as "skip this
    tile," matching Decision 9's "the full set where tile_exists() is true
    is exhaustive coverage" framing."""


class WebODMTilerError(Exception):
    """Raised on any other non-2xx response from WebODM's tiler endpoints
    (auth failure, task not found, server error) -- a real problem, unlike
    TileNotFound."""


def _tiles_url(webodm_url, project_pk, task_id, tile_type='orthophoto'):
    return f"{webodm_url.rstrip('/')}/api/projects/{project_pk}/tasks/{task_id}/{tile_type}"


def make_session(pool_size):
    """
    Builds a `requests.Session` whose connection pool is sized to
    `pool_size` (embed_generate.main's EMBED_MAX_WORKERS) -- `requests`'
    default `HTTPAdapter` pool (10 connections) would otherwise cap real
    concurrency below whatever the ThreadPoolExecutor is actually running
    once EMBED_MAX_WORKERS is configured above 10. A single shared Session
    reused across worker threads (not one per thread) is `requests`' own
    documented-safe pattern for concurrent requests.
    """
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_tile_coverage(webodm_url, project_pk, task_id, jwt, tile_type='orthophoto', session=None):
    """
    GET .../{tile_type}/tiles.json -- returns (minzoom, maxzoom, bounds)
    where bounds is (west, south, east, north) in EPSG:4326, exactly as
    WebODM's own `TileJson` view returns it (`get_extent(task, tile_type)
    .extent`). Used by `enumerate_tiles()` to compute the candidate (x, y)
    range at the requested zoom -- NOT to decide per-tile coverage (that's
    still `tile_exists()`, checked server-side per Decision 9; a tile inside
    this bbox range can still 404, e.g. a non-rectangular flight footprint).

    `session`: an optional shared `requests.Session` (see `make_session()`)
    -- falls back to a bare `requests.get()` (module-level connection pool)
    when not given, e.g. `enumerate_tiles()`'s own single, one-off call.
    """
    client = session or requests
    resp = client.get(
        _tiles_url(webodm_url, project_pk, task_id, tile_type) + '/tiles.json',
        params={'jwt': jwt},
        timeout=30,
    )
    if resp.status_code == 404:
        raise WebODMTilerError(
            f"WebODM reports no {tile_type} tiles.json for task {task_id} "
            f"(project {project_pk}) -- task/asset may not exist or isn't ready."
        )
    if not resp.ok:
        raise WebODMTilerError(
            f"WebODM tiles.json request failed: {resp.status_code} {resp.text[:300]}"
        )
    data = resp.json()
    return data['minzoom'], data['maxzoom'], tuple(data['bounds'])


def fetch_tile(webodm_url, project_pk, task_id, z, x, y, jwt, tile_type='orthophoto', session=None):
    """
    GET .../{tile_type}/tiles/{z}/{x}/{y}.png -- returns a PIL.Image (RGB)
    on success. Raises TileNotFound on a real 404 (rio_tiler's
    tile_exists() was False for this cell -- Decision 9's exhaustive-
    coverage check, enforced server-side, not reimplemented here) --
    callers should skip this (z, x, y), not treat it as fatal.

    `session`: an optional shared `requests.Session` (see `make_session()`)
    -- `embed_generate.main.run()`'s ThreadPoolExecutor passes one shared
    session sized to EMBED_MAX_WORKERS so concurrent tile fetches reuse
    pooled connections instead of each opening its own.
    """
    client = session or requests
    resp = client.get(
        _tiles_url(webodm_url, project_pk, task_id, tile_type) + f'/tiles/{z}/{x}/{y}.png',
        params={'jwt': jwt},
        timeout=30,
    )
    if resp.status_code == 404:
        raise TileNotFound(f"No {tile_type} tile at z={z} x={x} y={y} for task {task_id}.")
    if not resp.ok:
        raise WebODMTilerError(
            f"WebODM tile fetch failed for z={z} x={x} y={y}: "
            f"{resp.status_code} {resp.text[:300]}"
        )
    return Image.open(io.BytesIO(resp.content)).convert('RGB')


# --- Standard web-mercator (XYZ/slippy-map) tile math -----------------------
# Same convention WebODM's own tiler and every other XYZ tile consumer uses
# (OSM wiki "Slippy map tilenames") -- not a bespoke grid, per Decision 9.

def lonlat_to_tile(lon, lat, zoom):
    """Returns the (x, y) tile containing (lon, lat) at `zoom`."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lonlat(x, y, zoom):
    """Returns the (lon, lat) of the NW corner of tile (x, y) at `zoom`."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def tile_bounds_lonlat(x, y, zoom):
    """Returns (west, south, east, north) for tile (x, y, zoom)."""
    west, north = tile_to_lonlat(x, y, zoom)
    east, south = tile_to_lonlat(x + 1, y + 1, zoom)
    return west, south, east, north


def tile_center_lonlat(x, y, zoom):
    """Returns (lat, lon) of the center of tile (x, y, zoom) -- what
    `clay.embed_tile()` needs for its sin/cos lat/lon conditioning."""
    west, south, east, north = tile_bounds_lonlat(x, y, zoom)
    return (south + north) / 2.0, (west + east) / 2.0


def tile_bounds_wkt(x, y, zoom):
    """Returns a WKT POLYGON (SRID 4326 implied, applied by the caller's SQL
    `ST_GeomFromText(%s, 4326)`) for `tile_grid.bounds` -- a closed ring,
    matching PostGIS's own WKT polygon convention."""
    west, south, east, north = tile_bounds_lonlat(x, y, zoom)
    return (
        f"POLYGON(({west} {south}, {east} {south}, {east} {north}, "
        f"{west} {north}, {west} {south}))"
    )


def meters_per_pixel(zoom, lat, tile_size=256):
    """
    Standard web-mercator ground resolution formula (meters/pixel at a given
    zoom and latitude, for the conventional 256px tile) -- used for
    `tile_observations.pixel_size`, a real computed value grounded in the
    actual tile geometry rather than reusing Clay's own conditioning `gsd`
    (which describes the sensor profile Clay was trained on, not this
    specific raster's real resolution).
    """
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2 ** zoom) * (256 / tile_size)


def candidate_tiles(bounds, zoom):
    """
    Yields every (x, y) tile at `zoom` whose bbox overlaps `bounds` (west,
    south, east, north) -- the CANDIDATE set from bbox math alone. This is
    NOT the final coverage set (Decision 9: that's WebODM's own
    `tile_exists()`, checked per-tile server-side) -- it's the search space
    `enumerate_tiles()` probes via `fetch_tile()`, treating a 404
    (`TileNotFound`) as "not real coverage, skip it." A non-rectangular
    flight footprint means some candidates here will 404 -- expected, not
    an error.
    """
    west, south, east, north = bounds
    x_min, y_min = lonlat_to_tile(west, north, zoom)
    x_max, y_max = lonlat_to_tile(east, south, zoom)
    for x in range(min(x_min, x_max), max(x_min, x_max) + 1):
        for y in range(min(y_min, y_max), max(y_min, y_max) + 1):
            yield x, y
