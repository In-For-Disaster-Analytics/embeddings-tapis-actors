"""
Real Clay v1.5 encoder loading + per-tile inference.

Mirrors `embeddings-research/scripts/clay_embed_sized.py` exactly -- the
validated reference for how this repo's own Phase 1 research actually called
Clay (see that script's own docstring for why it instantiates
`EmbeddingEncoder` directly with explicit size args instead of going through
`claymodel.finetune.embedder.factory.Embedder`, which hardcodes the large
config regardless of the checkpoint passed in). Nothing here is a fresh
reinvention of Clay's conditioning inputs -- band wavelength/mean/std,
sin/cos time-of-year, sin/cos lat/lon, GSD -- all copied from that already-
validated script.

`claymodel` is Clay Foundation's own repo
(`embeddings-research/clay-model-src`, pyproject name `claymodel`), not a
PyPI package -- vendored into this Actor's image via the Dockerfile
(`pip install -e ./clay-model-src`, Decision 39). The checkpoint
(`clay-v1.5.ckpt`, ~4.8 GB) is baked into the image alongside it -- also
Decision 39 -- at `CLAY_CHECKPOINT_PATH`/`CLAY_MODEL_DIR` (env vars, see
`.env.example`).
"""

import logging
import math
import os
import re
import time

import numpy as np
import torch
import yaml
from claymodel.finetune.embedder.factory import EmbeddingEncoder

logger = logging.getLogger(__name__)

# Decision 40: WebODM-sourced visits only, this increment. WebODM's own
# orthophoto tiles are always RGB (Phase 1 research finding: "4/4 checked
# WebODM tasks came back RGB-only despite multispectral-capable sensors" --
# design spec "Phase 1 Research Findings"), and Clay's own metadata.yaml has
# exactly one RGB-only (3-band), no-NIR sensor profile: `linz`. This is not
# a guess -- confirmed by reading every sensor entry in
# clay-model-src/configs/metadata.yaml; `naip` is the next-closest but is
# 4-band (includes NIR), which would mismatch a genuine 3-channel RGB tile.
WEBODM_SENSOR = 'linz'

SIZE_ARGS = {
    'base': dict(dim=768, depth=12, heads=12, dim_head=64, mlp_ratio=4.0),
    'large': dict(dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0),
}

_ENCODER_KEY_RE = re.compile(r'^clay-v(?P<version>[\d.]+)-(?P<size>base|large)-(?P<band_config>.+)$')

# Module-level cache: loading a ~4.8GB checkpoint via torch.load() is
# expensive (seconds, not milliseconds) -- one Actor invocation embeds many
# tiles from the same encoder, so this is loaded once per process, not once
# per tile. Keyed by (size, model_dir, checkpoint_path) in case an Actor
# process ever handles more than one encoder config (it currently doesn't --
# one encoder per invocation message -- but this is cheap insurance against
# reloading the same config twice within one run()).
_encoder_cache = {}


def parse_encoder_key(encoder_key):
    """
    Splits the message payload's `encoder` string (e.g.
    "clay-v1.5-large-rgb") into the four `encoders` table columns:
    (name, version, size, band_config) -- e.g.
    ("clay-v1.5", "1.5", "large", "rgb").

    Raises ValueError on an unrecognized shape rather than guessing --
    only `clay-v<version>-<base|large>-<band_config>` is supported (matches
    every real `encoder` value used anywhere in the design spec/this repo).
    """
    m = _ENCODER_KEY_RE.match(encoder_key)
    if not m:
        raise ValueError(
            f"Unrecognized encoder key {encoder_key!r} -- expected "
            f"'clay-v<version>-<base|large>-<band_config>', e.g. "
            f"'clay-v1.5-large-rgb'."
        )
    version = m.group('version')
    return f"clay-v{version}", version, m.group('size'), m.group('band_config')


def load_encoder(size, model_dir, checkpoint_path):
    """
    Loads Clay's `EmbeddingEncoder` for `size` ('base' | 'large') from
    `checkpoint_path`, directly (bypassing `Embedder`, per this module's own
    docstring), asserting every parameter matches by name and shape -- same
    "refusing to embed with a partially-loaded encoder" guarantee
    `clay_embed_sized.py` already established. Returns an eval-mode,
    no-grad encoder ready for inference.

    Cached at module level per (size, model_dir, checkpoint_path) -- see
    `_encoder_cache` docstring above.
    """
    cache_key = (size, model_dir, checkpoint_path)
    if cache_key in _encoder_cache:
        return _encoder_cache[cache_key]

    if size not in SIZE_ARGS:
        raise ValueError(f"Unsupported Clay size {size!r} -- expected 'base' or 'large'.")

    # Loading the ~4.8GB checkpoint is genuinely slow (seconds to a couple
    # minutes on HPC-node disk) and happens silently on the very first tile
    # of a run -- log it so that first-tile delay isn't mistaken for a hang.
    logger.info("Loading Clay %s encoder from %s ...", size, checkpoint_path)
    start = time.monotonic()

    encoder = EmbeddingEncoder(img_size=256, patch_size=8, **SIZE_ARGS[size])
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    state_dict = {
        re.sub(r'^model\.encoder\.', '', k): v
        for k, v in state_dict.items() if k.startswith('model.encoder')
    }

    matched = mismatched = missing = 0
    with torch.no_grad():
        for name, param in encoder.named_parameters():
            if name in state_dict and param.size() == state_dict[name].size():
                param.data.copy_(state_dict[name])
                matched += 1
            elif name in state_dict:
                mismatched += 1
            else:
                missing += 1
    if mismatched or missing:
        raise RuntimeError(
            f"Refusing to embed with a partially-loaded {size} Clay encoder: "
            f"matched={matched} mismatched={mismatched} missing={missing} "
            f"(checkpoint={checkpoint_path})."
        )

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    _encoder_cache[cache_key] = encoder
    logger.info(
        "Clay %s encoder loaded in %.1fs (matched=%d)",
        size, time.monotonic() - start, matched,
    )
    return encoder


def _normalize_timestamp(date):
    week = date.isocalendar()[1] * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def _normalize_latlon(lat, lon):
    lat_r = lat * np.pi / 180
    lon_r = lon * np.pi / 180
    return (math.sin(lat_r), math.cos(lat_r)), (math.sin(lon_r), math.cos(lon_r))


_sensor_metadata_cache = {}


def _load_sensor_metadata(model_dir, sensor):
    """
    Cached per (model_dir, sensor): `embed_generate.main.run()`'s
    ThreadPoolExecutor calls `embed_tile()` once per tile (up to ~1764 per
    visit), and this file/YAML read never changes within one run -- an
    uncached re-read on every tile is pure-Python work that holds the GIL
    for no reason, cutting into the cross-tile parallelism the thread pool
    is there to provide.
    """
    cache_key = (model_dir, sensor)
    if cache_key in _sensor_metadata_cache:
        return _sensor_metadata_cache[cache_key]
    with open(f'{model_dir}/configs/metadata.yaml', 'r') as f:
        metadata = yaml.safe_load(f)
    sensor_meta = metadata[sensor]
    _sensor_metadata_cache[cache_key] = sensor_meta
    return sensor_meta


def disable_intraop_parallelism():
    """
    embed_generate.main.run() gets its cross-tile parallelism from running
    independent tiles concurrently in a ThreadPoolExecutor (real GIL
    release during both PyTorch's CPU compute and `requests`' network I/O),
    not from PyTorch's own intra-op (matmul/conv) threading within a single
    tile's forward pass. Leaving intra-op threading at its default would
    have each worker thread ALSO spin up its own BLAS-parallel region,
    oversubscribing the Tapis Job's real core allocation
    (`coresPerNode` in ls6/app.json) on top of the worker pool's own
    threads. Must be called once, before any tile is embedded --
    `torch.set_num_threads()` is process-global, not a per-task setting.

    `Dockerfile.embed-generate` also sets `OMP_NUM_THREADS=1`/
    `MKL_NUM_THREADS=1` as a defense-in-depth backstop: some BLAS backends
    read those env vars once at process start rather than respecting
    `torch.set_num_threads()` for every call, so this alone isn't fully
    sufficient on its own.
    """
    torch.set_num_threads(1)


def embed_tile(image, capture_date, center_lat, center_lon, size, model_dir, checkpoint_path):
    """
    Runs Clay v1.5 over one RGB `PIL.Image` tile (any size -- resized to
    256x256 here, matching Clay's own expected input), returning a 1D numpy
    embedding vector (dim 1024 for large / 768 for base -- see `SIZE_ARGS`).

    `capture_date`: a `datetime.date` (from `visits.capture_date`) --
    Clay's time-of-year conditioning only uses the ISO week number, so a
    bare date (no time-of-day) is set to noon, same as
    `clay_embed_sized.py`'s own `CAPTURE_DATE_STR` handling.

    `center_lat`/`center_lon`: the tile's own center (from
    `webodm_client.tile_center_lonlat()`), NOT the task's/site's centroid --
    Clay's positional conditioning is per-tile.

    Sensor profile is fixed to `WEBODM_SENSOR` ('linz') -- see this module's
    docstring for why that's the correct match for WebODM's RGB tiles, not
    a parameter here (Decision 40 scopes this increment to WebODM-sourced
    visits only, which are always this one sensor profile).
    """
    import datetime
    if isinstance(capture_date, datetime.datetime):
        pass
    elif isinstance(capture_date, datetime.date):
        capture_date = datetime.datetime.combine(capture_date, datetime.time(hour=12))
    else:
        raise TypeError(f"capture_date must be a date/datetime, got {type(capture_date)}")

    sensor_meta = _load_sensor_metadata(model_dir, WEBODM_SENSOR)
    waves = np.array([sensor_meta['bands']['wavelength'][b] for b in sensor_meta['band_order']], dtype=np.float32)
    means = np.array([sensor_meta['bands']['mean'][b] for b in sensor_meta['band_order']], dtype=np.float32)
    stds = np.array([sensor_meta['bands']['std'][b] for b in sensor_meta['band_order']], dtype=np.float32)
    gsd_val = sensor_meta['gsd']

    week_norm, hour_norm = _normalize_timestamp(capture_date)
    lat_norm, lon_norm = _normalize_latlon(center_lat, center_lon)
    time_vec = np.hstack([week_norm, hour_norm]).astype(np.float32)
    latlon_vec = np.hstack([lat_norm, lon_norm]).astype(np.float32)

    im = image.convert('RGB').resize((256, 256))
    arr = np.array(im).astype(np.float32)
    arr = (arr - means) / stds
    pixels = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    datacube = {
        'pixels': pixels,
        'time': torch.from_numpy(time_vec).unsqueeze(0),
        'latlon': torch.from_numpy(latlon_vec).unsqueeze(0),
        'waves': torch.from_numpy(waves),
        'gsd': torch.tensor([gsd_val], dtype=torch.float32),
    }

    encoder = load_encoder(size, model_dir, checkpoint_path)
    with torch.no_grad():
        emb = encoder(datacube)
    return emb.squeeze(0).numpy()
