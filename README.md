# embeddings-tapis-actors

Tapis Actors for the Geospatial Embeddings & Classification System: `embed-generate`
(runs the Clay v1.5 encoder over an orthophoto's or STAC item's tiles) and `model-train`
(trains a classifier over pooled embeddings/covariates/labels). This repo holds the
Actor code only — the WebODM-side plugin that calls these Actors lives in
`coreplugins/embeddings/` in the `WebODM` repo.

## Status: skeleton only

**Neither Actor runs real logic yet.** This increment mirrors how
`WebODM/coreplugins/embeddings/` (the sibling plugin these Actors serve) was scaffolded:
real module/function structure with docstrings describing exactly what each piece will
do, `NotImplementedError` bodies, no fabricated business logic. The blocking reason is
the same for both: the infrastructure these Actors depend on does not exist yet —

- `embeddingsdb` — the Postgres+pgvector Tapis Pod (Decision 26). Not stood up.
- `mlflow` — the MLflow Tracking Server Tapis Pod (Decision 17). Not stood up.
- A Clay v1.5 checkpoint deployed anywhere these Actors can read it. (A checkpoint
  exists at `embeddings-research/models/clay-v1.5.ckpt` in this monorepo-of-repos, used
  for Phase 1 research — it has not been wired into a production path from here.)
- The Actors themselves are not registered with Tapis (no `register_pod.py`/actor
  registration script yet — see "Next steps" below).

## Design spec

This repo implements two pieces of the design spec at:

```
WebODM/docs/design/2026-07-22-geospatial-embeddings-classification.md
```

(in the `odm-suite` monorepo-of-repos — the relative path doesn't cross repos, hence
spelling it out). Read that file in full before making non-trivial changes here,
especially these sections: **New Infrastructure** (the table listing `embed-generate`
and `model-train`), **Embeddings DB Schema**, **Tile Coverage**, **Train/Test Split,
Tuning, and Diagnostics**, and **Experiment Tracking and Model Registry: MLflow**.

### `embed-generate` — grounded in Decisions 9, 19, 20, 23, 24, 27

- **Decision 9**: given a task/visit and a zoom level, embeds *every* valid `(z, x, y)`
  tile at that zoom — not a hand-picked subset. Reuses WebODM's own `rio_tiler`-based
  tiler (`app/api/tiler.py`) coverage logic as the source of truth for which tiles exist.
- **Decision 24 / 27**: zoom is locked per `site_id` once set by that site's first visit;
  `site_id` is always user-chosen at trigger time, never inferred. The Actor honors
  whatever zoom/`zoom_override` the invocation payload carries — it does not decide
  zoom policy itself.
- **Decision 19 / 20 / 23**: branches on whether the visit has a `stac_item_id`. If not,
  it fetches pixels via WebODM's own tile endpoint; if so (a genuinely STAC-sourced
  visit, or a `webodm` visit that has been published to the DSO STAC API), it resolves
  the STAC item's asset href and tiles it directly via `rio_tiler` — the exact library
  WebODM's own tiler already depends on.
- Runs the Clay v1.5 encoder (RGB-only, per the approved Phase 1 recommendation) over
  each tile and writes `tile_observations`/`embeddings` rows; separately computes
  `covariates` (elevation/slope/aspect/CHM/NDVI/NDWI) from DSM/DTM/multispectral bands
  where available, inside the same Actor run (Decision 25 — covariates are not a
  separate Actor).

### `model-train` — grounded in Decisions 16, 17, 25, 28

- Takes an arbitrary set of `tile_observation_id`s (spanning any tasks/projects,
  Decision 6/8) and a `task_type` (v1 implements `classification` only).
- **Decision 25 / 28**: enforces the minimum-label floor before training — 30 labeled
  observations total, and at least 5 examples per distinct `label_classes` value
  actually present in the selection — both checked before any split/fit happens.
- **Decision 16**: real train/test methodology via `scikit-learn` — `train_test_split`
  (70/30, stratified), `GridSearchCV` + `StratifiedKFold(n_splits=10)` for tuning
  (`n_estimators`, `max_depth`), diagnostics via OOB error/`feature_importances_`,
  `roc_auc_score`/`roc_curve`, `confusion_matrix`, and `calibration_curve`. Split
  strategy is a named, selectable choice — `random_stratified`, `spatial_block` (holds
  out contiguous `tile_grid` regions), or `temporal_holdout` (holds out a whole
  `visit`) — because naive random splitting over spatially autocorrelated tiles risks
  leakage.
- **Decision 17**: wraps training in `mlflow.start_run()`, logs params/metrics/
  diagnostic-plot artifacts, and registers the fitted estimator via
  `mlflow.sklearn.log_model()`. `embeddingsdb` keeps only a `mlflow_run_id` pointer —
  it does not duplicate what MLflow already tracks.
- Writes `models`/`model_inputs`/`predictions` rows to `embeddingsdb` on completion.

## What is NOT in this increment

- No real database I/O (`embeddingsdb` connection code is not written — there is
  nothing running to connect to yet).
- No real MLflow calls.
- No real Clay v1.5 inference — no checkpoint loading, no `claymodel` encoder
  instantiation.
- No real STAC API or WebODM-tiler HTTP calls.
- No Tapis Actor registration script (the `embed-generate`/`model-train` equivalent of
  `label-studio-tapis-auth/tapis/register_pod.py` — note that script registers a Tapis
  **Pod**, not an **Actor**; Actor registration is a different Tapis API and is not
  designed here).
- No tests yet — there is no real behavior to test.

## How a future implementer should proceed

Roughly in dependency order, since each of these unblocks the next:

1. Stand up the `embeddingsdb` Postgres+pgvector Pod and apply the schema described in
   the design spec's **Embeddings DB Schema** section (`sites`, `tile_grid`, `visits`,
   `tile_observations`, `encoders`, `embeddings`, `covariates`, `label_classes`,
   `labels`, `model_algorithms`, `models`, `model_inputs`, `predictions`).
2. Stand up the `mlflow` Pod (backend store can share the `embeddingsdb` Postgres
   instance per the design spec; artifact store on a persistent Tapis Volume).
3. Decide where the Clay v1.5 checkpoint and the `claymodel` encoder package
   (`embeddings-research/clay-model-src`) actually get vendored into this repo's
   runtime — they are not committed here (see `requirements.txt` for why `claymodel`
   isn't a normal pip dependency).
4. Implement `embed_generate/main.py`'s functions against a real `embeddingsdb`
   connection and a real Clay v1.5 checkpoint, function by function, replacing each
   `NotImplementedError`.
5. Implement `model_train/main.py`'s functions the same way, against real
   `embeddingsdb` + `mlflow` connections.
6. Write a Tapis Actor registration script for both Actors (mirroring
   `label-studio-tapis-auth/tapis/register_pod.py`'s `--dry-run`/upsert conventions,
   adjusted for Tapis's Actor registration API rather than its Pods API — these are
   different Tapis subsystems, do not assume the same script works unmodified).
7. Wire `coreplugins/embeddings/embeddings_client.py` (in the `WebODM` repo, not yet
   created — see that repo's own design spec "Files Likely Affected") to actually
   invoke these Actors by ID.
8. Add real unit/integration tests once there is real behavior to test (see the design
   spec's **Test Plan** section, items 2-3, for what's expected of
   `embeddings_client.py`'s Actor-invocation payload shape, which these Actors must
   accept).

Update this README's Status section, and the design spec's own Status/Decisions
sections, as each step above actually lands.
