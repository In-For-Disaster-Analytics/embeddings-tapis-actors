# embeddings-tapis-actors

Tapis Actors for the Geospatial Embeddings & Classification System: `embed-generate`
(runs the Clay v1.5 encoder over an orthophoto's or STAC item's tiles) and `model-train`
(trains a classifier over pooled embeddings/covariates/labels). This repo holds the
Actor code only — the WebODM-side plugin that calls these Actors lives in
`coreplugins/embeddings/` in the `WebODM` repo.

## Status: embeddingsdb is live; both Actors have a real message contract; `embed-generate` has real DB writes; registration is dry-run-only

**As of 2026-07-23 (Decision 35):**

- Both Actors' `read_actor_message()` is real: confirmed against Tapis's own Actors
  docs that Abaco injects the invocation payload as the `MSG` environment variable
  (a JSON string) — `main.py` in each Actor reads `os.environ['MSG']` and
  `json.loads()`s it, raising a clear `RuntimeError`/`ValueError` (not a bare
  `KeyError`/`JSONDecodeError`) if it's unset or malformed. Verified end-to-end:
  built both Docker images, ran each with a realistic `MSG` payload, confirmed
  each parses correctly and dispatches into `run()`, failing with the *expected*
  `NotImplementedError` (deeper logic, not a message-parsing crash) — and
  confirmed the unset/malformed-`MSG` failure modes give the new, specific errors.
- `embed_generate/db.py` (new) implements real `psycopg2` writes for
  `write_tile_observation`, `write_embedding`, `write_covariates` — every column
  read directly from `schema/embeddingsdb.sql`, mirroring
  `coreplugins/embeddings/embeddings_client.py`'s style (Decision 34). Verified
  against the live `embeddingsdb` Pod in a transaction that was rolled back
  afterward — confirmed 0 rows persisted.
- `tapis/register_actor.py` (new) registers both Actors, mirroring
  `register_pod.py`'s `--dry-run`/`.env` conventions, adapted for the real
  Actors API (`create_actor`, not `create_pod`). `--dry-run` was actually run and
  confirmed working. The non-dry-run path (and the `docker push` to Docker Hub
  it depends on) was **not** run — no Docker Hub or Tapis credentials exist in
  this environment.
- **Still stubs, unchanged in scope:** `embed_generate`'s tile enumeration, pixel
  fetching, Clay v1.5 inference (`run()` still raises `NotImplementedError`
  before reaching the now-real write functions); all of `model_train` beyond its
  message-contract fix (its own embeddingsdb/MLflow un-stubbing is later work).

See design spec Decision 35 for the full account, including the real facts this
increment is grounded in (confirmed against Tapis's own Actors docs) and every
judgment call flagged along the way (Docker Hub image-name placeholder,
`find_existing_actor_id()`'s assumption about the Actors API's list/lookup
shape, and Decision 30's credential question, still unresolved).

`embeddingsdb` itself is real and running, per Decision 33:

- `embeddingsdb` — **live**, as of 2026-07-23. Postgres 17.5 + PostGIS 3.5.2 + pgvector
  0.8.5, full schema applied (all 13 tables from `schema/embeddingsdb.sql`), verified
  directly via `psql` (extensions active, a real vector distance query, all expected
  tables present).

  **How it was actually provisioned — different from this repo's original plan, and
  worth understanding before touching `tapis/register_pod.py`:** created directly from
  Tapis's own **`17postgis3.5` Pod template** (Postgres 17 + PostGIS 3.5 pre-installed),
  not the custom `pgvector/pgvector:pg16` image `register_pod.py` originally assumed.
  pgvector was then added to the *live* Pod via `tapis/install_pgvector.py`, which runs
  `apt-get install postgresql-17-pgvector` inside the running container through Tapis's
  real `exec_pod_commands` API (confirmed to have root/apt access — not guaranteed by
  Tapis's docs, verified empirically), followed by a plain `CREATE EXTENSION vector`
  over `psql`. `register_pod.py` has been updated to match this real path (`template=`,
  not a custom `image=`) — see its own module docstring for the full history.
- `mlflow` — the MLflow Tracking Server Tapis Pod (Decision 17). Not stood up.
- A Clay v1.5 checkpoint deployed anywhere these Actors can read it. (A checkpoint
  exists at `embeddings-research/models/clay-v1.5.ckpt` in this monorepo-of-repos, used
  for Phase 1 research — it has not been wired into a production path from here.)
- The Actors themselves (`embed-generate`/`model-train`) are not registered with Tapis
  — Actor registration is a different Tapis subsystem from `embeddingsdb`'s Pod
  registration (see "What is NOT in this increment" below) and is not designed here.

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

- No real Clay v1.5 inference — no checkpoint loading, no `claymodel` encoder
  instantiation.
- No real STAC API or WebODM-tiler HTTP calls, no real tile enumeration/pixel
  fetching — `embed_generate/main.py`'s `run()` still raises `NotImplementedError`
  before it would ever reach the now-real `write_*` functions (Decision 35).
- No real `model_train` database or MLflow I/O — only its message contract
  (`read_actor_message()`/`main()`) was fixed this increment (Decision 35);
  un-stubbing the rest of `model_train` (its own DB client module, MLflow
  client, scikit-learn training code) is explicitly separate, later work.
- `tapis/register_actor.py`'s **non-dry-run path was not run** — no Docker Hub
  or Tapis credentials exist in this environment. Neither was `docker push`.
  Only `--dry-run` has been exercised for real.
- `tapis/register_actor.py`'s upsert-by-name logic (`find_existing_actor_id()`)
  assumes a specific `t.actors.list_actors()` return shape that was **not**
  independently confirmed against a live Tapis tenant this session — flagged
  in that function's own docstring as a judgment call, not one of this
  increment's confirmed facts.
- Decision 30's credential question (stored service token vs. per-request JWT
  for the Actors' own async invocation) remains **unresolved** —
  `register_actor.py`'s `default_environment` is static registration-time
  config, not a live credential, and cannot resolve it by itself.
- No tests yet — there is no test framework in this repo; verification this
  increment was done by actually building the Docker images and running them
  (see design spec Decision 35), not via a unit-test suite.

## How a future implementer should proceed

Roughly in dependency order, since each of these unblocks the next:

1. ~~Stand up the `embeddingsdb` Postgres+pgvector Pod and apply its schema.~~ **Done**
   (2026-07-23). Real path taken (see "Status" above for the full story): created from
   Tapis's `17postgis3.5` template, pgvector added live via
   `python tapis/install_pgvector.py --pod-id embeddingsdb`, schema applied via
   `psql "$EMBEDDINGSDB_URL" -f schema/embeddingsdb.sql`. All 13 tables + 11 indexes
   confirmed present. `schema/embeddingsdb.sql`'s own header comment still documents
   the id-type/extension choices made there (`uuid` PKs, `pgvector`, `postgis`) and a
   flagged, real, still-unresolved schema tension around `embeddings.vector`'s fixed
   dimension vs. `encoders`' multi-config registry (Decision 3) — worth resolving
   before a second encoder size/config is ever actually run against this database.
2. Stand up the `mlflow` Pod (backend store can share the `embeddingsdb` Postgres
   instance per the design spec; artifact store on a persistent Tapis Volume).
3. Decide where the Clay v1.5 checkpoint and the `claymodel` encoder package
   (`embeddings-research/clay-model-src`) actually get vendored into this repo's
   runtime — they are not committed here (see `requirements.txt` for why `claymodel`
   isn't a normal pip dependency).
4. ~~Confirm the real Abaco message contract (`MSG` env var) and wire it into both
   Actors' `read_actor_message()`.~~ **Done** (2026-07-23, Decision 35). Confirmed
   against Tapis's own Actors docs, verified end-to-end via Docker (built both
   images, ran each with a realistic payload, confirmed correct parsing + dispatch
   into `run()`, plus the unset/malformed-`MSG` failure modes).
5. ~~Implement `embed_generate/main.py`'s embeddingsdb write functions
   (`write_tile_observation`/`write_embedding`/`write_covariates`).~~ **Done**
   (2026-07-23, Decision 35), via new `embed_generate/db.py`. Verified against the
   live Pod in a rolled-back transaction. **Not done**: everything upstream of
   these writes (tile enumeration, pixel fetching, Clay v1.5 inference) — still
   real `NotImplementedError` stubs; a real Clay v1.5 checkpoint and the
   `claymodel` encoder package (`embeddings-research/clay-model-src`) still need
   to be vendored into this repo's runtime before those can be un-stubbed (see
   `requirements.txt` for why `claymodel` isn't a normal pip dependency).
6. Stand up the `mlflow` Pod (backend store can share the `embeddingsdb` Postgres
   instance per the design spec; artifact store on a persistent Tapis Volume).
7. Implement `model_train/main.py`'s remaining functions (`load_observations`,
   `validate_label_counts`, `split_data`, `tune_hyperparameters`,
   `compute_diagnostics`, `log_to_mlflow`, `write_model_rows`, `run`) against real
   `embeddingsdb` + `mlflow` connections — a new `model_train/db.py` (mirroring
   `embed_generate/db.py`'s style) and an `mlflow` client are both still needed;
   explicitly out of scope for the Decision 35 increment, which only fixed
   `model_train`'s message contract.
8. ~~Write a Tapis Actor registration script for both Actors.~~ **Done**
   (2026-07-23, Decision 35): `tapis/register_actor.py`, mirroring
   `register_pod.py`'s `--dry-run`/upsert conventions, adapted for the real
   Actors API (`create_actor`/`send_message`, confirmed against Tapis's own
   docs — not `create_pod`/`get_pod`). **Not done**: the actual `docker push` to
   public Docker Hub (Abaco's real image-source requirement, also confirmed this
   increment — see the Dockerfile's own note) and the script's non-dry-run
   registration call itself — no Docker Hub or Tapis credentials exist in this
   environment. Once both Actors are registered for real, update
   `WO_EMBEDDINGS_ACTOR_ID`/`WO_MODEL_ACTOR_ID` in WebODM's settings with the
   real `actor_id`s the script prints.
9. Wire `coreplugins/embeddings/embeddings_client.py`'s `queue_embed_generate()`/
   `queue_model_train()` (in the `WebODM` repo — real client module exists per
   Decision 34, but these two functions are still deliberate
   `NotImplementedError` stubs) to actually invoke these Actors by ID, once step
   8's registration has produced real `actor_id`s. This is also where Decision
   30's still-open credential question (stored service token vs. per-request
   JWT for the Actors' own async authorization to embeddingsdb/Tapis) needs to
   finally be resolved — `register_actor.py`'s `default_environment` is static
   registration-time config and cannot resolve it.
10. Add real unit/integration tests once there is real behavior to test (see the
    design spec's **Test Plan** section, items 2-3, for what's expected of
    `embeddings_client.py`'s Actor-invocation payload shape, which these Actors
    must accept).

Update this README's Status section, and the design spec's own Status/Decisions
sections, as each step above actually lands.
