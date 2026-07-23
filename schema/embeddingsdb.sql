-- embeddingsdb -- Postgres+pgvector schema
--
-- Grounded in the design spec's "Embeddings DB Schema" section:
--   WebODM/docs/design/2026-07-22-geospatial-embeddings-classification.md
-- (in the odm-suite monorepo-of-repos). Read that section (and the cited
-- Decisions below) before changing this file -- most non-obvious choices
-- here trace back to a specific, already-approved Decision, not a fresh
-- judgment call made while writing SQL.
--
-- This is the schema ONLY. It is applied separately from the Tapis Pod
-- itself (tapis/register_pod.py creates the Pod; this file is applied
-- afterwards via `psql $EMBEDDINGSDB_URL -f schema/embeddingsdb.sql`) --
-- see this repo's README "Next steps" for why these are two separate
-- steps rather than one.
--
-- Conventions used throughout this file:
--   * Primary keys are `uuid DEFAULT gen_random_uuid()`. gen_random_uuid()
--     has been built into Postgres core since PG13 (no `pgcrypto` extension
--     needed) -- confirmed against the Postgres 16 release this schema
--     targets (pgvector/pgvector:pg16, see tapis/register_pod.py). uuid was
--     chosen over bigserial specifically because `visits.webodm_task_id`
--     must match WebODM's own `Task.id` type -- `app/models/task.py`:
--     `id = models.UUIDField(primary_key=True, default=uuid_module.uuid4,
--     ...)` -- so a bigserial embeddingsdb PK would force an awkward
--     uuid<->bigint mapping at exactly the one FK that crosses into
--     WebODM's own domain. Using uuid everywhere keeps every table
--     consistent rather than mixing key types table-by-table. This also
--     matches how the Actor stub code (embed_generate/main.py,
--     model_train/main.py) already describes every id in its example
--     invocation payloads as an opaque string ("...", not an integer).
--   * `jsonb` for JSON-shaped columns (`models.split_params`,
--     `model_algorithms.hyperparameters_schema`) -- queryable and indexable,
--     unlike plain `json`.
--   * Timestamps are `timestamptz`, defaulting to `now()` where the design
--     spec doesn't otherwise specify a value (e.g. `created_at` columns not
--     explicitly named in the schema sketch, but implied by "created_by,
--     created_at" on `label_classes`/`labels`).

CREATE EXTENSION IF NOT EXISTS vector;

-- PostGIS is required for tile_grid.bounds (geometry column, below). This
-- matches WebODM's own Postgres+PostGIS convention (see WebODM/CLAUDE.md:
-- "Uses PostgreSQL with PostGIS extensions for spatial operations") -- this
-- is a different Postgres instance from webodm_dev (Decision 26: zero
-- WebODM-database schema changes; embeddingsdb is fully decoupled), but the
-- same extension requirement applies here independently, because tile_grid
-- needs real spatial geometry, not just a bare bounding-box array.
CREATE EXTENSION IF NOT EXISTS postgis;


-- ---------------------------------------------------------------------------
-- sites
-- ---------------------------------------------------------------------------
-- A named, real-world location a user has explicitly created or chosen --
-- NEVER inferred from task/project metadata (Decision 27). tile_grid's zoom
-- is locked per site (Decision 24), so `sites` is the anchor that makes
-- cross-visit change detection possible at all.
CREATE TABLE sites (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE sites IS
    'A real-world location, explicitly created/chosen by a user at embed-generate '
    'time -- never inferred from task/project metadata. Decision 27.';


-- ---------------------------------------------------------------------------
-- tile_grid
-- ---------------------------------------------------------------------------
-- A STABLE spatial cell for a site, independent of any one survey date.
-- (z, x, y) are WebODM's OWN tile coordinates (a standard, fixed, global
-- web-mercator grid -- Decision 9), reused directly rather than inventing a
-- new grid (see design spec "Tile Coverage").
CREATE TABLE tile_grid (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id     uuid NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    z           integer NOT NULL,
    x           integer NOT NULL,
    y           integer NOT NULL,
    -- Tile footprint in web-mercator-derived lat/lon (SRID 4326), matching
    -- the (z, x, y) XYZ tiling convention WebODM's own tiler already uses.
    -- Requires the `postgis` extension enabled above.
    bounds      geometry(Polygon, 4326),
    created_at  timestamptz NOT NULL DEFAULT now(),

    -- Decision 24: zoom is locked per site, and matching across visits is an
    -- EXACT (site_id, z, x, y) lookup -- "no IoU threshold, no centroid-in-
    -- cell heuristic, no spatial join at all." That only holds if a given
    -- (site_id, z, x, y) can never be represented by more than one row.
    CONSTRAINT tile_grid_site_zxy_uniq UNIQUE (site_id, z, x, y)
);

COMMENT ON TABLE tile_grid IS
    'A stable spatial cell for a site, independent of survey date. (z, x, y) '
    'are WebODM''s own tile coordinates, reused directly (Decision 9). '
    'UNIQUE(site_id, z, x, y) makes cross-visit matching an exact lookup, '
    'per Decision 24.';
COMMENT ON COLUMN tile_grid.bounds IS
    'Tile footprint geometry (SRID 4326). Requires CREATE EXTENSION postgis.';


-- ---------------------------------------------------------------------------
-- visits
-- ---------------------------------------------------------------------------
-- ONE survey/capture event at a site -- what actually varies over time,
-- distinct from the stable tile_grid cell it's observed against.
CREATE TABLE visits (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id             uuid NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    source              text NOT NULL
                          CHECK (source IN ('webodm', 'stac', 'openaerialmap', 'dronedb')),

    -- Decision 26: the webodm_task_id mapping lives ENTIRELY inside
    -- embeddingsdb -- zero WebODM-database schema changes. Nullable because
    -- non-webodm visits (source='stac'/'openaerialmap'/'dronedb') have none.
    -- References app/models/task.py's Task.id (a UUIDField), NOT a Django
    -- FK -- webodm_dev and embeddingsdb are separate Postgres instances, so
    -- this cannot be a real foreign key; referential integrity across the
    -- two databases is enforced by the Django post_delete signal described
    -- in Decision 26, not by Postgres itself.
    webodm_task_id      uuid,

    -- Decision 19/20/23: nullable, populated independently of `source` --
    -- a source='webodm' visit gets these once its task is opted into
    -- "Publish to STAC" (Decision 20); a source='stac' visit has these from
    -- import with no webodm_task_id at all. Both stac_collection_id and
    -- stac_item_id are plain text ids from the DSO STAC API
    -- (modflow-suite/stac-platform), not local FKs -- that catalog is a
    -- separate external service, not a table in this database.
    stac_collection_id  text,
    stac_item_id        text,

    capture_date        date,
    created_at          timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE visits IS
    'One survey/capture event at a site. source distinguishes true origin; '
    'webodm_task_id/stac_collection_id/stac_item_id are independent, '
    'nullable pointers -- see Decisions 19, 20, 23, 26.';
COMMENT ON COLUMN visits.webodm_task_id IS
    'Maps to WebODM''s own Task.id (a UUIDField, app/models/task.py). Lives '
    'here, not as a field on WebODM''s Task model -- Decision 26 (zero '
    'WebODM-database schema changes). Not a real FK: separate Postgres '
    'instance from webodm_dev.';
COMMENT ON COLUMN visits.stac_collection_id IS
    'DSO STAC API (modflow-suite/stac-platform) collection id. Populated '
    'for source=''stac'' visits, and for source=''webodm'' visits once '
    'published (Decision 20). Not a local FK -- external catalog.';
COMMENT ON COLUMN visits.stac_item_id IS
    'DSO STAC API item id. See stac_collection_id. embed-generate branches '
    'on whether this is set (Decision 20''s refinement of Decision 19), not '
    'on source directly.';


-- ---------------------------------------------------------------------------
-- tile_observations
-- ---------------------------------------------------------------------------
-- ONE observation of a grid cell, at one point in time -- what embeddings /
-- covariates / labels / model_inputs actually key off, not a flat "tile."
CREATE TABLE tile_observations (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tile_grid_id  uuid NOT NULL REFERENCES tile_grid(id) ON DELETE CASCADE,
    visit_id      uuid NOT NULL REFERENCES visits(id) ON DELETE CASCADE,
    pixel_size    double precision,
    created_at    timestamptz NOT NULL DEFAULT now(),

    -- Architect review feedback (spec review pass, Decisions 26-31): one
    -- observation per (tile_grid cell, visit) -- embed-generate's own
    -- write_tile_observation() is described as an upsert keyed on exactly
    -- this pair (see embed_generate/main.py), which only behaves like an
    -- upsert if the pair is actually unique at the database level.
    CONSTRAINT tile_observations_grid_visit_uniq UNIQUE (tile_grid_id, visit_id)
);

COMMENT ON TABLE tile_observations IS
    'One observation of a tile_grid cell at one visit (point in time). '
    'embeddings/covariates/labels/model_inputs all key off this, not off a '
    'flat tile. UNIQUE(tile_grid_id, visit_id) per spec-review feedback -- '
    'embed-generate upserts on exactly this pair.';


-- ---------------------------------------------------------------------------
-- encoders
-- ---------------------------------------------------------------------------
-- Decision 3: each distinct band/modality configuration is its own row --
-- Clay-base != Clay-large != Clay+DSM != Galileo+DSM produce non-
-- interchangeable vectors (Phase 1 bake-off finding).
CREATE TABLE encoders (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name         text NOT NULL,        -- e.g. 'clay-v1.5'
    version      text NOT NULL,        -- e.g. '1.5'
    size         text NOT NULL,        -- e.g. 'base' | 'large'
    band_config  text NOT NULL,        -- e.g. 'rgb' | 'rgb+dsm'
    created_at   timestamptz NOT NULL DEFAULT now(),

    -- Architect review feedback: prevents silently re-registering what is
    -- semantically the same encoder config as a second row (which would
    -- split one config's embeddings across two encoder_ids for no reason).
    CONSTRAINT encoders_name_version_size_band_uniq
        UNIQUE (name, version, size, band_config)
);

COMMENT ON TABLE encoders IS
    'Registry of distinct encoder configurations (Decision 3) -- e.g. '
    '"clay-v1.5-large-rgb". UNIQUE(name, version, size, band_config) per '
    'spec-review feedback, so the same config is never registered twice.';


-- ---------------------------------------------------------------------------
-- embeddings
-- ---------------------------------------------------------------------------
-- KNOWN SCHEMA TENSION, flagged rather than silently resolved (see comment
-- on the `vector` column below): a single `vector(N)` column cannot hold
-- multiple encoders' output dimensions at once. This is fine as long as
-- exactly one encoder config is live at a time, but the `encoders` registry
-- (Decision 3) explicitly anticipates MULTIPLE simultaneous configs (base
-- vs. large, RGB vs. RGB+DSM) -- a real mismatch between "one N-dim column"
-- and "many possible encoders" that this migration does not resolve. Flagged
-- here as a follow-up decision worth making explicitly (e.g. per-encoder
-- embeddings tables, or a variable-length representation) before a second
-- encoder config is ever actually run against production data.
CREATE TABLE embeddings (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tile_observation_id  uuid NOT NULL REFERENCES tile_observations(id) ON DELETE CASCADE,
    encoder_id           uuid NOT NULL REFERENCES encoders(id),
    -- Placeholder dimension: 1024, matching Clay v1.5 LARGE's embedding size
    -- (dim=1024, depth=24, heads=16) -- confirmed against
    -- embeddings-research/scripts/clay_embed_sized.py's SIZE_ARGS dict,
    -- which also documents Clay v1.5 BASE as dim=768. MUST match whichever
    -- encoder actually produced the vector (see encoders.size above) --
    -- this column's dimension is fixed at table-creation time, so a
    -- large-vs-base mismatch is a real, not-yet-resolved constraint (see the
    -- table-level comment above and Decision 3's per-encoder-config table).
    vector               vector(1024) NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE embeddings IS
    'One Clay v1.5 embedding vector for one tile_observation, from one '
    'encoder config (encoders FK, Decision 3). See the table-creation '
    'comment above for a real, unresolved tension between this table''s '
    'single vector(N) column and encoders'' multi-config design.';
COMMENT ON COLUMN embeddings.vector IS
    'pgvector column, dimension 1024 (Clay v1.5 LARGE, per '
    'embeddings-research/scripts/clay_embed_sized.py SIZE_ARGS). Clay v1.5 '
    'BASE uses dim=768 -- a base-encoder row would NOT fit this column as-is. '
    'This is a known limitation, not an oversight: a single vector(N) column '
    'cannot represent multiple encoder dimensions simultaneously. Flagged as '
    'a follow-up decision (e.g. one embeddings table per distinct dimension, '
    'or a separate embeddings_base/embeddings_large split) before a second '
    'encoder size is actually run in production -- not resolved here.';


-- ---------------------------------------------------------------------------
-- covariates
-- ---------------------------------------------------------------------------
-- Decision 2: stored separately from embeddings, never fused -- injecting
-- DSM elevation into the embedding vector hurt accuracy in every Phase 1
-- configuration tried. Plain raster-derived features, independent of any
-- encoder.
CREATE TABLE covariates (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tile_observation_id  uuid NOT NULL REFERENCES tile_observations(id) ON DELETE CASCADE,
    elevation            double precision,
    slope                double precision,
    aspect               double precision,
    chm                  double precision,
    ndvi                 double precision,
    ndwi                 double precision,
    created_at           timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE covariates IS
    'Raster-derived features (elevation/slope/aspect/CHM/NDVI/NDWI), stored '
    'separately from embeddings and never fused into the encoder input -- '
    'Decision 2. Populated only where DSM/DTM/multispectral bands exist for '
    'the source visit; absent, never backfilled or faked (Decision 25).';


-- ---------------------------------------------------------------------------
-- label_classes
-- ---------------------------------------------------------------------------
-- Decision 12: extensible controlled vocabulary for category labels.
-- Phase 1's 7 land-cover classes ship as instance-wide defaults
-- (site_id IS NULL), not a hardcoded ceiling.
CREATE TABLE label_classes (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id       uuid REFERENCES sites(id) ON DELETE CASCADE,  -- NULL = instance-wide default
    value         text NOT NULL,
    display_name  text NOT NULL,
    color_hex     text,
    created_by    text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE label_classes IS
    'Controlled vocabulary for category labels (Decision 12). site_id NULL '
    'rows are instance-wide defaults (Phase 1''s 7 land-cover classes), any '
    'site can add its own on top.';
COMMENT ON COLUMN label_classes.site_id IS
    'NULL = instance-wide default class, visible to every site. Non-NULL = '
    'a class added by/for that specific site.';


-- ---------------------------------------------------------------------------
-- labels
-- ---------------------------------------------------------------------------
-- Decision 7: value_type generalizes labels beyond classification (category
-- vs. continuous), tied to the tile_observation (not to any one vector) so
-- labels survive encoder swaps.
CREATE TABLE labels (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tile_observation_id  uuid NOT NULL REFERENCES tile_observations(id) ON DELETE CASCADE,
    value_type           text NOT NULL CHECK (value_type IN ('category', 'continuous')),
    -- Stored as text regardless of value_type: category values reference
    -- label_classes.value (enforced at the application layer -- import/label
    -- endpoints -- NOT a DB FK, since continuous labels have no taxonomy row
    -- to point at); continuous values are cast at the application layer.
    value                text NOT NULL,
    source               text NOT NULL CHECK (source IN ('label_studio', 'manual', 'geojson_import')),
    created_by           text,
    created_at           timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE labels IS
    'A label for one tile_observation. value_type (category/continuous) and '
    'source track provenance and shape -- Decision 7. For value_type='
    '''category'', value SHOULD reference label_classes.value, enforced at '
    'the application layer (Decision 12), not a DB FK, since continuous '
    'labels have no taxonomy row to reference.';


-- ---------------------------------------------------------------------------
-- model_algorithms
-- ---------------------------------------------------------------------------
-- Decision 15: developer-extensible registry of what model-train's code
-- actually implements -- distinct in kind from label_classes (user-
-- extensible). Adding a row requires adding real training code to the Actor.
CREATE TABLE model_algorithms (
    id                        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type                 text NOT NULL,   -- e.g. 'classification' (v1) | 'regression' | 'change_detection'
    key                       text NOT NULL,   -- e.g. 'random_forest'
    display_name              text NOT NULL,
    hyperparameters_schema    jsonb,           -- future hook (e.g. n_estimators for RF); v1 exposes none in the UI
    is_default                boolean NOT NULL DEFAULT false,
    created_at                timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT model_algorithms_task_type_key_uniq UNIQUE (task_type, key)
);

COMMENT ON TABLE model_algorithms IS
    'Registry of implemented (task_type, algorithm) pairs -- Decision 15. '
    'Developer-extensible only: adding a row requires adding the '
    'corresponding training function to model-train. v1 ships one default '
    'algorithm per implemented task_type.';
COMMENT ON COLUMN model_algorithms.hyperparameters_schema IS
    'Future hook for UI-exposed tunable hyperparameters (e.g. n_estimators '
    'for random_forest). v1 exposes none in the UI -- one sensible default '
    'per task_type, per Decision 15.';


-- ---------------------------------------------------------------------------
-- models
-- ---------------------------------------------------------------------------
-- Decision 8: no project_id -- model_inputs is the explicit join to whatever
-- tile_observations (and therefore whatever tasks/projects) a model actually
-- trained on, since training pools across projects (Decision 6).
-- Decision 17: MLflow is the system of record for params/metrics/artifacts --
-- mlflow_run_id is a thin pointer, NOT a duplicate model_metrics table.
CREATE TABLE models (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type       text NOT NULL,   -- 'classification' (v1) | 'regression' | 'change_detection' (schema-ready, not implemented)
    algorithm       text NOT NULL,   -- references model_algorithms.key (application-layer, not a DB FK -- see note below)
    encoder_id      uuid NOT NULL REFERENCES encoders(id),
    -- Decision 17: pointer into MLflow's own tracking store. NOT NULL is
    -- deliberately not enforced here -- a models row is written only once
    -- log_to_mlflow() has returned a real run_id (see model_train/main.py's
    -- write_model_rows()), so in practice this is always populated by the
    -- time a row exists, but the column stays nullable rather than
    -- pretending Postgres can enforce an ordering across two databases.
    mlflow_run_id   text,
    -- Decision 16: named, selectable split strategy -- naive random
    -- splitting over spatially autocorrelated tiles risks leakage.
    split_strategy  text NOT NULL CHECK (split_strategy IN ('random_stratified', 'spatial_block', 'temporal_holdout')),
    split_params    jsonb,
    trained_at      timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE models IS
    'A trained model -- thin join between "predicts on our tile_observations" '
    'and "the actual tracked MLflow run" (mlflow_run_id). No project_id '
    '(Decision 8) -- see model_inputs for the explicit set of observations '
    'trained on, which may span any tasks/projects (Decision 6).';
COMMENT ON COLUMN models.algorithm IS
    'References model_algorithms.key. Not a DB FK to keep model_algorithms '
    'purely a developer-maintained registry (Decision 15) -- the same '
    'pattern labels.value uses for label_classes.value.';
COMMENT ON COLUMN models.mlflow_run_id IS
    'Pointer into MLflow''s tracking store (Decision 17). Params, metrics, '
    'and the serialized model artifact live in MLflow, not duplicated as '
    'embeddingsdb columns -- explicitly no model_metrics table.';
COMMENT ON COLUMN models.split_strategy IS
    'random_stratified | spatial_block | temporal_holdout -- a real, named '
    'choice (Decision 16), not always plain random, because naive random '
    'splitting over spatially autocorrelated tiles risks leakage.';


-- ---------------------------------------------------------------------------
-- model_inputs
-- ---------------------------------------------------------------------------
-- Decision 16: explicit set of observations a model trained on, with which
-- side of the train/test split each landed on -- auditable, not just
-- asserted. pair_tile_observation_id (Decision 16 / change-detection
-- schema-readiness, Decision 7) links two observations of the same
-- tile_grid_id at different dates, for change_detection models only.
CREATE TABLE model_inputs (
    id                          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id                    uuid NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    tile_observation_id         uuid NOT NULL REFERENCES tile_observations(id) ON DELETE CASCADE,
    -- Self-referencing tile_observations, NOT model_inputs -- set only for
    -- change_detection models, linking this row's tile_observation to
    -- another observation of the SAME tile_grid_id at a different visit/date.
    pair_tile_observation_id    uuid REFERENCES tile_observations(id) ON DELETE CASCADE,
    split                       text NOT NULL CHECK (split IN ('train', 'test')),
    created_at                  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE model_inputs IS
    'Explicit set of tile_observations a model trained on, regardless of '
    'source task/project (Decision 6/8). split records which side of the '
    'train/test holdout each observation landed on, per Decision 16 -- '
    'auditable, not just asserted.';
COMMENT ON COLUMN model_inputs.pair_tile_observation_id IS
    'Set only for change_detection models: links to another '
    'tile_observations row for the same tile_grid_id at a different date. '
    'NULL for classification/regression models (v1). Decision 16.';


-- ---------------------------------------------------------------------------
-- predictions
-- ---------------------------------------------------------------------------
CREATE TABLE predictions (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tile_observation_id  uuid NOT NULL REFERENCES tile_observations(id) ON DELETE CASCADE,
    model_id             uuid NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    value_type           text NOT NULL CHECK (value_type IN ('category', 'continuous')),
    value                text NOT NULL,
    confidence           double precision,
    created_at           timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE predictions IS
    'A model''s prediction for one tile_observation. value_type mirrors '
    'labels.value_type (Decision 7). confidence backs the low-confidence '
    'review flow (Decision 13/14) -- amber-bordered tiles below threshold.';


-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
-- Not exhaustive query-tuning -- just the lookups explicitly implied by the
-- design spec's own data-flow descriptions (workspace/browse pooling across
-- tasks, embed-status polling, per-site predictions map).
CREATE INDEX idx_tile_grid_site_id ON tile_grid (site_id);
CREATE INDEX idx_visits_site_id ON visits (site_id);
CREATE INDEX idx_visits_webodm_task_id ON visits (webodm_task_id);
CREATE INDEX idx_tile_observations_visit_id ON tile_observations (visit_id);
CREATE INDEX idx_tile_observations_tile_grid_id ON tile_observations (tile_grid_id);
CREATE INDEX idx_embeddings_tile_observation_id ON embeddings (tile_observation_id);
CREATE INDEX idx_covariates_tile_observation_id ON covariates (tile_observation_id);
CREATE INDEX idx_labels_tile_observation_id ON labels (tile_observation_id);
CREATE INDEX idx_model_inputs_model_id ON model_inputs (model_id);
CREATE INDEX idx_predictions_model_id ON predictions (model_id);
CREATE INDEX idx_predictions_tile_observation_id ON predictions (tile_observation_id);

-- pgvector ANN index, deferred: an ivfflat/hnsw index needs a representative
-- data sample to build well (ivfflat's `lists` parameter in particular is
-- tuned against real row counts) and embeddingsdb has zero rows at schema-
-- creation time (see README "Status: skeleton only"). Add one of:
--   CREATE INDEX ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
--   CREATE INDEX ON embeddings USING hnsw (vector vector_cosine_ops);
-- once embed-generate has actually populated this table at realistic scale.
