-- Incremental migration for an already-live embeddingsdb Pod.
--
-- embeddingsdb has no migration framework (schema/embeddingsdb.sql is
-- applied once via `psql $EMBEDDINGSDB_URL -f schema/embeddingsdb.sql`,
-- per that file's own header) -- this is the first schema change since
-- initial creation (Decision 33). schema/embeddingsdb.sql has ALSO been
-- updated in place with this same table, so a fresh deploy from scratch
-- gets it automatically; this file is only for applying the change to a
-- Pod that already exists and already has data, without re-running (and
-- erroring on) every CREATE TABLE that came before it.
--
-- Apply directly against the live Pod:
--   psql $EMBEDDINGSDB_URL -f schema/0002_label_studio_tasks.sql
--
-- See schema/embeddingsdb.sql's own CREATE TABLE label_studio_tasks block
-- for the full rationale (Decision 49 / Decision 29's webhook scope gap).

CREATE TABLE IF NOT EXISTS label_studio_tasks (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    label_studio_project_id integer NOT NULL,
    label_studio_task_id    integer NOT NULL,
    tile_observation_id     uuid NOT NULL REFERENCES tile_observations(id) ON DELETE CASCADE,
    created_at              timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT label_studio_tasks_project_task_uniq
        UNIQUE (label_studio_project_id, label_studio_task_id)
);

COMMENT ON TABLE label_studio_tasks IS
    'WebODM''s own ledger of which (Label Studio project, task) pair maps to '
    'which real tile_observation_id, recorded at import time. Decision 49 -- '
    'closes Decision 29''s webhook scope-validation gap: the webhook handler '
    'looks up this table rather than trusting the payload''s own claimed '
    'tile_observation_id. ON DELETE CASCADE via tile_observations, so a '
    'deleted task/observation cannot leave a stale, still-valid mapping row '
    'behind.';

CREATE INDEX IF NOT EXISTS idx_label_studio_tasks_tile_observation_id
    ON label_studio_tasks (tile_observation_id);
