-- Incremental migration for an already-live embeddingsdb Pod.
--
-- embeddingsdb has no migration framework (schema/embeddingsdb.sql is
-- applied once via `psql $EMBEDDINGSDB_URL -f schema/embeddingsdb.sql`) --
-- schema/embeddingsdb.sql has ALSO been updated in place with this same
-- column, so a fresh deploy from scratch gets it automatically; this file
-- is only for an existing live Pod. See schema/embeddingsdb.sql's own
-- `labels.label_studio_annotation_id` comment for the full rationale
-- (Decision 53 -- the "eraser" flow needs to know exactly which Label
-- Studio annotation to delete, not just which task).
--
-- Apply directly against the live Pod:
--   psql $EMBEDDINGSDB_URL -f schema/0003_labels_annotation_id.sql

ALTER TABLE labels ADD COLUMN IF NOT EXISTS label_studio_annotation_id integer;

COMMENT ON COLUMN labels.label_studio_annotation_id IS
    'The real Label Studio annotation id this row mirrors (Decision 50/53) '
    '-- lets the eraser flow delete the exact right annotation rather than '
    'every annotation on the task. NULL for non-label_studio sources.';
