"""
model-train Tapis Actor -- entrypoint.

SKELETON ONLY. Every function below has a docstring describing what it will
do once the supporting infrastructure (embeddingsdb Pod, mlflow Pod, Tapis
Actor registration) exists, and a `NotImplementedError` body -- no
fabricated logic. See design spec:
WebODM/docs/design/2026-07-22-geospatial-embeddings-classification.md
(in the odm-suite monorepo-of-repos), Decisions 16, 17, 25, 28, and the
"Train/Test Split, Tuning, and Diagnostics" / "Experiment Tracking and Model
Registry: MLflow" sections.

What this Actor is meant to do, end to end, once implemented
--------------------------------------------------------------
Given an invocation payload naming a `task_type` (v1 implements
`classification` only -- other values are rejected upstream by WebODM's
`POST .../workspace/train` returning 501, per the design spec), an
`algorithm` (from `model_algorithms`, Decision 15), an `encoder_id`, a
`split_strategy`, and an arbitrary set of `tile_observation_id`s that may
span any combination of WebODM tasks/projects (Decision 6/8):

1. Load and join `embeddings` ⨝ `covariates` ⨝ `labels` for exactly those
   `tile_observation_id`s from embeddingsdb.
2. Enforce the minimum-label floor BEFORE splitting or fitting anything
   (Decision 25/28): at least 30 labeled observations total, AND at least 5
   examples per distinct `label_classes` value actually present in the
   selection. Fail loudly (not silently proceed with too little data) if
   either floor isn't met.
3. Split into train/test per `split_strategy` (Decision 16):
   - `random_stratified`: `sklearn.model_selection.train_test_split(...,
     test_size=0.3, stratify=labels)` -- for small pilot datasets where a
     spatial/temporal holdout would leave too little data on either side.
   - `spatial_block`: hold out whole contiguous `tile_grid` regions, not
     scattered individual tiles -- avoids leakage from spatially
     autocorrelated adjacent tiles.
   - `temporal_holdout`: hold out an entire `visit` (mirrors Wing et al.
     2021's year-by-year validation).
   Record which side each observation landed on in `model_inputs.split` --
   auditable after the fact, not just asserted.
4. Tune hyperparameters via `GridSearchCV` with
   `StratifiedKFold(n_splits=10)`, run ONLY on the training split -- the test
   set is never touched until final evaluation.
5. Fit the final estimator (v1 default: `RandomForestClassifier(oob_score=
   True)`) with the winning hyperparameters.
6. Compute diagnostics: OOB error, `.feature_importances_`,
   `sklearn.metrics.roc_auc_score`/`roc_curve`, `confusion_matrix`,
   `sklearn.calibration.calibration_curve` -- evaluated on the held-out test
   split.
7. Log everything to MLflow (Decision 17): wrap the whole run in
   `mlflow.start_run()`; log params (search space, winning hyperparameters),
   metrics (per-fold CV scores, ROC-AUC, OOB error), and diagnostic plots as
   artifacts; call `mlflow.sklearn.log_model(...)` to register the fitted
   estimator in MLflow's Model Registry.
8. Write `models` (with `mlflow_run_id` pointing at the MLflow run),
   `model_inputs` (one row per observation, with its `split`), and
   `predictions` (for the test split, at minimum) rows to embeddingsdb.

Explicitly NOT implemented in this increment
---------------------------------------------
- No embeddingsdb connection (the Pod doesn't exist yet).
- No MLflow connection (the Pod doesn't exist yet).
- No real scikit-learn training code -- function bodies below are stubs.
- No Tapis service-token handling (Decision 30) -- credential plumbing for
  this Actor's async invocation is unresolved, see this repo's README.
"""

import os


def main():
    """
    Actor entrypoint. A real Tapis Actor reads its invocation message (JSON,
    conventionally via the `MSG` environment variable or a mounted file --
    exact mechanism depends on how this Actor is ultimately registered, see
    this repo's README "Next steps" item 6) and dispatches to `run()`.
    """
    message = read_actor_message()
    run(message)


def read_actor_message():
    """
    Parse this invocation's message payload.

    Expected shape (per the design spec's `POST .../workspace/train`
    endpoint, which is what queues this Actor):
        {
            "task_type": "classification",
            "algorithm": "random_forest",
            "encoder": "clay-v1.5-large-rgb",
            "tile_observation_ids": ["...", "...", ...],
            "split_strategy": "spatial_block",
        }
    """
    raise NotImplementedError(
        "Actor message parsing is not implemented yet -- no Tapis Actor "
        "registration exists for this Actor, so the real invocation "
        "envelope (env var vs. mounted file, message schema) is unconfirmed. "
        "See design spec 'API Endpoints' > POST .../workspace/train."
    )


def load_observations(tile_observation_ids, encoder_id):
    """
    Join embeddings <-> covariates <-> labels in embeddingsdb for exactly
    the given `tile_observation_id`s, regardless of which webodm_task_id(s)
    or project(s) they came from (Decision 6/8). Returns feature matrix X,
    label vector y, and enough metadata (tile_grid_id, visit_id per row) for
    `spatial_block`/`temporal_holdout` splitting to be possible.
    """
    raise NotImplementedError(
        "Loading pooled observations is not implemented yet -- depends on "
        "the embeddingsdb Pod, which does not exist yet. See design spec "
        "Embeddings DB Schema."
    )


def validate_label_counts(y):
    """
    Decision 25/28: raise before any split/fit happens if `y` has fewer than
    30 total labeled observations, or fewer than 5 examples for any distinct
    label_classes value present. The caller (WebODM's plugin) is expected to
    surface this as a 400 with an actionable message -- this function's job
    is only to detect the violation, not to format the HTTP response.
    """
    raise NotImplementedError(
        "Label-count validation is not implemented yet. See design spec "
        "Decisions 25 and 28, and Test Plan item 7."
    )


def split_data(X, y, metadata, split_strategy, split_params):
    """
    Decision 16: dispatch to the requested split strategy --
    'random_stratified' (sklearn.model_selection.train_test_split,
    test_size=0.3, stratify=y), 'spatial_block' (hold out contiguous
    tile_grid regions using `metadata`'s tile_grid_id), or 'temporal_holdout'
    (hold out an entire visit using `metadata`'s visit_id). Returns
    train/test indices plus a per-observation split label ('train'/'test')
    suitable for writing to `model_inputs.split`.
    """
    raise NotImplementedError(
        "Split-strategy dispatch is not implemented yet. See design spec "
        "'Train/Test Split, Tuning, and Diagnostics' and Test Plan item 11 "
        "(the spatial/temporal leakage check)."
    )


def tune_hyperparameters(X_train, y_train, algorithm):
    """
    Decision 16: run GridSearchCV with StratifiedKFold(n_splits=10) on the
    TRAINING split only, for the given `algorithm` (from `model_algorithms`,
    v1 default: random_forest tuning n_estimators/max_depth). Returns the
    winning estimator and per-fold CV scores -- both destined for MLflow
    logging, not a bespoke embeddingsdb table (Decision 17).
    """
    raise NotImplementedError(
        "Hyperparameter tuning is not implemented yet. See design spec "
        "'Train/Test Split, Tuning, and Diagnostics', Decision 16."
    )


def compute_diagnostics(fitted_estimator, X_test, y_test):
    """
    Decision 16/18: compute OOB error/feature_importances_ (from the fitted
    estimator itself), ROC-AUC/ROC curve, confusion matrix, and a calibration
    curve against the held-out test split. Returns a structure suitable for
    logging to MLflow as metrics + figure artifacts, and for the Diagnostics
    page (`.../models/{model_id}/`, Decision 18) to render once proxied back
    out of MLflow.
    """
    raise NotImplementedError(
        "Diagnostics computation is not implemented yet. See design spec "
        "'Train/Test Split, Tuning, and Diagnostics' and Decision 18."
    )


def log_to_mlflow(algorithm, winning_params, cv_scores, diagnostics, fitted_estimator):
    """
    Decision 17: wrap the run in mlflow.start_run(); log params (search
    space + winning hyperparameters), metrics (per-fold CV scores, ROC-AUC,
    OOB error), diagnostic plots as artifacts (mlflow.log_figure); register
    the fitted estimator via mlflow.sklearn.log_model(...). Returns the
    resulting MLflow run_id, which is the only new value embeddingsdb needs
    to persist (`models.mlflow_run_id`) -- no duplicate model_metrics table.
    """
    raise NotImplementedError(
        "MLflow logging is not implemented yet -- the mlflow Pod does not "
        "exist yet. See WO_MLFLOW_TRACKING_URI in .env.example and design "
        "spec 'Experiment Tracking and Model Registry: MLflow'."
    )


def write_model_rows(task_type, algorithm, encoder_id, split_strategy,
                      split_params, mlflow_run_id, model_inputs, predictions):
    """
    Write one `models` row (with `mlflow_run_id`), one `model_inputs` row per
    observation (train/test split recorded per Decision 16), and `predictions`
    rows for the test split (at minimum) to embeddingsdb.
    """
    raise NotImplementedError(
        "embeddingsdb writes are not implemented yet -- the Pod does not "
        "exist yet. See EMBEDDINGSDB_URL in .env.example."
    )


def run(message):
    """
    Orchestrates the full model-train flow described in the module
    docstring above, for one invocation message.
    """
    raise NotImplementedError(
        "model-train's end-to-end flow is not implemented yet -- see this "
        "module's docstring for the intended sequence, and this repo's "
        "README 'Next steps' for what has to exist first."
    )


if __name__ == "__main__":
    main()
