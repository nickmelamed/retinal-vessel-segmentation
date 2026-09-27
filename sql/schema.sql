-- System of record for runs, folds, thresholds, and metrics (SPEC section 8).
-- Created by retinal_vessels.db.create_schema. Every statement is idempotent.
-- Booleans are INTEGER 0 or 1. Timestamps are ISO 8601 text in UTC.
--
-- Cross-validation folds are numbered from 1. In training_history, fold 0 is
-- the frozen final model trained on all labeled images.
--
-- fold_assignments deliberately allows an image to hold two roles in one fold,
-- or to be a test image in two folds. Those are the leaks that
-- queries/04_leakage_audit.sql detects, and the audit is only meaningful if
-- such rows can exist.
--
-- Anomaly module tables are added with schema version 2 (SPEC section 10).

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL PRIMARY KEY,
    applied_at TEXT NOT NULL
) STRICT;

INSERT OR IGNORE INTO schema_version (version, applied_at)
VALUES (1, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));

CREATE TABLE IF NOT EXISTS images (
    dataset TEXT NOT NULL CHECK (dataset IN ('drive', 'stare', 'chase')),
    image_id TEXT NOT NULL,
    -- Child ID for CHASE_DB1. NULL where the dataset publishes none.
    patient_id TEXT,
    split TEXT NOT NULL CHECK (split IN ('training', 'test', 'external')),
    has_abnormality INTEGER NOT NULL CHECK (has_abnormality IN (0, 1)),
    -- Verbatim from the dataset's official site.
    abnormality_note TEXT,
    fov_pixels INTEGER NOT NULL CHECK (fov_pixels > 0),
    fov_source TEXT NOT NULL CHECK (fov_source IN ('official', 'generated')),
    vessel_fraction_in_fov REAL CHECK (vessel_fraction_in_fov BETWEEN 0 AND 1),
    has_labels INTEGER NOT NULL CHECK (has_labels IN (0, 1)),
    PRIMARY KEY (dataset, image_id),
    CHECK (has_labels = 1 OR vessel_fraction_in_fov IS NULL)
) STRICT;

CREATE TABLE IF NOT EXISTS frozen_models (
    model_id TEXT NOT NULL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    checkpoint_sha256 TEXT NOT NULL,
    threshold REAL NOT NULL CHECK (threshold BETWEEN 0 AND 1),
    n_epochs INTEGER NOT NULL CHECK (n_epochs > 0),
    git_commit TEXT NOT NULL,
    frozen_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT NOT NULL PRIMARY KEY,
    variant TEXT NOT NULL,
    -- The full config as JSON text.
    config TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    seed INTEGER NOT NULL,
    git_commit TEXT NOT NULL,
    git_dirty INTEGER NOT NULL CHECK (git_dirty IN (0, 1)),
    -- The tag on the commit, NULL when HEAD is untagged.
    git_tag TEXT,
    data_hash TEXT NOT NULL,
    python_version TEXT NOT NULL,
    tensorflow_version TEXT NOT NULL,
    cuda_version TEXT,
    device TEXT NOT NULL,
    gpu_type TEXT NOT NULL,
    compute_platform TEXT NOT NULL,
    deterministic_ops INTEGER NOT NULL CHECK (deterministic_ops IN (0, 1)),
    -- Set for external evaluations. The frozen model audit checks both
    -- against frozen_models, so a run cannot quietly apply another threshold.
    frozen_model_id TEXT REFERENCES frozen_models (model_id),
    applied_threshold REAL CHECK (applied_threshold BETWEEN 0 AND 1),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    is_reported INTEGER NOT NULL DEFAULT 0 CHECK (is_reported IN (0, 1)),
    -- Only runs from a clean, tagged commit can back a reported number.
    CHECK (is_reported = 0 OR (git_dirty = 0 AND git_tag IS NOT NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS fold_assignments (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    fold INTEGER NOT NULL CHECK (fold >= 1),
    dataset TEXT NOT NULL,
    image_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('train', 'val', 'test')),
    PRIMARY KEY (run_id, fold, image_id, role),
    FOREIGN KEY (dataset, image_id) REFERENCES images (dataset, image_id)
) STRICT;

CREATE TABLE IF NOT EXISTS fold_status (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    fold INTEGER NOT NULL CHECK (fold >= 1),
    status TEXT NOT NULL CHECK (status IN ('running', 'complete')),
    checkpoint_sha256 TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 1 CHECK (attempts >= 1),
    PRIMARY KEY (run_id, fold),
    CHECK (
        status = 'running'
        OR (checkpoint_sha256 IS NOT NULL AND completed_at IS NOT NULL)
    )
) STRICT;

CREATE TABLE IF NOT EXISTS thresholds (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    fold INTEGER NOT NULL CHECK (fold >= 1),
    threshold REAL NOT NULL CHECK (threshold BETWEEN 0 AND 1),
    selection_rule TEXT NOT NULL,
    val_dice REAL CHECK (val_dice BETWEEN 0 AND 1),
    PRIMARY KEY (run_id, fold)
) STRICT;

CREATE TABLE IF NOT EXISTS per_image_metrics (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    dataset TEXT NOT NULL,
    image_id TEXT NOT NULL,
    -- NULL for external datasets, which are not split into folds.
    fold INTEGER CHECK (fold >= 1),
    prediction_mode TEXT NOT NULL CHECK (prediction_mode IN ('single', 'tta')),
    dice REAL CHECK (dice BETWEEN 0 AND 1),
    sensitivity REAL CHECK (sensitivity BETWEEN 0 AND 1),
    specificity REAL CHECK (specificity BETWEEN 0 AND 1),
    precision_score REAL CHECK (precision_score BETWEEN 0 AND 1),
    accuracy REAL CHECK (accuracy BETWEEN 0 AND 1),
    auc_roc REAL CHECK (auc_roc BETWEEN 0 AND 1),
    auc_pr REAL CHECK (auc_pr BETWEEN 0 AND 1),
    brier REAL CHECK (brier BETWEEN 0 AND 1),
    thin_sensitivity REAL CHECK (thin_sensitivity BETWEEN 0 AND 1),
    thick_sensitivity REAL CHECK (thick_sensitivity BETWEEN 0 AND 1),
    predicted_vessel_fraction REAL CHECK (predicted_vessel_fraction BETWEEN 0 AND 1),
    mean_uncertainty REAL CHECK (mean_uncertainty >= 0),
    uncertainty_error_auc REAL CHECK (uncertainty_error_auc BETWEEN 0 AND 1),
    PRIMARY KEY (run_id, dataset, image_id, prediction_mode),
    FOREIGN KEY (dataset, image_id) REFERENCES images (dataset, image_id)
) STRICT;

CREATE TABLE IF NOT EXISTS risk_coverage (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    dataset TEXT NOT NULL,
    image_id TEXT NOT NULL,
    fraction_referred REAL NOT NULL CHECK (fraction_referred BETWEEN 0 AND 1),
    dice_retained REAL CHECK (dice_retained BETWEEN 0 AND 1),
    PRIMARY KEY (run_id, dataset, image_id, fraction_referred),
    FOREIGN KEY (dataset, image_id) REFERENCES images (dataset, image_id)
) STRICT;

CREATE TABLE IF NOT EXISTS training_history (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    fold INTEGER NOT NULL CHECK (fold >= 0),
    epoch INTEGER NOT NULL CHECK (epoch >= 0),
    train_loss REAL NOT NULL,
    val_dice REAL CHECK (val_dice BETWEEN 0 AND 1),
    PRIMARY KEY (run_id, fold, epoch)
) STRICT;
