-- Question: per held-out DRIVE image, how much does each other variant
-- differ from the baseline?
--
-- One row per image for every pair of a finished baseline run and a
-- finished run of another variant. Each delta is the other variant minus
-- the baseline, so a positive Dice delta means the other variant did
-- better on that image. Folds are meant to be shared across variants
-- (D-019), but a run from an older config could differ. Two runs are paired
-- only when their fold assignments are identical, every fold with the same
-- training, validation, and test images, and they trained on the same data,
-- so the models behind each paired image saw the same images.

SELECT
    rb.run_id AS baseline_run_id,
    ro.run_id AS other_run_id,
    ro.variant AS other_variant,
    rb.is_reported AS baseline_is_reported,
    ro.is_reported AS other_is_reported,
    b.image_id,
    b.fold,
    o.dice - b.dice AS delta_dice,
    o.sensitivity - b.sensitivity AS delta_sensitivity,
    o.specificity - b.specificity AS delta_specificity,
    o.precision_score - b.precision_score AS delta_precision,
    o.auc_pr - b.auc_pr AS delta_auc_pr,
    o.thin_sensitivity - b.thin_sensitivity AS delta_thin_sensitivity,
    o.thick_sensitivity - b.thick_sensitivity AS delta_thick_sensitivity
FROM per_image_metrics AS b
INNER JOIN runs AS rb ON b.run_id = rb.run_id
INNER JOIN per_image_metrics AS o
    ON
        b.dataset = o.dataset
        AND b.image_id = o.image_id
        AND b.prediction_mode = o.prediction_mode
INNER JOIN runs AS ro ON o.run_id = ro.run_id
WHERE
    b.dataset = 'drive'
    AND b.prediction_mode = 'single'
    AND b.fold IS NOT NULL
    AND o.fold IS NOT NULL
    AND rb.variant = 'baseline'
    AND ro.variant != 'baseline'
    AND rb.finished_at IS NOT NULL
    AND ro.finished_at IS NOT NULL
    AND rb.data_hash = ro.data_hash
    AND NOT EXISTS (
        SELECT
            fa.fold,
            fa.image_id,
            fa.role
        FROM fold_assignments AS fa
        WHERE fa.run_id = rb.run_id
        EXCEPT
        SELECT
            fa.fold,
            fa.image_id,
            fa.role
        FROM fold_assignments AS fa
        WHERE fa.run_id = ro.run_id
    )
    AND NOT EXISTS (
        SELECT
            fa.fold,
            fa.image_id,
            fa.role
        FROM fold_assignments AS fa
        WHERE fa.run_id = ro.run_id
        EXCEPT
        SELECT
            fa.fold,
            fa.image_id,
            fa.role
        FROM fold_assignments AS fa
        WHERE fa.run_id = rb.run_id
    )
ORDER BY ro.variant, rb.run_id, ro.run_id, b.image_id;
