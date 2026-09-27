-- Question: per held-out DRIVE image, how much does each other variant
-- differ from the baseline?
--
-- One row per image for every pair of a finished baseline run and a
-- finished run of another variant. Each delta is the other variant minus
-- the baseline, so a positive Dice delta means the other variant did
-- better on that image. Folds are meant to be shared across variants
-- (D-019), but a run from an older config could differ, so an image is paired
-- only when both runs held it out in the same fold and trained on the same
-- data.

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
        AND b.fold = o.fold
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
ORDER BY ro.variant, rb.run_id, ro.run_id, b.image_id;
