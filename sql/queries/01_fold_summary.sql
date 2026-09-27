-- Question: how does each finished run score, fold by fold, on its held-out
-- DRIVE images?
--
-- One row per run and fold, from single-pass predictions (SPEC section 8).
-- Averages skip NULLs, so n_evaluated counts the images whose AUCs and Brier
-- score are filled in, and a thin or thick mean covers only images with
-- skeleton pixels in that bin.

SELECT
    r.run_id,
    r.variant,
    r.is_reported,
    m.fold,
    count(*) AS n_images,
    count(m.brier) AS n_evaluated,
    avg(m.dice) AS mean_dice,
    avg(m.sensitivity) AS mean_sensitivity,
    avg(m.specificity) AS mean_specificity,
    avg(m.precision_score) AS mean_precision,
    avg(m.accuracy) AS mean_accuracy,
    avg(m.auc_roc) AS mean_auc_roc,
    avg(m.auc_pr) AS mean_auc_pr,
    avg(m.brier) AS mean_brier,
    avg(m.thin_sensitivity) AS mean_thin_sensitivity,
    avg(m.thick_sensitivity) AS mean_thick_sensitivity
FROM per_image_metrics AS m
INNER JOIN runs AS r ON m.run_id = r.run_id
WHERE
    m.dataset = 'drive'
    AND m.prediction_mode = 'single'
    AND m.fold IS NOT NULL
    AND r.finished_at IS NOT NULL
GROUP BY r.run_id, r.variant, r.is_reported, m.fold
ORDER BY r.variant, r.run_id, m.fold;
