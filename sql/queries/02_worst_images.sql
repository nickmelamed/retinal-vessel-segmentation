-- Question: which held-out DRIVE images have the lowest Dice in each
-- finished run?
--
-- Every out-of-fold image of every finished run, ranked by Dice within its
-- run, lowest first, with ties broken by image id. The figures take the top
-- of each run's list (SPEC section 11), and has_abnormality marks images
-- 25, 26, and 32.

SELECT
    r.run_id,
    r.variant,
    r.is_reported,
    m.image_id,
    m.fold,
    i.has_abnormality,
    m.dice,
    m.sensitivity,
    m.precision_score,
    m.thin_sensitivity,
    m.thick_sensitivity,
    row_number() OVER (PARTITION BY m.run_id ORDER BY m.dice, m.image_id) AS dice_rank
FROM per_image_metrics AS m
INNER JOIN runs AS r ON m.run_id = r.run_id
INNER JOIN images AS i ON m.dataset = i.dataset AND m.image_id = i.image_id
WHERE
    m.dataset = 'drive'
    AND m.prediction_mode = 'single'
    AND m.fold IS NOT NULL
    AND r.finished_at IS NOT NULL
ORDER BY r.variant, r.run_id, dice_rank;
