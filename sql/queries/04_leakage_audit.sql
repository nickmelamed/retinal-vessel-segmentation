-- Question: does any image leak between roles in a cross-validation run?
--
-- Returns one row per leak, and must return zero rows (SPEC section 8).
-- An image leaks if it holds more than one role within a fold, or if it is
-- a test image in more than one fold of the same run. The schema allows
-- both on purpose, so that this audit can fail (D-007).

SELECT
    run_id,
    dataset,
    image_id,
    'more than one role in fold ' || fold AS problem,
    group_concat(DISTINCT role) AS detail
FROM fold_assignments
GROUP BY run_id, fold, dataset, image_id
HAVING count(DISTINCT role) > 1

UNION ALL

SELECT
    run_id,
    dataset,
    image_id,
    'test in more than one fold' AS problem,
    group_concat(DISTINCT fold) AS detail
FROM fold_assignments
WHERE role = 'test'
GROUP BY run_id, dataset, image_id
HAVING count(DISTINCT fold) > 1

ORDER BY run_id, dataset, image_id, problem;
