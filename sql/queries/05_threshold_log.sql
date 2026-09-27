-- Question: which threshold did each fold choose, by what rule, and after
-- how many epochs?
--
-- One row per run and fold, finished or not. best_epoch is the first epoch
-- whose validation Dice equals the fold's recorded one, which is the epoch
-- whose checkpoint and threshold were kept (D-020). n_epochs counts every
-- epoch trained before early stopping.

SELECT
    r.run_id,
    r.variant,
    r.is_reported,
    t.fold,
    t.threshold,
    t.selection_rule,
    t.val_dice,
    r.finished_at IS NOT NULL AS is_finished,
    (
        SELECT min(h.epoch)
        FROM training_history AS h
        WHERE h.run_id = t.run_id AND h.fold = t.fold AND h.val_dice = t.val_dice
    ) AS best_epoch,
    (
        SELECT max(h.epoch)
        FROM training_history AS h
        WHERE h.run_id = t.run_id AND h.fold = t.fold
    ) AS n_epochs
FROM thresholds AS t
INNER JOIN runs AS r ON t.run_id = r.run_id
ORDER BY r.variant, r.run_id, t.fold;
