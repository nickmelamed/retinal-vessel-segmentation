---
name: scratch-train
description: Run the train CLI end to end into throwaway directories to check the pipeline during development. Never for reported runs.
---

Check that training works end to end without touching `results/experiments.db`,
`results/`, or `models/`. $ARGUMENTS names the data, `synthetic` (the default)
or `real`.

1. Make `<scratchpad>/<name>/`, where `<scratchpad>` is this session's
   scratchpad directory. Pass every path below as its own argument, since zsh
   does not word-split a variable holding several.
2. For synthetic data, write it with
   `tests.fixtures.synthetic_drive.write_synthetic_drive(<dir>/data)`, then run
   `scripts/check_data.py --data-root <dir>/data --checksums <dir>/data/CHECKSUMS.sha256
   --db <dir>/experiments.db --init`. For the real data, leave `--data-root` and
   `--checksums` at their defaults so the committed checksums are verified.
3. Run `uv run python -m retinal_vessels.train --config configs/smoke.yaml`
   with `--db <dir>/experiments.db`, `--results-dir <dir>/results`, and
   `--models-dir <dir>/models`, plus `--data-root` and `--checksums` for
   synthetic data.
4. Run `uv run python scripts/verify_checkpoints.py --db <dir>/experiments.db
   --models-dir <dir>/models`.
5. Report the fold log lines and whether verification passed. Quote no Dice or
   other metric as a result: these runs use a tiny model and are not reported
   (rule 2).
