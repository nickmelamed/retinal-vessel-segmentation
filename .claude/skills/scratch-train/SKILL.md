---
name: scratch-train
description: Run the train and evaluate CLIs end to end on synthetic data in throwaway directories to check the pipeline during development. Never for reported runs.
---

Check that training and evaluation work end to end without touching
`results/experiments.db`, `results/`, or `models/`. Synthetic data only.
Never point this at the real DRIVE images, since that would put held-out
results in a database during development (D-021). Real data is exercised
only by reported runs on Colab (`/colab-run`).

1. Make `<scratchpad>/<name>/`, where `<scratchpad>` is this session's
   scratchpad directory. Pass every path below as its own argument, since zsh
   does not word-split a variable holding several.
2. Write the data with
   `tests.fixtures.synthetic_drive.write_synthetic_drive(<dir>/data)`, then run
   `scripts/check_data.py --data-root <dir>/data --checksums <dir>/data/CHECKSUMS.sha256
   --db <dir>/experiments.db --init`.
3. Run `uv run python -m retinal_vessels.train --config configs/smoke.yaml`
   with `--data-root <dir>/data`, `--checksums <dir>/data/CHECKSUMS.sha256`,
   `--db <dir>/experiments.db`, `--results-dir <dir>/results`, and
   `--models-dir <dir>/models`.
4. Run `uv run python -m retinal_vessels.evaluate` with the same
   `--data-root`, `--checksums`, `--db`, and `--results-dir`.
5. Run `uv run python scripts/verify_checkpoints.py --db <dir>/experiments.db
   --models-dir <dir>/models`.
6. Report the exit code of each step and whether verification passed. Quote
   no Dice or other metric as a result: these runs use a tiny model on
   synthetic data and are not reported (rule 2).
