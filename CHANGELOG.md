# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-28

### Added

- Locked Python environment with uv, generated `requirements.txt`, and MIT license.
- Package skeleton with logging setup and seeding.
- Synthetic DRIVE-shaped fixtures for tests.
- Provenance module and run manifests, with `make check-data` verifying the data against committed checksums.
- SQLite schema version 1 for runs, folds, thresholds, and metrics.
- Run manifests record the git tag, and the schema accepts a reported run only from a clean, tagged commit.
- Built wheels ship a copy of the schema.
- Makefile entry points, pre-commit hooks, CI, a Claude pull request review workflow, and a pull request template.
- Design decisions log, data download instructions, and repository settings notes.
- Typed, strict experiment configs (`configs/baseline.yaml`, `configs/smoke.yaml`).
- DRIVE loader with layout checks and the official abnormality notes, quoted verbatim.
- Image-level 5-fold cross-validation folds, with the abnormal training images in different test folds.
- Leakage audit query 04, and a runner for numbered queries.
- Preprocessing (green channel, CLAHE, scaling, FOV standardization) and seeded `tf.data` patch sampling.
- `make check-data` writes the `images` table and logs the within-FOV vessel fraction.
- `make mutate` for mutation testing of the fold code.
- `folds.seed` draws the cross-validation split separately from the training seed.
- Small configurable U-Net, and FOV-masked BCE + Dice and Dice-only losses.
- Sliding-window inference and FOV-masked confusion-matrix metrics with the validation threshold sweep.
- `make train VARIANT=<name>`: resumable 5-fold cross-validation that logs runs, folds, thresholds, per-image metrics, and training curves to the database and writes a run manifest.
- `make verify-checkpoints` and a Colab runner notebook for GPU runs.
- AUC-ROC, AUC-PR (average precision), the Brier score, a pooled reliability table, and thin and thick vessel sensitivity, all inside the FOV.
- `make evaluate RUN=<run_id>`: fills those metrics for a finished run from its saved predictions and writes `results/<run_id>/evaluation.json`, after checking the prediction hashes, the data hash, and the stored confusion metrics. It records the evaluating commit, and evaluates a clean-tree run only from a clean tree at that run's commit.
- Each fold records the SHA-256 of its saved predictions.
- An `evaluation` config section for the reliability bins and the thin/thick quantile.
- SQL queries 01 (fold summary), 02 (worst images), 03 (variant comparison), and 05 (threshold log).
- Hypothesis property tests for the metrics, and `make mutate` now covers `metrics.py`.
- `configs/reporting.yaml` for the reported hardware, table rounding and bootstrap settings, and figure colors.
- `make mark-reported RUN=<run_id>`: marks a run reported only after checking its commit, tag, hardware, evaluation, checkpoints, and the leakage audit.
- `make snapshot TAG=<version>`: exports the reported runs, metrics only, to `results/release/`.
- `make tables`: Markdown tables in `results/tables` from the reported runs, with bootstrap intervals and each table's source run.
- `make figures`: the hero, best and worst, training curve, reliability, and thin and thick figures, with a sidecar naming each figure's run and images.
- `make presentation`: the repository preview image.
- The first reported run, the baseline trained from `v0.1.0-rc.1`, with its tables and figures.
- The README results, clinical data, governance, reproduce, and limitations sections, and the first model card.

### Changed

- `/scratch-train` runs on synthetic data only.
- The project title is "Retinal vessel segmentation on DRIVE".
- The smoke test also writes the tables of its run.

### Fixed

- `set_seed` now seeds Keras, so models built after the same seed start from the same weights.
- `make verify-checkpoints` fails on an unfinished run unless `UNFINISHED=1` is set, so a partial run can no longer pass.
- `make evaluate` rewrites a run's manifest from the database, so a crash after the run finished cannot leave `finished_at` missing.
- The Colab runner sets `MPLBACKEND=Agg`, since Colab's inline plotting backend stopped TensorFlow from importing.

[Unreleased]: https://github.com/nickmelamed/retinal-vessel-segmentation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nickmelamed/retinal-vessel-segmentation/releases/tag/v0.1.0
