# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
