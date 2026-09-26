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
- Makefile entry points, pre-commit hooks, CI, a Claude pull request review workflow, and a pull request template.
- Design decisions log, data download instructions, and repository settings notes.
