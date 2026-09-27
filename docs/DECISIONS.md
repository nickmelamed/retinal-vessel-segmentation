# Design decisions

Each entry records a decision, the date, and why it was made. Section numbers
refer to docs/SPEC.md. A decision that changes gets a new entry that names the
one it replaces.

## D-001 Python `3.13` (2026-09-26)

`.python-version` pins Python `3.13`. TensorFlow 2.21, the current stable
release, publishes wheels for Python `3.10` to `3.13` and none for `3.14`.
Colab's default runtime moved to Python `3.13` on 2026-08-19, and matching it
keeps local, CI, and reported GPU runs on the same interpreter (section 16).

## D-002 MIT license for the code (2026-09-26)

The owner chose MIT. `LICENSE` covers the code only. DRIVE publishes no
explicit license, and its images are never redistributed (section 4).

## D-003 uv manages the environment (2026-09-26)

`uv.lock` is the source of truth. `requirements.txt` is generated from it by
`make lock` for Colab, and `make lock-check` fails CI when the two disagree.
The Makefile and the local pre-commit hooks call tools through `uv run`, so
the locked versions are used rather than whatever is on the PATH. For that
reason `make agent-check` runs the list in PROGRESS.md with a `uv run` prefix.

TensorFlow is locked from phase 0 so that one lock covers the whole stack and
`requirements.txt` is complete for Colab. CI caches the wheels through uv.

`hypothesis` (property-based tests) and `mutmut` (mutation testing) are dev
dependencies, approved by the owner on 2026-09-26.

## D-004 Checksums are verified, never rewritten (2026-09-26)

`make check-data` verifies the data against `data/CHECKSUMS.sha256` and
reports missing, changed, and unlisted files by name. It writes the file only
when none exists, which is the first-run case in section 17, and never
changes an existing one. A changed dataset is an error for the owner to
investigate.

Paths in the manifest start with the dataset directory (`DRIVE/...`), and
`DRIVE_DIR` points at the DRIVE directory itself, defaulting to `data/DRIVE`.
Each external dataset will get its own directory variable in phase 7. The
run's data hash is a SHA-256 of the sorted manifest lines for the datasets it
used, so it does not depend on file order.

## D-005 R and Quarto join CI in phase 8 (2026-09-26)

The owner chose to wait. There is no report to render until phase 8, and
installing R and Quarto would slow every CI run for nothing. `make setup`
skips `renv::restore()` until then.

## D-006 Claude reviews every pull request (2026-09-26)

The owner approved a CI workflow in which Claude reviews each same-repository
pull request against the project rules and the SPEC. It needs an
`ANTHROPIC_API_KEY` repository secret, which the owner adds by hand (see
docs/REPO_SETTINGS.md). Pull requests from forks are skipped because secrets
are not available to them. The workflow passes the built-in `GITHUB_TOKEN`, so
the Claude GitHub App does not need to be installed.

## D-007 Database schema version 1 (2026-09-26)

`sql/schema.sql` creates the section 8 tables with these refinements.

Tables are `STRICT`, so SQLite enforces column types. Enumerations and metric
ranges are `CHECK` constraints, and foreign keys tie folds and metrics to runs
and images. A run can be reported only if it finished and came from a clean tree
and a tagged commit (rule 7), so `runs.git_tag` records the tag on HEAD, and the
manifest carries it too. A fold cannot be complete without its checkpoint
hash.

`fold_assignments` rejects exact duplicate rows but allows an image to hold
two roles in one fold, or to be a test image in two folds. Those are the leaks
query 04 detects. If the schema made them impossible to write, the audit could
never fail and its zero-row result would show nothing.

The precision metric is stored as `precision_score` because `PRECISION` is an
SQL keyword.

Every timestamp must equal its own `strftime('%Y-%m-%dT%H:%M:%SZ')` round
trip, so all of them share one whole-second UTC format and compare correctly
as text in query 09. An image with `has_abnormality = 1` must carry a
non-empty `abnormality_note`, and a normal image must not have one, so the
verbatim notes for images 25, 26, and 32 cannot be dropped on load.

`runs.frozen_model_id` links an external evaluation to its row in
`frozen_models`, and `runs.applied_threshold` records the threshold it
actually used. Query 09 compares both with the frozen record, so a run that
applied any other threshold is caught. Section 8 requires that check, and
`thresholds` cannot hold it because external runs have no fold.
The two columns are set together or not at all, so a NULL threshold cannot
hide behind a recorded model. Query 09 must not rely on these columns to find
external evaluations, because a run that leaves both NULL would escape it.
It finds them from their data instead: any run with `per_image_metrics` rows
on images whose split is `external`. It then flags each one whose
`frozen_model_id` is NULL, whose threshold differs from the frozen record, or
that started before the model was frozen.

`images.split` takes `external` for datasets that have no official split.

The anomaly module tables are left for schema version 2 in phase 9, when
section 10's design fixes their columns.

## D-008 The agent hook scripts are outside ruff's scope (2026-09-26)

`scripts/agent/` holds the protected, stdlib-only scripts that the Claude Code
hooks run. They predate the linter and are not part of the package, so
`pyproject.toml` excludes them from ruff rather than reformatting protected
files.

## D-009 Deferred from phase 0 (2026-09-26)

`retinal_vessels.config` and `configs/*.yaml` arrive with the first real
settings in phase 1 or 2. Phase 0 code needs only paths, which are CLI
arguments with defaults.

`CITATION.cff` cites only DRIVE for now. STARE and CHASE_DB1 are added in
phase 7, after their sources and terms are verified (section 4).

## D-010 Fold numbering (2026-09-26)

Cross-validation folds are numbered 1 to 5, matching SPEC section 14's
wording ("folds 3–5"). The schema rejects fold 0 in every fold table except
`training_history`, where fold 0 holds the frozen final model trained on all
20 labeled images (section 5). The owner approved this on 2026-09-26.

## D-011 Colab installs with uv (2026-09-26)

This amends D-003. Every make target runs its tools through `uv run`, which
builds `.venv` from `uv.lock` and ignores packages installed with pip. So a
Colab session that pip-installs `requirements.txt` and then calls make would
install everything twice. `requirements.txt` also leaves out the project
itself, so pip alone never installs `retinal_vessels`.

The owner chose uv on Colab. The Colab runner installs uv and runs
`uv sync --locked`, the same command CI runs, so Colab, CI, and the laptop
share one environment. The dev group is included because `uv run` syncs it
anyway. `requirements.txt` stays, generated by `make lock` and checked by
`make lock-check`, for anyone installing with pip. SPEC section 16 step 2
and the `/colab-run` skill were updated to match.

## D-012 The leakage audit arrives with the folds (2026-09-26)

Query 04 was scheduled for phase 3, but phase 1 is the first phase that
writes `fold_assignments`. The owner approved moving query 04 and its two
tests (zero rows on a valid database, rows on a leaky one) into phase 1, so
the fold code is audited from the first commit that can leak.

## D-013 Reported runs use a T4 on a paid Colab plan (2026-09-26)

The owner has a paid Colab plan, so the GPU type is a choice rather than
whatever the free tier assigns. Every reported run uses a T4. The whole
project needs roughly 20 to 25 fold trainings of a small U-Net, an estimate
of a few T4 hours, and at this size training is likely limited by patch
preparation on the CPU more than by the GPU. The T4 is the most reliably
available Colab GPU and the cheapest in compute units, and it matches
SPEC section 16. If T4s become hard to get, the small total compute means
every reported run can be repeated on one other GPU type, keeping the
same-hardware rule.

## D-014 Writing checksums needs `--init` (2026-09-26)

This amends D-004. `check_data.py` used to look for the checksum file under
`--data-root` and write one if it was missing. On Colab, with the data in
another directory, that would write a fresh manifest from whatever was on
disk and pass without checking anything, which defeats SPEC section 16 step 3.
The checksum file now always defaults to the committed
`data/CHECKSUMS.sha256`, wherever the data lives. A missing file is an error
unless `--init` is passed, and an existing file is still never changed. The
committed file covers section 17's first-run case. Found by the Claude review
on PR #1.
