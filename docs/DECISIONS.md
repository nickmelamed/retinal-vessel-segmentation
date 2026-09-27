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

## D-015 Abnormality notes are quoted from the site (2026-09-26)

SPEC section 4 used to paraphrase the official notes for the seven images
with abnormalities. For example, it gave image 26's note as "atrophy around
the optic disc", where the site says "atrophy around optic disk". The
`images.abnormality_note` column must hold the site's own words, so the
owner chose to replace the paraphrase with the text of
https://drive.grand-challenge.org/ as retrieved on 2026-09-26. The loader
stores the same strings. If the site's wording changes, this entry and the
loader change together.

## D-016 Abnormal images go to different test folds (2026-09-26)

This refines SPEC section 5, with the owner's approval. `make_folds` puts
DRIVE's abnormal training images, 25, 26, and 32, in three different test
folds, and shuffles the other 17 into the remaining test slots with the same
seed. With a plain shuffle, two or three of them could share a fold, and that
fold's threshold and scores would then be driven by pathology. The split is
still by whole image, every image is a test image exactly once, and each
fold has 4 test, 2 validation, and 14 training images. Validation
images are drawn at random from each fold's 16 non-test images.

The folds depend only on the set of image ids and the seed, never on their
order, so every variant run with the same seed gets the same assignment
(section 7).

## D-017 Phase 1 data conventions (2026-09-26)

This completes the config part of D-009. `retinal_vessels.config` validates
each YAML file with pydantic in strict mode. Every key is required, since a
default would hide a missing setting, and unknown keys are errors. Strict
types mean `true` is not a number and `1` is not a boolean. The YAML is
validated through JSON so that lists can fill tuple fields.

DRIVE image ids are the two-digit strings from the file names ("03", "21").
Training and test ids do not overlap, so `(dataset, image_id)` is unique.

When the green channel is switched off, preprocessing converts the image to
grayscale, since CLAHE needs a single channel. Pixels outside the FOV are
set to 0 before CLAHE and again at the end, so the output depends only on
pixels inside the FOV. The owner chose this after the phase 1 review found
that CLAHE's tile histograms saw the raw border. It matters for external
validation (section 5), since STARE and CHASE_DB1 borders differ from
DRIVE's and must not change what the model sees inside the FOV.

Patch batches carry the FOV patch as a third element, so the phase 2 loss
can ignore pixels outside the FOV. Every random choice in patch sampling is
drawn up front from one seeded NumPy generator, which keeps the `tf.data`
pipeline deterministic even with parallel maps.

The owner asked for the flip probability to be a config setting,
`patches.flip_probability`, in place of an on or off `flip` flag. One
number cannot contradict itself the way a flag and a probability could, and
0 switches flips off. Both shipped configs use 0.5. The flip draws are
consumed even at 0, so changing the probability never shifts the other
augmentation draws for a given seed.

The within-FOV vessel fraction counts labeled pixels inside the FOV only.
The real DRIVE labels mark a few pixels outside it.

## D-018 `make check-data` writes the `images` table (2026-09-26)

The owner chose to have `check_data.py` write the `images` rows, instead of
leaving them for training in phase 2. SPEC section 6 gives the script the
dataset stats, and section 4 says the within-FOV vessel fraction is quoted
as computed, so the computed values belong in the database that documents
draw numbers from (rule 2). Once the checksums pass, the script loads every
image through the DRIVE loader and writes its row to `results/experiments.db`,
or the path given with `--db`. Repeating the command adds nothing. A stored
row that no longer matches the data is an error, like a checksum mismatch
(D-004). The pooled fraction is only logged, and it goes into documents
through the tables generated from the database.

## D-019 The folds have their own seed (2026-09-27)

This amends D-016. The folds used to be drawn with the top-level `seed`,
which also drives patch sampling and, from phase 2, training. SPEC section 7
needs every variant to use the same fold assignment so that R can pair
per-image results, and that held only because every config copied the same
seed. The phase 5 rerun-variance measurement also needs new training seeds
on unchanged folds, which a shared seed cannot give.

The owner chose a separate `folds.seed` over a check at training time that
compares a new run's folds with the baseline's. The check could only fire
once a baseline run existed, and it would have refused the rerun-variance
runs it was meant to protect. A test instead requires every file in
`configs/` to share one `folds.seed` and the same fold sizes, so a drifting
config fails `make ci` before any run exists.

`folds.seed` is 20260926, the value the baseline already used, so the
baseline's fold assignment is unchanged. The smoke config now uses the
baseline's folds too, and keeps its own top-level seed.
