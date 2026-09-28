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
seed. Any variant given another seed would have silently changed its folds.

The owner chose a separate `folds.seed` over a check at training time that
compares a new run's folds with the baseline's. The check could only fire
once a baseline run existed, and only at training time, possibly on Colab.
A test instead requires every cross-validation
config in `configs/` to share one `folds.seed` and the same fold sizes, so a
drifting config fails `make ci` before any run exists. `final.yaml` and
`external.yaml`, when they arrive, are outside that rule, since the frozen
model trains on all 20 images and external validation has no folds.

`runs.seed` records the top-level seed, not the fold seed, which lives in
`runs.config`. Anything that pairs runs by their folds, like the section 7
comparisons in R, must join on `fold_assignments`, not on `runs.seed`.

`folds.seed` is 20260926, the value the baseline already used, so the
baseline's fold assignment is unchanged. The smoke config now uses the
baseline's folds too, and keeps its own top-level seed.

## D-020 Phase 2 training conventions (2026-09-27)

Early stopping and the fold threshold need Dice on whole validation images,
so the owner moved sliding-window inference (`predict.py`) and the
confusion-matrix part of `metrics.py` (Dice, sensitivity, specificity,
precision, accuracy, predicted vessel fraction) from phase 3 into phase 2.
AUCs, the Brier score, thin and thick sensitivity, the Hypothesis tests, and
mutmut on `metrics.py` stay in phase 3. Each fold saves its test images'
probabilities as `results/<run_id>/predictions/<image_id>.npy`, so phase 3
computes those metrics without retraining. Checkpoints go to
`models/<run_id>/fold_<k>.keras`.

After every epoch, training sweeps the candidate thresholds on the fold's two
validation images and records the best mean per-image Dice. The owner chose
this over monitoring Dice at a fixed 0.5. Training stops once that value has
not improved for `training.patience` epochs, and the best epoch's checkpoint
and threshold are kept. That Dice is `thresholds.val_dice`, and the
`training_history` curve holds the same quantity per epoch. The candidates are
`i / threshold.divisions` for i from 1 to divisions minus 1, an integer count
in place of a float step so the grid stays exact. A probability equal to the
threshold counts as vessel. Ties go to the lowest threshold. Dice is 1 when
neither the label nor the prediction has a vessel pixel. Other ratios with a
zero denominator are NULL.

An epoch is `patches.per_epoch` patches. Weights are initialized from a seed
mixed from (seed, fold), and each epoch's patches from (seed, fold, epoch),
with NumPy's `SeedSequence`. Every fold depends only on its own seeds, so a
resumed fold repeats an uninterrupted one bit for bit when deterministic ops
are on, which the resumability tests check. `set_seed` also seeds Keras,
whose layers draw initial weights from their own generator. The baseline
trains for at most 100 epochs with patience 10.

Rerunning `make train` continues the unfinished run whose variant, config
hash, commit, and data hash all match. Only clean-tree runs are resumed, and
only from a clean tree. A dirty tree always starts a new run, since
uncommitted edits could differ from the code that trained the earlier folds
while the run's row names a single commit (section 17). The phase 2 spec
review found this gap. More than one match is an error, and
`--new` always starts a fresh run. A finished run is never resumed, so
the same command then starts a new one. Resuming is
refused if the stored fold assignments differ from the computed ones, or if
the environment differs (Python, TensorFlow or CUDA version, device, GPU
type, or platform), since reported comparisons must not mix hardware
(section 16). Resuming is also refused when a complete fold's
checkpoint is missing or does not match its stored SHA-256, or when any of
its prediction files is missing, since the run would otherwise finish
without an out-of-fold prediction for every image. A resumed run rebuilds its manifest from the `runs` row,
because a fresh Colab machine may not have the original `manifest.json`.

A fold's threshold, test-image metrics, and history are written together
with its `complete` status in one transaction. A crash therefore leaves a
fold either complete with all its rows or `running`, and a running fold's
partial rows are deleted before it is retrained.

The phase 2 smoke test trains the full smoke cross-validation through the
CLI. The evaluation and table steps that SPEC section 14 lists join the smoke
test in phases 3 and 4.

## D-021 Phase 3 evaluation conventions (2026-09-27)

The owner settled these before phase 3 work began.

AUC-ROC and AUC-PR are computed in `metrics.py` with NumPy, so scikit-learn
stays out of the lockfile. AUC-ROC counts, for each vessel pixel, the
background pixels scored below it plus half of those tied with it. AUC-PR is
the step-wise average precision, with tied scores treated as one threshold.
It is not the trapezoidal area, which overstates a precision-recall curve.
AUC-ROC is NULL when the FOV holds only one class, and AUC-PR is NULL when it
holds no vessel pixel.

A separate `evaluate` command reads a finished run's saved probabilities and
fills the AUC, Brier, and thin and thick sensitivity columns of the existing
`single` rows of `per_image_metrics`. The primary key already identifies
those rows, so the schema does not change. Rerunning it gives the same
values.

The thin and thick bins follow rule 3. Each fold sets its own edge at the
`evaluation.thin_quantile` quantile (0.5) of skeleton radius over the ground
truth of its 16 training and validation images. A skeleton pixel is thin when
its radius is at most the edge. The fold's held-out images are scored with
that edge. Each fold's edge and bin pixel counts are saved with the results.
The skeleton comes from scikit-image and the radius from OpenCV's exact
Euclidean distance transform. Both are computed on the label inside the FOV,
so pixels outside the FOV never change a score.

The reliability diagram pools every out-of-fold FOV pixel into
`evaluation.reliability_bins` (10) equal-width bins and keeps each bin's pixel
count, so a sparse bin can be seen as such. The evaluation settings are a
section of each variant config, so a run's stored config records how it was
evaluated.

Each fold writes `predictions/fold_<k>.sha256` next to its probability files.
Evaluation checks those hashes, then recomputes the confusion metrics at the
fold's stored threshold and requires them to equal the stored row exactly. A
missing, stale, or swapped file therefore stops evaluation. The owner chose
this over a checksum column, which would have needed a schema change.
Evaluation also requires the data checksums to match the run's `data_hash`, so
the labels are the ones training used.

The phase 3 spec review found that evaluation recorded no code version, so
reported numbers could come from a dirty or later checkout without a trace
(rule 7). The owner chose to have `evaluation.json` record the evaluating
commit, dirty flag, and tag. A run trained from a clean tree is evaluated
only from a clean tree at that same commit, which mirrors the resume rule. A
dirty-tree run can be evaluated from any checkout, and the record shows
which.

Once every check passes, evaluation rewrites `manifest.json` from the `runs`
row. A crash between marking the run finished and writing the manifest would
otherwise leave the file without `finished_at`.

Items from the phase 2 reviews are settled as follows.
`verify_checkpoints.py` fails unless the run is finished with every fold
complete (`--allow-unfinished` overrides this), and it always logs how many
folds are complete. The smoke test compares its two training runs, each in
its own process, and requires identical thresholds, validation Dice, and
per-image Dice. `/scratch-train` no longer trains on the real DRIVE images,
so no held-out result reaches a database during development.

## D-022 Phase 4 reporting conventions (2026-09-27)

The owner settled these while planning phase 4.

The first reported baseline run trains from the tag `v0.1.0-rc.1` on
`0452dba`, the phase 3 merge, which already holds everything training and
evaluation need. `v0.1.0` goes later on the commit that finishes the
documents. That commit descends from the rc, so the run's commit can be
reached from the release tag (docs/CONTRIBUTING.md). A fix needed before
the run finishes means a new commit, `v0.1.0-rc.2`, and a fresh run. A run
is never patched.

Settings for presenting results live in `configs/reporting.yaml`, with its
own strict model, instead of in a section of the variant configs. Each run
stores its variant config, and the rc run's stored config must keep
validating for `evaluate` and resume, so the variant `Config` does not
change in this phase.

Nothing set `runs.is_reported` before this phase. `make mark-reported
RUN=<id>` sets it only when all of these hold. The run finished every fold
from a clean, tagged commit, on the GPU and platform the reporting config
names (Tesla T4 on Colab, D-013). Every out-of-fold image is evaluated,
and `evaluation.json` was written by a clean checkout of the run's own
commit. The checkpoints match `fold_status`, and the leakage audit returns
no rows. Every failed check is listed at once, and nothing changes on
failure.

`make snapshot TAG=<version>` writes `results/release/experiments_<tag>.db`
with the full schema, every row of `images`, and the rows of reported runs
from each table that has a `run_id`. A table in neither group is refused,
so a new table is placed on purpose. Each run's `evaluation.json` is copied
beside it, since the reliability table and the fold edges live only there.
Foreign keys must resolve and the leakage audit must be empty before the
snapshot is kept, and an existing snapshot is never replaced.

`make tables` and `make figures` read reported runs only, from the database
and each run's `evaluation.json`, whose thresholds must match the database.
Intervals are percentile bootstraps of the mean over images, 10,000
resamples with a fixed seed, and every metric of a run uses the same image
draws. The R report in phase 8 recomputes them, and the two should agree
up to resampling noise. Numbers are printed with a fixed count of decimals
so that documents can copy them exactly. Every table file and every figure
sidecar names the runs, tag, commit, and data hash it came from, and
regenerating either gives the same bytes. Two reported runs of one variant
are refused until phase 5 decides how rerun variance is tabulated. The
figures rethreshold the saved probabilities and require the stored Dice,
so no figure can show a prediction other than the one that was scored. A
figure over the pre-commit hook's 1 MB limit is an error, and the layout
shrinks instead of the limit rising.

The thin and thick bins keep the per-fold quantile edge of D-021. The
per-fold table, the thin and thick table, and the chart say plainly
whether the folds set different edges.

The README carries no coverage badge in v0.1.0. CI prints coverage but
publishes nothing a badge could read, and publishing it needs either a
third-party service or a CI job with write access. Phase 10 revisits it.

The first Colab session could not import TensorFlow. Colab sets
`MPLBACKEND` to its inline backend, which the locked environment does not
include, and Keras imports matplotlib as TensorFlow loads. The runner
notebook now sets `MPLBACKEND=Agg` before any cell runs uv. No project code
changed, so the run on the rc tag stays reportable.

The evaluated-run test fixture moved from `tests/unit/test_evaluate.py` to
`tests/fixtures/evaluated_run.py`, taking two setup assertions with it,
so the reporting, snapshot, and table tests can share it. The Stop hook
counts assertions per file against `main`, so it flagged the move as a
weakened test. The owner approved the move, and it reached `main` on its
own in PR #8.

The reported run is `20260928T013730Z-4f5d08`, trained on a Colab Tesla
T4 with deterministic ops on. No op raised an unimplemented-determinism
error. Each fold logged one `meta_optimizer.cc:967] layout failed:
INVALID_ARGUMENT` line from TensorFlow's graph optimizer, which could not
rewrite the tensor layout around the dropout ops and left that part of
the graph as it was. The line is logged at error level, but training
continued, every fold completed on its first attempt, and the checkpoints
and predictions verified on the laptop. The whole run came to about
446 MB of checkpoints, over the Colab extension's download limit of
about 512 MB once encoded, so the outputs came down as one zip of
`results/` and one zip per checkpoint.
