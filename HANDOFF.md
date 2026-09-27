# Handoff for planning phase 3

This note is for a new Claude Code session that will plan phase 3
(evaluation) with the owner. Read CLAUDE.md first, then PROGRESS.md, then
SPEC sections 5, 8, and 14, and D-020 in docs/DECISIONS.md. Plan only. Wait
for the owner's approval before building. Do the work on the
`phase/3-evaluation` branch, which was cut from `main` after phase 2 merged
and so far holds only an update to this note.

## Where things stand (2026-09-27)

Phase 2 is merged into `main` (PR #5, merge commit `18fcc01`) and ticked in
PROGRESS.md. CI and the Claude review passed on its final commit.

What phase 2 added:

- Config sections `model`, `loss`, `training`, `inference`, and `threshold`
  in both YAML files. `Config` checks that `patches.size` and
  `inference.window` divide by `2**model.depth`. The threshold candidates are
  `i / threshold.divisions` (D-020).
- `model.py`: `build_unet(ModelConfig)`, sigmoid output, any input size that
  divides by `2**depth`, transposed-convolution upsampling.
- `losses.py`: FOV-masked soft Dice and BCE + Dice. The FOV weight travels
  as the second channel of the target, built by `pack_target`.
- `predict.py`: `predict_image(model, image, fov, InferenceConfig)`, sliding
  windows averaged, zero outside the FOV.
- `metrics.py` (protected): `confusion_counts`, `binary_metrics` (Dice,
  sensitivity, specificity, precision, accuracy, predicted vessel fraction),
  `binarize` (compares in float64), `threshold_grid`, `dice_per_threshold`,
  `best_threshold`, and `selection_rule`. Zero-denominator ratios are None,
  and Dice is 1 when both label and prediction are empty.
- `db.py`: `insert_run`, `run_manifest`, `find_resumable_run`,
  `stored_fold_rows`, `fold_statuses`, `start_fold`, `complete_fold` (one
  transaction), and `finish_run`.
- `train.py` CLI (`make train VARIANT=...`). Per fold, it stops early on the
  best-threshold validation Dice, keeps the best checkpoint at
  `models/<run_id>/fold_<k>.keras`, saves test-image probabilities to
  `results/<run_id>/predictions/<image_id>.npy` (float32, 0 outside the FOV),
  and writes `per_image_metrics` rows with the confusion metrics filled and
  the AUC, Brier, thin/thick, and uncertainty columns NULL.
- Resume rules (D-020). Only unfinished runs from a clean tree, matching on
  variant, config hash, commit, and data hash, are resumed, and only from a
  clean tree. Stored fold assignments and the environment must match, and
  each complete fold's checkpoint must match its stored SHA-256 and have all
  its prediction files present, so restore both `results/` and `models/`
  before resuming on a new machine.
- `scripts/verify_checkpoints.py` (`make verify-checkpoints`) and
  `notebooks/colab_runner.ipynb`.
- `set_seed` now also seeds Keras.

No reported run exists yet. The smoke config was run once on the real DRIVE
data into scratch directories to check the pipeline, and nothing from it was
kept. No number from any run may appear in a document until it comes from a
generated table (rule 2).

## What phase 3 must deliver

The phase 3 line in PROGRESS.md: the metrics that phase 2 did not build,
calibration, thin and thick vessel sensitivity, and SQL queries 01, 02, 03,
and 05. Also Hypothesis property tests for `metrics.py` (Dice in [0, 1],
pixels outside the FOV never change a score, a perfect prediction scores 1)
and `mutmut` on `metrics.py`.

- AUC-ROC and AUC-PR per image, the Brier score, and a reliability diagram
  from pooled out-of-fold probabilities (SPEC section 5).
- Thin and thick sensitivity: skeletonize the ground truth, estimate width
  with a distance transform, and bin skeleton pixels. The bins are tuned on
  training data only and documented (rule 3).
- An evaluation step that reads the saved predictions and fills the NULL
  columns of `per_image_metrics`, without retraining.
- The smoke test gains its evaluate step (section 14). Tables come in phase 4.

## Decisions to raise with the owner

The owner wants every decision in this section, including the open review
items below, addressed as soon as phase 3 begins. Raise them at the start of
planning, before any other phase 3 work.

- AUCs need either scikit-learn, which changes the lockfile (ask first), or
  a rank-based implementation in `metrics.py` tested against hand-worked
  cases.
- How evaluation writes its results: an `evaluate.py` CLI that updates the
  existing rows, or new rows. The primary key is (run_id, dataset, image_id,
  prediction_mode), so updating in place fits the schema without a change.
- What "tune the width bins on training data" means under cross-validation:
  per fold on that fold's training images, or once on all 20 labeled images.
  The second uses held-out images to set a reporting bin, not a model
  decision, but rule 3 should be checked with the owner.
- The number of reliability-diagram bins, as a config setting.
- Whether to store a SHA-256 for each prediction file in the database, so
  that phase 3 lineage can detect a missing or mixed-up set. The spec review
  suggested it. It would need a schema change, and `schema.sql` is protected.

Items left open from the phase 2 reviews, for the owner to decide:

- A crash after `finish_run` but before the final `write_manifest` leaves
  `manifest.json` without `finished_at`, and a finished run is never resumed.
  Writing the manifest from the database whenever `train` exits would close it.
- The ambiguity and resume tests all run in one process. The smoke test runs
  `train` twice in subprocesses, and comparing those two runs' thresholds would
  also test determinism across processes.
- `verify_checkpoints.py` can pass on a partial run. By default it checks the
  run with the largest `run_id`, and rerunning the notebook's train cell after
  a run finishes starts a new run, so verification would then check the new
  run's few complete folds and pass. Options are to log how many folds are
  complete, to require a finished run unless a flag says otherwise, or both.
  Raised by the spec review and again by the Claude review on PR #5.
- The `real` mode of `/scratch-train` trains on the real DRIVE images and
  leaves per-image metrics for real held-out images in a scratch database the
  agent can read, and its report includes fold log lines with validation Dice
  and thresholds. It breaks no rule today, but it makes held-out results easy
  to see during development (rule 3). Options are to drop the real mode, or
  to keep it as a pipeline check that reports only whether it ran and
  verified. Raised by the Claude review on PR #5.
- First Colab session: confirm that `COLAB_RELEASE_TAG` is set (the notebook
  prints it), that a T4 is assigned, and that `enable_op_determinism` raises
  no unimplemented-determinism error on the GPU for these ops.

## Working notes

- Protected files trigger an approval prompt, which is expected. They are
  hooks, CI, lockfiles, `data/CHECKSUMS.sha256`, docs/SPEC.md, the Makefile,
  and the section 5 files (see `.claude/protected-paths`), including
  `metrics.py`. Edit them with the Edit tool, never through a shell script,
  so the prompt appears.
- Write commit messages to a scratch file and commit with `git commit -F`.
  The allowed types are `feat`, `fix`, `refactor`, `perf`, `test`, `docs`,
  `build`, `ci`, `chore`, and `exp`. `style` is rejected.
- Running one test file with `pytest` fails the 85% coverage floor. Use
  `--no-cov` for partial runs, or `make test` for the full suite. Add
  `-p no:warnings` to hide TensorFlow's `gast` deprecation noise.
- Pytest treats `ResourceWarning` as an error. Open PIL images with `with`,
  and close SQLite connections with `contextlib.closing`, since
  `with sqlite3.connect(...)` only commits.
- Anything that calls a CLI's `main()` in process must restore the root
  logger afterwards (see `tests/unit/test_train.py`).
- Tests that call `check_data.py` or `train` must pass `--db` and the output
  directories into `tmp_path`.
- The resume tests record runs against a throwaway git repository, so they
  pass whether or not the working copy is dirty. A run of `train` from this
  checkout with uncommitted changes always starts a new run.
- Keras 3 seeds weights from its own generator. `set_seed` handles it, and
  any new seeding must go through `set_seed`.
- NumPy 2 compares a float32 array with a Python float in float32. Threshold
  with `metrics.binarize`, never `prob >= t`.
- In zsh, `$VAR` holding several arguments is not word-split. Use an array or
  `xargs`.
- `uv run ruff format` rewrites files, so read them again before an Edit.
- `check_numbers.py` skips itself until `results/tables` exists.
- This file carries context between sessions. At the end of a session or
  phase, rewrite it for the next piece of work and replace anything out of
  date.
