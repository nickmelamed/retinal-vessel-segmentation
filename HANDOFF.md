# Handoff for planning phase 4

This note is for a new Claude Code session that will plan phase 4 (the
v0.1.0 ship point) with the owner. Read CLAUDE.md first, then PROGRESS.md,
then SPEC sections 2, 5, 11, 12, and 17, and D-020 and D-021 in
docs/DECISIONS.md. Plan only, and wait for the owner's approval before
building. Cut `phase/4-ship` from `main` once phase 3 has merged.

## Where things stand (2026-09-27)

Phase 3 is merged into `main` (PR #6, merge commit `869778a`) and ticked in
PROGRESS.md. CI and the Claude review passed. Two items from that review
were fixed on `fix/review-followups`: `evaluate` refuses non-finite or
out-of-range probabilities, which SQLite would otherwise store as NULL, and
query 03 pairs images only on matching folds and data hashes. Check with
`gh pr list` that that branch has merged too.

What phase 3 added:

- `metrics.py` (protected) gained `auc_roc`, `average_precision` (reported
  as AUC-PR), `brier`, `reliability` (a pooled `ReliabilityTable`),
  `skeleton_radius`, `thin_edge`, and `width_sensitivity`. Everything is
  computed on `label & fov`, so pixels outside the FOV never change a score.
  The skeleton comes from scikit-image and the radius from OpenCV's exact
  distance transform, and the lockfile did not change.
- `evaluate.py` CLI (`make evaluate RUN=<run_id>`, default the latest
  finished run). It fills `auc_roc`, `auc_pr`, `brier`, `thin_sensitivity`,
  and `thick_sensitivity` in the existing `single` rows and writes
  `results/<run_id>/evaluation.json`. That file holds each fold's threshold,
  thin/thick edge, and bin pixel counts, the reliability table, and the
  pooled Brier score. Before writing anything it requires a finished run,
  data matching `runs.data_hash`, prediction files matching
  `predictions/fold_<k>.sha256`, and recomputed confusion metrics equal to
  the stored rows. A run trained from a clean tree is evaluated only from a
  clean tree at its own commit, and `evaluation.json` records the
  evaluating commit, dirty flag, and tag. Once the checks pass it rewrites
  `manifest.json` from the database. Rerunning gives identical output.
  Runs trained before phase 3 cannot be evaluated, since their stored
  config has no `evaluation` section and they have no prediction hash
  files. None of them is a reported run.
- `train.py` saves predictions through `save_predictions`, which also
  writes each fold's hash file.
- `db.py`: `fold_thresholds`, `stored_image_metrics`, `EvaluationRecord`,
  and `fill_evaluation` (one transaction, exactly one row per record).
- The `evaluation` config section: `reliability_bins: 10` and
  `thin_quantile: 0.5`. This changed the baseline config hash, which is
  fine since no reported run exists.
- SQL queries 01, 02, 03, and 05. The first three cover finished runs only.
  Every row carries `is_reported`, so the phase 4 tables can keep to
  reported runs.
- `verify_checkpoints.py` fails on an unfinished run unless
  `--allow-unfinished` (`make verify-checkpoints UNFINISHED=1`) is given.
- The smoke test evaluates its run and requires the second training run,
  in a new process, to reproduce the first exactly. It does on CPU.
- `/scratch-train` runs on synthetic data only, and now also evaluates.
- Hypothesis property tests in `tests/unit/test_metrics_properties.py`.
  mutmut covers `data.py` and `metrics.py`: 737 of 768 mutants killed, and
  the 31 survivors are equivalent (listed in commit `2d75828`).

No reported run exists yet, and no number from any run may appear in a
document until it comes from a generated table (rule 2).

## What phase 4 must deliver

The phase 4 line in PROGRESS.md: reported baseline runs from a clean,
tagged tree, then `make snapshot` and `make tables`, the hero,
best/worst, training-curve, reliability, and thin/thick figures, the README
first screen and its early sections, the first MODEL_CARD.md, badges, the
social preview image, and `docs/REPO_SETTINGS.md`. It ends with a check of
every document against section 2's rules, and then asking the owner to tag
`v0.1.0`.

The first reported run happens on Colab (`/colab-run`). After downloading,
check out the run's tagged commit with a clean tree on the laptop, then run
`make verify-checkpoints` and `make evaluate`.

## Decisions to raise with the owner

Raise these at the start of planning, before any other phase 4 work.

- The first Colab session still has to confirm that `COLAB_RELEASE_TAG` is
  set (the notebook prints it), that a T4 is assigned, and that
  `enable_op_determinism` raises no unimplemented-determinism error on the
  GPU for these ops. If it raises, deciding what to do is the owner's call.
- A reported run needs a tagged commit, and `v0.1.0` is meant to mark the
  finished ship point. Which tag should the reported baseline run use, for
  example a pre-release tag such as `v0.1.0-rc.1`?
- The format and location of `make tables` output (markdown under
  `results/tables/`, each with its run ids per section 17), and whether the
  reliability and thin/thick figures read `evaluation.json` or the
  database. `check_numbers.py` starts checking documents once
  `results/tables` exists.
- `make snapshot` (`scripts/snapshot_db.py`, which exports reported runs to
  `results/release/`, metrics only) is not built yet. Decide what it
  exports.
- Skeleton radii take a few discrete values (1, about 1.41, 2, about 2.24,
  and so on), and each fold's edge snaps to one of them, so folds on the
  real data may end up with different edges. Then "thin" means a different
  set of vessels in each fold. `evaluation.json` records every fold's edge.
  The thin/thick chart and the README must state the edges, and say plainly
  if they differ. Raised by the Claude review of PR #6.
- Figures need a plotting module (`retinal_vessels.figures`, SPEC section
  11). Load the dataviz skill before writing chart code.

## Working notes

- Protected files trigger an approval prompt, which is expected. They
  include `metrics.py`, `schema.sql`, the Makefile, lockfiles, CI, hooks,
  and docs/SPEC.md (see `.claude/protected-paths`). Edit them with the Edit
  tool, never a shell script. `ruff format` from the shell rewrites
  protected files too, so run `ruff format --check` on them and fix them
  with Edit.
- Write commit messages to a scratch file and commit with `git commit -F`.
  The subject is at most 72 characters. The allowed types are `feat`,
  `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`, and
  `exp`.
- Running one test file with `pytest` fails the 85% coverage floor. Use
  `--no-cov` for partial runs, or `make test` for the full suite. Add
  `-p no:warnings` to hide TensorFlow's `gast` deprecation noise.
- Pytest treats `ResourceWarning` as an error. Open PIL images with `with`,
  and close SQLite connections with `contextlib.closing`.
- Anything that calls a CLI's `main()` in process must restore the root
  logger afterwards (see `tests/unit/test_evaluate.py`).
- Tests that call a CLI must pass `--db` and the output directories into
  `tmp_path` or the synthetic data root.
- `tests/unit/test_evaluate.py` builds a finished run from synthetic data
  and fake probabilities without TensorFlow. Reuse its fixture for any test
  that needs an evaluated run.
- Keras 3 seeds weights from its own generator, and `set_seed` handles it.
  Any new seeding goes through `set_seed`.
- NumPy 2 compares a float32 array with a Python float in float32.
  Threshold with `metrics.binarize`, never `prob >= t`.
- Sums of fractions can round just above 1, which the schema's CHECK
  constraints reject. Divide once at the end, as `average_precision` does.
- On macOS, OpenCV's thread pool crashes a forked process. `make mutate`
  loads `tests/fixtures/single_thread_opencv.py` to avoid it. Anything else
  that forks after using OpenCV needs the same.
- In zsh, `$VAR` holding several arguments is not word-split. Use an array
  or `xargs`.
- This file carries context between sessions. At the end of a session or
  phase, rewrite it for the next piece of work and replace anything out of
  date.
