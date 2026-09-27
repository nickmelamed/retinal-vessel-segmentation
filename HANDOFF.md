# Handoff for planning phase 2

This note is for a new Claude Code session that will plan phase 2 (model and
training) with the owner. Read CLAUDE.md first, then PROGRESS.md, then the
SPEC sections named below. Plan only. Wait for the owner's approval before
building, and do the work on a `phase/2-model` branch cut from `main` after
phase 1 merges.

## Where things stand (2026-09-26)

Phase 1 is finished on `phase/1-data` and ticked in PROGRESS.md. It is not
pushed yet, and the pull request (merge commit, not squash) waits for the
owner. Check `gh pr list`, and cut `phase/2-model` from `main` once it has
merged. `make ci` passes on the branch: 218 tests at 97.82% coverage.

What phase 1 added:

- `config.py` loads `configs/baseline.yaml` and `configs/smoke.yaml` into
  strict pydantic models. Every key is required and unknown keys are errors
  (D-017). The sections so far are `variant`, `seed`, `folds`, `preprocess`,
  and `patches`. Phase 2 adds model and training sections to both files and
  to `Config`.
- `data.py` (protected) holds `Sample`, which validates itself, and
  `make_folds`. Folds are 14/2/4, numbered 1 to 5, depend only on the id set
  and seed, and put images 25, 26, and 32 in different test folds (D-016).
  `fold_rows` flattens folds for the database.
- `datasets/drive.py` checks the exact layout and file formats, loads
  training images with labels and test images without, and carries the
  seven official abnormality notes verbatim (D-015, SPEC section 4 now
  quotes them).
- `db.py` gained `image_record`, `write_images` (safe to repeat, raises on
  changed data), `write_fold_assignments`, and `run_query`. Built wheels
  ship `sql/queries/`.
- `sql/queries/04_leakage_audit.sql` (protected) returns zero rows on valid
  folds and a named row for each kind of leak.
- `preprocess.py` runs green channel (grayscale when off), CLAHE, a [0, 1]
  scale, and FOV standardization. The border is zeroed before CLAHE and at
  the end, so the output depends only on FOV pixels. The owner chose this
  after the spec review found CLAHE saw the raw border (D-017).
- `patches.py` builds a batched `tf.data` dataset of `(x, y, w)` patches,
  with `w` the FOV patch for masking the loss. All randomness is drawn up
  front from one NumPy generator, so one seed gives one sequence of batches.
  Pass a new seed for each epoch. The flip chance is `flip_probability`.
- `make check-data` also writes the 40 DRIVE `images` rows to
  `results/experiments.db` and logs the pooled within-FOV vessel fraction
  (D-018). It has been run on the real data. No document quotes the number
  yet, and none may until it comes from a generated table (rule 2).
- `make mutate` runs mutmut on `data.py`. It kills 153 of 156 mutants. The
  3 survivors are equivalent: `strict=None` or no `strict` in the `zip` over
  spread ids, and `replace=None` in `rng.choice`, all behave like `False`.

## What phase 2 must deliver

The phase 2 line in PROGRESS.md says: U-Net, losses, resumable CV training
with database logging and run manifests, `colab_runner.ipynb`, the smoke
integration test, and the resumability tests. SPEC sections 7 (model, loss,
training), 8 (`runs`, `fold_status`, `thresholds`, `training_history`), 14
(the resumability tests and the smoke test), 16 (Colab workflow, resumable
runs), and 17 (run manifest) give the details.

- `model.py`: a small U-Net (3 to 4 levels, base 16 to 32 filters, batch
  norm, dropout), every size from config.
- `losses.py`: BCE + Dice (baseline) and Dice only (the ablation), masked
  by the FOV weight `w` from `patches.py`.
- `train.py` (CLI): full 5-fold CV for one config. Adam, early stopping on
  validation-image Dice computed on whole validation images inside the FOV,
  the best checkpoint per fold, and training curves in `training_history`.
  A fold is marked complete in `fold_status` only after its checkpoint,
  threshold, metrics, and history are all written. A restart skips complete
  folds and retrains a `running` one from scratch.
- A `runs` row and `results/<run_id>/manifest.json` for every run, through
  `provenance.build_manifest` and `Config.as_json()` with
  `provenance.config_hash`.
- Required tests: interrupting a synthetic CV run after fold 2 and restarting
  it trains folds 3 to 5 only, with the same fold assignments and database
  state as an uninterrupted run. A fold left `running` is retrained. The
  smoke test trains, predicts, evaluates, and writes tables end to end on
  synthetic data in about two minutes on CPU.
- `notebooks/colab_runner.ipynb` with only the section 16 steps.

## Decisions to raise with the owner

- Early stopping and threshold choice need validation Dice on whole images,
  which means sliding-window inference and a FOV-masked Dice. PROGRESS.md
  puts sliding-window inference and the metrics in phase 3, and `metrics.py`
  is protected (section 5). Ask whether phase 2 builds `predict.py` and the
  Dice part of `metrics.py` now, or uses something smaller until phase 3.
- The threshold rule (section 5: maximize validation Dice) needs a candidate
  grid. Its spacing is a config setting to agree on.
- What counts as an epoch: `patches.per_epoch` patches with seed
  `seed + epoch` is the simplest reading. Confirm it, and the patience for
  early stopping.
- Confirm that the VS Code Colab extension's runtimes set
  `COLAB_RELEASE_TAG`, which `compute_platform` relies on. Colab installs
  with `uv sync --locked` (D-011). Reported runs use a T4 (D-013).

The owner settled the optional items from the phase 1 review. `check_data.py`
logs a database constraint error as one line, and a split with no abnormal
images logs "no abnormality notes". The patch flip probability is the
`patches.flip_probability` setting (D-017). Dropping the planned check that
rejects labels outside the FOV is accepted, since real DRIVE labels mark a
few pixels outside it and the vessel fraction counts only pixels inside.

## Working notes

- Protected files trigger an approval prompt, which is expected. They are
  hooks, CI, lockfiles, `data/CHECKSUMS.sha256`, docs/SPEC.md, and the
  section 5 files (see `.claude/protected-paths`). `metrics.py` is on that
  list.
- Write commit messages to a scratch file and commit with `git commit -F`,
  since the Bash guard matches blocked flags anywhere in the command text.
- Running one test file with `pytest` fails the 85% coverage floor. Use
  `--no-cov` for partial runs, or `make test` for the full suite. Add
  `-p no:warnings` to hide TensorFlow's `gast` deprecation noise.
- Pytest treats `ResourceWarning` as an error. Open PIL images with `with`.
- Anything that calls a CLI's `main()` in process must restore the root
  logger afterwards, or `setup_logging` in later tests becomes a no-op (see
  `tests/unit/test_check_data.py`).
- Tests that call `check_data.py` must pass `--db` into `tmp_path`, or they
  write synthetic rows into the real `results/experiments.db`.
- Pydantic strict mode rejects YAML lists for tuple fields, so `load_config`
  validates through JSON.
- mutmut 3 copies only the mutated file into `mutants/src`, so
  `pyproject.toml` lists `src/` under `also_copy`. `make mutate` clears
  `mutants/` first.
- Pillow reads a GIF with a gray palette back as mode `L`, and an all-black
  image saved with palette optimization as `P`. Save test masks with
  `optimize=False`.
- In zsh, `$VAR` holding a file list is not word-split. Use `xargs`.
  `check_numbers.py` skips itself until `results/tables` exists.
- `uv run ruff format` rewrites files, so read them again before an Edit.
- Ask before pushing, opening a pull request, or changing the lockfile.
- This file carries context between sessions. At the end of a session or
  phase, rewrite it for the next piece of work and replace anything out of
  date.
