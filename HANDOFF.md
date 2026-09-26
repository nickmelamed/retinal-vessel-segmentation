# Handoff for planning phase 1

This note is for a new Claude Code session that will plan phase 1 (data) with
the owner. Read CLAUDE.md first, then PROGRESS.md, then the SPEC sections
named below. Plan only. Wait for the owner's approval before building, and
do the work on a `phase/1-data` branch cut from `main` after phase 0 merges.

## Where things stand (2026-09-26)

Phase 0 is built on `phase/0-setup` and ticked in PROGRESS.md. Unless the
owner has since pushed it, the branch is local only, with no pull request
yet. Check with `git log origin/main..phase/0-setup` and `gh pr list`. If it
is not merged, finish that first: push, open the PR with the template, and
merge with a merge commit, after asking the owner each time.

What exists now:

- `pyproject.toml`, `uv.lock`, generated `requirements.txt`, and
  `.python-version` pinned to 3.13 with TensorFlow 2.21 (D-001, D-003).
- `src/retinal_vessels/` holds four modules:
  - `utils.py` for `setup_logging` and `set_seed`
  - `provenance.py` for checksums, git state, environment, and run manifests
  - `db.py` for `connect` and `create_schema`
  - `__init__.py`
- `sql/schema.sql` is schema version 1 (D-007). It has no anomaly tables yet.
  `runs.applied_threshold` and `runs.frozen_model_id` exist for query 09.
- `scripts/check_data.py` (`make check-data`) verifies `data/DRIVE` against
  `data/CHECKSUMS.sha256` and never rewrites it (D-004). It passes on the
  real data. `DRIVE_DIR` points at the DRIVE directory itself.
- `tests/fixtures/synthetic_drive.py` writes a DRIVE-shaped tree with the
  real names, sizes, and formats. The labels and masks are grayscale GIFs
  holding 0 and 255, which is how the real files read back. `conftest.py`
  exposes it as `synthetic_data_root`.
- Makefile targets: `setup`, `lock`, `lock-check`, `check-data`, `lint`,
  `test`, `smoke`, `ci`, and `agent-check` (the full list, through `uv run`).
- Pre-commit hooks from CONTRIBUTING.md, including a commit-msg check.
- CI (`ci.yml`), the Claude PR review (`claude-review.yml`), and the PR
  template.
- Docs: DECISIONS.md (D-001 to D-009), DATA.md, REPO_SETTINGS.md,
  CHANGELOG.md, CITATION.cff, the MIT LICENSE, and an interim README.
- `make ci` passes: 53 tests at 98% coverage, with the 85% floor set in
  `pyproject.toml`. Pytest treats `ResourceWarning` as an error.

The owner still has to add the `ANTHROPIC_API_KEY` secret and set branch
protection (docs/REPO_SETTINGS.md). CI has not yet run on GitHub.

## What phase 1 must deliver

The phase 1 line in PROGRESS.md says: DRIVE loader adapter with pathology
metadata, validation, folds, preprocessing, patch sampling, and their tests.
SPEC sections 4, 5 (folds only), 6, 7 (preprocessing and patches), 8
(`images` and `fold_assignments`), and 14 give the details.

- `datasets/drive.py` loads images, labels, and FOV masks into the common
  sample type in `data.py`. It must fail loudly on any layout mismatch
  (section 4 gives the expected layout). The `images` rows include the
  abnormality notes for images 25, 26, and 32, quoted verbatim from the
  official site (rule 9). SPEC section 4 has the wording. Check it against
  the site and record the retrieval date in a comment.
- `retinal_vessels.config` and `configs/baseline.yaml` are deferred from
  phase 0 (D-009). Use typed, validated config (pydantic is locked) where
  unknown or missing keys are errors, with a test for each.
- Folds: 5-fold over the 20 labeled images, split by whole image. Each fold
  has 4 test, 2 val, and 14 train images, deterministic from the seed, and
  stored in `fold_assignments`. `data.py` is protected (section 5), so edits
  prompt for approval. The required tests say folds are disjoint and every
  image is `test` exactly once.
- Preprocessing (`preprocess.py`) runs green channel, CLAHE, a [0, 1] scale,
  and per-image standardization within the FOV. Each step can be toggled
  from config.
- Patch sampling (`patches.py`) builds seeded `tf.data` 64×64 patches with
  centers inside the FOV, using flips, 90° rotations, and brightness and
  contrast jitter. Test the shapes and that centers lie in the FOV.
- SPEC section 6 says `check_data.py` also reports dataset stats. The
  within-FOV vessel fraction must be computed from the data and quoted as
  computed (section 4), and it belongs in `images.vessel_fraction_in_fov`.
- PROGRESS.md follow-up: run `mutmut` on `data.py` and the fold code, and
  add tests for any surviving mutants.

## Decisions to raise with the owner

- **Colab install path (from phase 0 review).** Every make target calls
  `uv run`, which builds its own `.venv` from `uv.lock` and ignores a pip
  install of `requirements.txt`. `requirements.txt` is also exported with
  `--no-emit-project`, so a pure pip setup never installs `retinal_vessels`,
  and `__version__` then raises `PackageNotFoundError`. Choose before
  phase 2 writes `colab_runner.ipynb`. One option is uv on Colab (`uv sync
  --locked --no-dev`). The other is pip plus `pip install -e . --no-deps`
  with make targets that do not force `uv run`. Amend D-003 either way.
- **Fold numbering.** The schema allows `fold >= 0`. SPEC section 14's
  resumability test talks about "fold 2" and "folds 3–5", which suggests
  1-based numbering. Decide, and decide what fold the frozen model's
  training history uses.
- **Moving query 04 earlier.** Phase 1 starts writing `fold_assignments`,
  but query 04 (the leakage audit) is scheduled for phase 3. Consider
  adding it and its two tests in phase 1. It is protected.
- **Optional items from the phase 0 spec review**, none acted on:
  - `db.SCHEMA_PATH` uses `parents[2]`, which only works for editable
    installs. This matters for the phase 12 Dockerfile.
  - Manifests write `+00:00` timestamps with microseconds, while
    `schema.sql` writes a `Z` suffix. Query 09's "before freeze time" check
    should compare with `julianday()` or use one format.
  - `build_manifest` accepts any checksum mapping. It could require a
    passing `ChecksumReport` instead.
  - Rule 7 needs tagged commits, and nothing records the git tag yet. One
    option is `git describe --exact-match` in the manifest and `runs`.
  - The commit-msg hook checks only the type, not lowercase or the trailing
    period.
  - `ci.yml` cancels in-progress runs on `main` too.
  - `set_seed` sets `PYTHONHASHSEED` inside the running process, which has
    no effect there.
  - Colab detection relies on `COLAB_RELEASE_TAG`. Confirm the VS Code
    Colab extension's runtimes set it.

## Working notes

- Protected files trigger an approval prompt, which is expected. They are
  hooks, CI, lockfiles, `data/CHECKSUMS.sha256`, docs/SPEC.md, and the
  section 5 files (see `.claude/protected-paths`).
- Pre-commit hooks are installed, including commit-msg. Write commit
  messages to a scratch file and commit with `git commit -F <file>`, since
  the Bash guard matches blocked flags anywhere in the command text.
- Running one test file with `pytest` fails the 85% coverage floor. Use
  `--no-cov` for partial runs, or `make test` for the full suite.
- In zsh, `$VAR` holding a file list is not word-split. Use `xargs`.
  `check_numbers.py` silently skips paths that do not exist.
- macOS sed has no `\b`. Use perl for substitutions.
- `uv run ruff format` rewrites files, so read them again before an Edit.
- The `/finish-phase` slash command was not found in the VS Code extension
  at first, then worked later in the same session. The cause is unknown.
- Ask before pushing, opening a pull request, or changing the lockfile.
- This file carries context between sessions. At the end of a session or
  phase, rewrite it for the next piece of work and replace anything out of
  date.
