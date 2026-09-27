# Handoff for planning phase 1

This note is for a new Claude Code session that will plan phase 1 (data) with
the owner. Read CLAUDE.md first, then PROGRESS.md, then the SPEC sections
named below. Plan only. Wait for the owner's approval before building, and
do the work on a `phase/1-data` branch cut from `main` after phase 0 merges.

## Where things stand (2026-09-26)

Phase 0 is built on `phase/0-setup` and ticked in PROGRESS.md. The branch
is pushed and open as PR #1. CI passes on GitHub. Check `gh pr view 1` for
whether it has merged, and if not, ask the owner before merging (merge
commit, not squash).

What exists now:

- `pyproject.toml`, `uv.lock`, generated `requirements.txt`, and
  `.python-version` pinned to 3.13 with TensorFlow 2.21 (D-001, D-003).
- `src/retinal_vessels/` holds four modules:
  - `utils.py` for `setup_logging` and `set_seed`
  - `provenance.py` for checksums, git state and tag, environment, UTC
    timestamps, and run manifests. `build_manifest` takes a passing
    `ChecksumReport` and refuses a failed one.
  - `db.py` for `connect` and `create_schema`
  - `__init__.py`
- `sql/schema.sql` is schema version 1 (D-007). It has no anomaly tables yet.
  `runs.applied_threshold` and `runs.frozen_model_id` exist for query 09.
  Folds count from 1, and fold 0 in `training_history` is the frozen model
  (D-010). A reported run must be clean and tagged (`runs.git_tag`). Built
  wheels carry a copy of the schema.
- `scripts/check_data.py` (`make check-data`) verifies `data/DRIVE` against
  `data/CHECKSUMS.sha256` and never rewrites it (D-004). The checksum file
  always defaults to the committed one, and writing a missing one needs
  `--init` (D-014). It passes on the real data. `DRIVE_DIR` points at the
  DRIVE directory itself.
- `tests/fixtures/synthetic_drive.py` writes a DRIVE-shaped tree with the
  real names, sizes, and formats. The labels and masks are grayscale GIFs
  holding 0 and 255, which is how the real files read back. `conftest.py`
  exposes it as `synthetic_data_root`.
- Makefile targets: `setup`, `lock`, `lock-check`, `check-data`, `lint`,
  `test`, `smoke`, `ci`, and `agent-check` (the full list, through `uv run`).
- Pre-commit hooks from CONTRIBUTING.md, including a commit-msg check.
- CI (`ci.yml`), the Claude PR review (`claude-review.yml`), and the PR
  template.
- Docs: DECISIONS.md (D-001 to D-014), DATA.md, REPO_SETTINGS.md,
  CHANGELOG.md, CITATION.cff, the MIT LICENSE, and an interim README.
- `make ci` passes: 71 tests at 97% coverage, with the 85% floor set in
  `pyproject.toml`. Pytest treats `ResourceWarning` as an error.
- `.claude/skills/commit/` holds the commit procedure as a skill.

The Claude review job fails until the owner adds the `ANTHROPIC_API_KEY`
secret. Branch protection is also the owner's to set (docs/REPO_SETTINGS.md).

## What phase 1 must deliver

The phase 1 line in PROGRESS.md says: DRIVE loader adapter with pathology
metadata, validation, folds, preprocessing, patch sampling, the leakage
audit query 04 (moved from phase 3, D-012), and their tests.
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
  stored in `fold_assignments`, numbered 1 to 5 (D-010). `data.py` is
  protected (section 5), so edits prompt for approval. The required tests
  say folds are disjoint and every image is `test` exactly once.
- `sql/queries/04_leakage_audit.sql` (protected) returns rows only if an
  image has two roles in one fold or is `test` in two folds. Test that it
  returns zero rows on a valid database and rows on a leaky one. The schema
  deliberately allows those leaky rows (D-007).
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

- Reported runs use a T4 on the owner's paid Colab plan (D-013). No other
  GPU decision is open.
- Everything from the phase 0 review and the Claude review on PR #1 is
  settled (D-010 to D-014). The owner declined a stricter commit-msg hook, since
  the `/commit` skill covers the subject rules.
- Phase 2 only: confirm the VS Code Colab extension's runtimes set
  `COLAB_RELEASE_TAG`, which `compute_platform` relies on. Colab installs
  with `uv sync --locked` (D-011).

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
