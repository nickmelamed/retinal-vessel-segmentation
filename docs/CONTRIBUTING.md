# Contributing

## 15. Git and commit discipline

### Commits

- **Conventional Commits:** `type(scope): subject`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`, and `exp` (experiment configs and reported runs). Scopes match modules or layers: `data`, `preprocess`, `model`, `train`, `metrics`, `uncertainty`, `external`, `db`, `sql`, `r`, `anomaly`, `figures`, `site`, `docs`, `ci`.
- **Subject line:** imperative mood, lowercase after the colon, no trailing period, at most 72 characters. Example: `feat(metrics): add width-stratified vessel sensitivity`.
- **Body** (required for anything non-trivial): explain *what changed and why*, not how. Wrap at 72. Reference `docs/DECISIONS.md` entries when a commit implements a design decision. Add `BREAKING CHANGE:` in the footer when a config key, schema, or CLI changes incompatibly.
- **Atomic:** one logical change per commit. Code and its tests go in the same commit. Formatting-only changes go in their own commit. Never mix a refactor with a behavior change.
- **Every commit passes** lint and tests. Pre-commit hooks must pass; never use `--no-verify`.
- **Never commit:** DRIVE data, model checkpoints, the experiments database, notebook outputs, secrets, credentials, or files over 1 MB (enforced by hooks). Committed exceptions: `data/CHECKSUMS.sha256`, `results/tables/`, `results/release/` (metrics-only snapshots of reported runs), and final `figures/` (including the GIF).

### Pre-commit hooks (`.pre-commit-config.yaml`)

ruff (lint + format), mypy, sqlfluff, nbstripout, `check-added-large-files` (1 MB), `detect-private-key`, `check-merge-conflict`, `end-of-file-fixer`, `trailing-whitespace`, `check-yaml`, and a Conventional Commits message check. Add R `styler`/`lintr` hooks once the R layer exists.

### Branches, pull requests, and releases

- `main` is protected and always green. Work happens on one branch per phase: `phase/<n>-<slug>` (e.g. `phase/1-data`). Small fixes use `fix/<slug>`.
- Each phase merges through a pull request that uses the template: what changed, why, how it was tested, and any results or decisions. CI must pass. Merge with a merge commit (not squash) so the atomic history is preserved.
- **Releases** use semantic versioning with annotated tags (`v0.1.0`, …) and a `CHANGELOG.md` entry. Any commit whose runs are reported in documents must be reachable from a release tag; the README states which tag produced its numbers.

### What Claude Code may do without asking

Create branches, stage, and commit locally, following the rules above. **Ask first** before pushing, opening or merging pull requests, creating tags or releases, or changing hooks, CI, or the lockfile. Never force-push, rewrite pushed history, or amend a commit that has been pushed.
