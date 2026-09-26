# Contributing

## 15. Git and commit discipline

### Commits

Commit messages follow Conventional Commits, written as `type(scope): subject`. The types are `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`, and `exp` (experiment configs and reported runs). Scopes match modules or layers: `data`, `preprocess`, `model`, `train`, `metrics`, `uncertainty`, `external`, `db`, `sql`, `r`, `anomaly`, `figures`, `site`, `docs`, `ci`, and `agent` for the Claude Code tooling.

The subject line is in the imperative mood, lowercase after the colon, with no trailing period, and at most 72 characters. For example, `feat(metrics): add width-stratified vessel sensitivity`.

Anything non-trivial needs a body that explains what changed and why, not how. Wrap it at 72 characters. Reference `docs/DECISIONS.md` entries when a commit implements a design decision. Add `BREAKING CHANGE:` in the footer when a config key, schema, or CLI changes incompatibly.

Each commit holds one logical change. Code and its tests go in the same commit. Formatting-only changes go in their own commit. Never mix a refactor with a behavior change.

Every commit passes lint and tests, and the pre-commit hooks must pass. Never use `--no-verify`.

Never commit DRIVE data, model checkpoints, the experiments database, notebook outputs, secrets, credentials, or files over 1 MB. The hooks enforce this. The committed exceptions are `data/CHECKSUMS.sha256`, `results/tables/`, `results/release/` (metrics-only snapshots of reported runs), and the final `figures/`, including the GIF.

### Pre-commit hooks

`.pre-commit-config.yaml` runs these hooks:

- ruff lint and ruff format, mypy, and sqlfluff, through `uv run` so they use the locked versions
- nbstripout
- `check-added-large-files` (1 MB), `detect-private-key`, `check-merge-conflict`, `end-of-file-fixer`, `trailing-whitespace`, and `check-yaml`
- a Conventional Commits check on the commit message
- the comment and doc style check in `scripts/agent/check_style.py`

`make setup` installs both the pre-commit and commit-msg hooks. R `styler` and `lintr` hooks are added once the R layer exists in phase 8.

### Branches, pull requests, and releases

`main` is protected and always green. Work happens on one branch per phase, named `phase/<n>-<slug>` (for example `phase/1-data`). Small fixes use `fix/<slug>`.

Each phase merges through a pull request that uses the template, covering what changed, why, how it was tested, and any results or decisions. CI must pass. Merge with a merge commit rather than a squash so the atomic history is preserved.

Releases use semantic versioning with annotated tags (`v0.1.0` and so on) and a `CHANGELOG.md` entry. Any commit whose runs are reported in documents must be reachable from a release tag, and the README states which tag produced its numbers.

### What Claude Code may do without asking

Claude Code may create branches, stage, and commit locally, following the rules above. It asks before pushing, opening or merging pull requests, creating tags or releases, or changing hooks, CI, or the lockfile. It never force-pushes, rewrites pushed history, or amends a commit that has been pushed.
