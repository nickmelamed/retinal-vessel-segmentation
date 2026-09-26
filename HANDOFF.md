# Handoff for planning phase 0

This note is for a new Claude Code session that will plan phase 0 (setup) with
the owner. Read CLAUDE.md first, then PROGRESS.md, then the SPEC sections
named below. Plan only. Wait for the owner's approval before building
anything, and do the work on a `phase/0-setup` branch.

## Where things stand (2026-09-26)

`main` is at `6efae69` and matches `origin/main`. There is no project code
yet. The repo holds only the agent setup.

- CLAUDE.md has the always-on rules. docs/SPEC.md has the design, with the
  original section numbers. docs/CONTRIBUTING.md has the git conventions, and
  PROGRESS.md has the phase list. Every phase is unchecked, including 0.
- `.claude/` has the path-scoped rules, the spec-reviewer and style-reviewer
  agents, the `/finish-phase`, `/release`, and `/colab-run` skills, the
  protected-paths list, and the hook settings.
- `scripts/agent/` has the hook scripts. They are already running in Claude
  Code sessions for this repo.
- The Makefile has only `help` and `agent-check`, and `agent-check` runs only
  `check_style.py`. `.pre-commit-config.yaml` has only the local check-style
  hook.
- `data/` is gitignored except `data/CHECKSUMS.sha256`, which is committed.
  It lists 100 DRIVE files with paths relative to `data/` (20 each of test
  images, test masks, training images, training labels, and training masks).
  The DRIVE files themselves are on disk under `data/DRIVE/`.
- README.md holds only the title from the initial commit.

## What phase 0 must deliver

The phase 0 line in PROGRESS.md lists the scaffolding. SPEC sections 6, 13,
14, and 17 give the details, and docs/CONTRIBUTING.md covers the hooks and
branch rules. PROGRESS.md also lists these follow-ups:

1. Extend `make agent-check` to the full list in PROGRESS.md (ruff, mypy,
   fast pytest, the style check, and the numbers check). Register the `slow`
   pytest marker in `pyproject.toml`.
2. Extend `.pre-commit-config.yaml` with the hooks in docs/CONTRIBUTING.md,
   and keep the local check-style hook.
3. Add `hypothesis` and `mutmut` as dev dependencies. The owner has approved
   this.
4. Add a CI workflow that has Claude review every pull request. The owner has
   approved it. It needs an `ANTHROPIC_API_KEY` repository secret, which the
   owner adds by hand, so list it in `docs/REPO_SETTINGS.md`.
5. Restyle docs/CONTRIBUTING.md so it passes `check_style.py`, then remove it
   from `.claude/style-ignore`. It was copied word for word from the old
   CLAUDE.md and fails on bold lead-ins and semicolons.
6. `make check-data` must verify the existing `data/CHECKSUMS.sha256` and not
   overwrite it. The file is protected.

## Decisions to raise with the owner while planning

- Which license to use for the code. SPEC section 17 says to ask before
  adding a LICENSE.
- The Python version for `.python-version`. The default `python3` here is
  3.14, and TensorFlow may not support it yet. Check which versions current
  TensorFlow supports, and match what Colab runs, before picking one. uv can
  see 3.11, 3.12, 3.13, and 3.14.
- Whether the phase 0 CI should install R and Quarto yet, or wait until
  phase 8 adds the report.

## Local environment

These are installed: `uv` 0.11.14, `gh`, Docker, and `R`. A `ruff` from a
miniforge install is on the PATH, but the project should use the version
locked in `uv.lock`. These are missing: `pre-commit`, `mypy`, `Rscript`, and
`quarto`. `make setup` should install what it can through uv.

## Working notes

- Changes to protected files (hooks, CI, lockfiles, `data/CHECKSUMS.sha256`,
  docs/SPEC.md) trigger an approval prompt. That is expected.
- The Bash guard matches blocked flags anywhere in the command text,
  including inside commit messages written in a heredoc. Write commit
  messages to a scratch file and commit with `git commit -F <file>`.
- Ask before pushing, opening a pull request, or changing the lockfile.
- This file carries context between sessions for the rest of the project.
  At the end of a session or phase, rewrite it for the next piece of work
  and replace anything that is out of date.
