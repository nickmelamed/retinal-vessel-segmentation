# Handoff for planning phase 5

This note is for a new Claude Code session that will plan phase 5 (the
ablations and rerun variance) with the owner. Read CLAUDE.md first, then
PROGRESS.md, then SPEC sections 5, 7, 9, 16, and 17, and D-016, D-019,
D-020, D-021, and D-022 in docs/DECISIONS.md. Plan only, and wait for the
owner's approval before building. Cut `phase/5-ablations` from `main` once
phase 4 has merged and `v0.1.0` is tagged.

## Where things stand (2026-09-28)

Phase 4 is on `phase/4-ship`. Check with `gh pr list` whether its PR has
merged, and with `git tag` whether `v0.1.0` exists. The release itself
(`/release v0.1.0`) runs after the merge. It makes the snapshot with
`make snapshot TAG=v0.1.0`, regenerates the tables and figures, moves the
changelog entries, ticks phase 4 in PROGRESS.md, and asks the owner to tag.

What phase 4 added:

- The first reported run, `20260928T013730Z-4f5d08`: the baseline, trained
  on a Colab Tesla T4 from the tag `v0.1.0-rc.1` (commit `0452dba`) with
  deterministic ops on, and evaluated on the laptop from a clean checkout of
  that tag. It is the only row with `is_reported = 1`.
- `configs/reporting.yaml` with its own strict model. It is kept out of the
  variant configs, so a run's stored config never changes when presentation
  settings do.
- `make mark-reported RUN=<id>` (`retinal_vessels.reporting`), which sets
  `is_reported` only after every check in D-022 passes.
- `make snapshot TAG=<version>`, which writes `results/release/`.
- `make tables` (`retinal_vessels.tables`), which writes
  `results/tables/*.md`. Documents copy every number from these files, and
  `check_numbers.py` checks them.
- `make figures` (`retinal_vessels.figures`) and `make presentation`
  (`scripts/make_social_preview.py`). The figures are committed with JSON
  sidecars.
- The README, the first MODEL_CARD.md, REPO_SETTINGS, and D-022. The title
  is now "Retinal vessel segmentation on DRIVE", at the owner's request.

The main results, in `results/tables`, are a mean Dice of 0.794 and much
lower sensitivity on thin vessels than on thick ones. Image 34 is the one
clear failure. Every fold set the same thin/thick edge.

## What phase 5 must deliver

The PROGRESS.md line: `no_clahe` and `dice_only` runs, query 03, and a
rerun-variance measurement. Each ablation is a full 5-fold run with the
baseline's fold assignment (SPEC section 7, D-019), on the same GPU type
(D-013). Query 03 already pairs runs only when their fold assignments and
data hashes match. The R paired tests come in phase 8, so phase 5 reports
the paired per-image deltas descriptively.

## Decisions to raise with the owner

Raise these before any other phase 5 work.

- The configs `no_clahe.yaml` and `dice_only.yaml` are new files, so the
  ablation runs need a new tagged commit. Which tag should they use, for
  example `v0.2.0-rc.1`? The baseline trained from `v0.1.0-rc.1`. The
  training code has not changed since, but the ablations would then compare
  runs from different commits. One option is to rerun the baseline at the
  new tag too, which also gives one rerun-variance sample.
- `make tables` refuses two reported runs of one variant (D-022), since a
  document could not tell which to quote. Rerun variance needs at least two
  runs of the baseline. Decide how they are recorded and tabulated: for
  example, a headline from one designated run plus a separate rerun table,
  or a way to mark which run documents quote.
- How many reruns, with which seeds. SPEC section 17 asks for the measured
  variance with the same seed. On the T4 with deterministic ops on, the
  same seed may give identical numbers, which is itself worth reporting.
  The top-level seed also drives the weights and patches, so a different
  seed measures seed variance instead. The two answer different questions.
- Whether the ablation tables and figures change the README now, or wait
  for the R report in phase 8.

## Working notes

- The Colab extension cannot download a file over about 512 MB after
  encoding, and each checkpoint is about 90 MB. Zip `results/` on its own
  and each checkpoint on its own (see D-022). The notebook cannot run a
  cell while training runs, so mid-run downloads go through the Colab
  terminal (command palette, Colab: Open Terminal). The Colab Contents view
  opens from the command palette (View: Open View..., then Colab, Contents),
  and right-click, Download... saves a file.
- The runner notebook sets `MPLBACKEND=Agg`, since Colab's inline backend
  broke TensorFlow's import.
- Each fold logs `meta_optimizer.cc:967] layout failed` at error level. It
  is harmless (D-022).
- Evaluation needs a clean tree at the run's tag. Stash any local notebook
  edits first, and pop them afterwards. The owner's edits from the rc.1
  session are in `git stash list`, and nothing needs them.
- The old development database is at `results/dev_experiments.db`. The live
  `results/experiments.db` holds the Colab run.
- The Stop hook counts assertions per file against `main`. Moving tests
  between files trips it even when nothing is weakened. Ask the owner, and
  land the move on `main` first, as PR #8 did.
- `make figures` fails on any PNG over the hook's 1 MB limit. Shrink the
  layout, never the limit. `best_worst.png` is the largest, at 763 KB.
- The tests for the reporting code share the `finished` and `reportable`
  fixtures in `tests/fixtures/evaluated_run.py`.
- Protected files trigger an approval prompt, which is expected. Edit them
  with the Edit tool. Write commit messages to a scratch file and commit
  with `git commit -F`.
- This file carries context between sessions. At the end of a session or
  phase, rewrite it for the next piece of work and replace anything out of
  date.
