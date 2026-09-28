---
name: report-run
description: Bring a finished Colab run onto the laptop, verify it, evaluate it at its tag, mark it reported, and regenerate the tables and figures. Use after /colab-run, once the output zips are in ~/Downloads.
---

Turn a finished Colab run into a reported run (SPEC section 17, D-022). Stop
and tell the owner at the first failed check. Never patch a run, its files,
or its database rows to make a check pass.

1. List `~/Downloads/results.zip` and `~/Downloads/model_fold_*.zip`, run
   `unzip -tq` on each, and check that the contents are there: the database,
   the run's `manifest.json`, one `.npy` per labeled image, a
   `fold_<k>.sha256` per fold, and one checkpoint per fold. Tell the owner
   to keep the Colab session connected until step 4 passes.
2. Move the live `results/experiments.db` aside without overwriting
   anything, for example to `results/dev_experiments.db`. Check first that
   the target does not exist.
3. Unzip `results.zip` into the repo root, and each checkpoint with
   `unzip -qj` into `models/<run_id>/`.
4. Run `make verify-checkpoints RUN=<run_id>`. In
   `results/<run_id>/predictions/`, run `shasum -a 256 -c fold_<k>.sha256`
   for every fold. Read the `runs` row. It must show the expected tag and
   commit, `git_dirty = 0`, `Tesla T4` on `colab`, deterministic ops on, and
   every fold complete. The owner can now disconnect Colab.
5. Evaluation runs only from a clean checkout of the run's own commit
   (D-021). Stash any local edits with a message saying whose they are,
   then `git checkout <tag>` and confirm `git status --short` is empty.
6. Run `make evaluate RUN=<run_id>`. Check that `evaluation.json` records
   the run's commit with `dirty: false`, and that no evaluated column is
   NULL.
7. Switch back to the working branch and `git stash pop`. Never drop the
   owner's stash.
8. Run `make mark-reported RUN=<run_id>`. It lists every failed check. If any
   fails, stop and show them to the owner.
9. Run `make tables figures`. Copy `results/tables` and `figures` aside, run
   it again, and require no difference. Every figure must be under the
   1 MB limit, and `make figures` fails if one is not. Shrink the layout,
   never the limit.
10. Read every table, and look at every figure before writing anything
    about them. Nothing held out informs a decision. Describe failures by
    what is visible, without guessing causes.
11. Commit `results/tables` and `figures` as `exp:`, naming the run, tag,
    GPU, and the checks that passed. Add how the run went to the phase's
    decision entry.
12. Documents quote numbers only from `results/tables`. Run
    `scripts/agent/check_numbers.py` after every edit to them.
