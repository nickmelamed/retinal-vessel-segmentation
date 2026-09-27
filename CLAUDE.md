# Retinal vessel segmentation on DRIVE

A self-directed learning project on public data. A small U-Net (TensorFlow/Keras)
segments vessels in the DRIVE fundus images. It is evaluated with image-level
5-fold cross-validation, zero-shot external validation on STARE and CHASE_DB1,
and per-pixel uncertainty maps, and every result is reported honestly,
including weaknesses. Supporting layers are SQLite (run records and leakage
audits), R (statistics), and an LSTM autoencoder that detects simulated
narrowings in vessel width profiles.

The full design is in docs/SPEC.md, which keeps the original section numbers,
so "section 5" means SPEC.md section 5. Read the relevant section before
changing anything it covers. Status and next steps are in PROGRESS.md.
Read HANDOFF.md at the start of every session for context from the last one.
Decisions and their reasons are in docs/DECISIONS.md. Git conventions are in
docs/CONTRIBUTING.md.

## Non-negotiable rules

1. Framing. This is a learning project on public data. Nothing may suggest
   clinical experience, clinical validation, or clinical use. Never write
   "diagnostic", "clinically validated", "vascular surgery model", or similar.
   The accurate description is vessel segmentation.
2. No invented numbers. Every number in the README, model card, or reports
   comes from the database or a file under `results/`. If it does not exist
   yet, write TBD. `scripts/agent/check_numbers.py` enforces this.
3. Held-out data never informs a decision. Thresholds, early stopping,
   hyperparameters, preprocessing, and anomaly thresholds are chosen on
   training or validation images only. External datasets are touched only
   after the model is frozen (section 5), and nothing is tuned on them. If you
   notice a decision informed by held-out or external results, stop and flag it.
4. No data in git. No images from any dataset. `data/` is gitignored. The
   README links to each official source and cites each paper.
5. Report negative results plainly. Compare to published DRIVE results only in
   general terms, and note that CV numbers are not comparable to results on
   the official test split.
6. Ask before changing the evaluation design in section 5. The files that
   implement it are listed in `.claude/protected-paths`.
7. Reported results come only from clean, tagged commits, with the data
   checksum recorded (section 17). Dirty-tree runs are fine for development
   and are flagged in the database.
8. Never weaken a test, lint rule, audit query, or check to make it pass.
9. Dataset facts in section 4 are verified. Do not contradict or paraphrase
   them, and quote the official abnormality notes verbatim.

## Commands

```bash
make setup          # uv sync --locked, renv::restore(), pre-commit install
make check-data     # validate datasets against data/CHECKSUMS.sha256
make train VARIANT=baseline
make test           # pytest with coverage
make lint           # ruff, mypy, sqlfluff, lintr
make smoke          # end-to-end run on synthetic data
make ci             # exactly what CI runs
make agent-check    # the fast checks the Stop hook runs
```

`make help` lists everything. Dependencies change only through `make lock`.

## Where things live

- `src/retinal_vessels/`: the package. One loader adapter per dataset in
  `datasets/`. Anomaly module in `anomaly/`.
- `sql/schema.sql` and `sql/queries/`: the system of record and audits.
  Queries 04 and 09 must return zero rows.
- `R/analysis.qmd`: statistics. `site/`: the Quarto results site.
- `tests/`: synthetic fixtures only, never real data.
- `.claude/rules/`: standards for Python, SQL, R, tests, and writing that load
  when you touch those files.
- `.claude/skills/`: `/finish-phase`, `/colab-run`, `/release`.

## How to work here

- Work one phase at a time on `phase/<n>-<slug>`, following PROGRESS.md. Plan
  first when a task touches several modules, and wait for approval.
- A task is done when the Stop hook's checks pass and you have shown the
  output. Show evidence, not claims.
- Small atomic Conventional Commits. Code and its tests in one commit.
  Never `--no-verify`, never force-push, never rewrite pushed history.
- You may branch and commit locally. Ask before pushing, opening or merging
  pull requests, tagging, or changing hooks, CI, or the lockfile.
- Library code logs with `logging`, never `print`. No magic numbers, all
  settings come from `configs/*.yaml`.
- Colab sessions are disposable. Never edit code there (see `/colab-run`).
- If you repeat a multi-step procedure, or I give you the same instructions
  twice, propose a skill for it in `.claude/skills/` and wait for approval.
- When compacting, keep the modified files, the current plan, the current
  phase, and any failing checks.
