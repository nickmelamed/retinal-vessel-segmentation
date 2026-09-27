# Progress

## 18. Build phases

At the end of each phase, run /finish-phase.

Phase 0 extended the `agent-check` target in the Makefile to the full list below, with each tool run through `uv run` (D-003).

```make
agent-check:  ## Fast checks the Claude Code Stop hook runs
	ruff check . && ruff format --check .
	mypy src
	pytest -x -q -m "not slow" --no-cov
	python3 scripts/agent/check_style.py .
	python3 scripts/agent/check_numbers.py README.md MODEL_CARD.md docs/one_pager.md --sources results/tables results/release
```

Phase 0 should also mark slow tests with `@pytest.mark.slow` and register the marker in `pyproject.toml`, and extend `.pre-commit-config.yaml` with the hooks listed in docs/CONTRIBUTING.md.

The owner approved two additions on 2026-09-26. Build them in the phases named here.

- Phase 0: add `hypothesis` and `mutmut` as dev dependencies in `pyproject.toml` and `uv.lock`. Add a Claude review of every pull request to CI as its own workflow. It needs an `ANTHROPIC_API_KEY` repository secret, which the owner adds by hand, so list that in `docs/REPO_SETTINGS.md`.
- Phase 1: run `mutmut` on `data.py` and the fold code, and add tests for any surviving mutants.
- Phase 3: add Hypothesis property tests for `metrics.py`. Dice stays in [0, 1], pixels outside the FOV never change a score, and a perfect prediction scores 1. Run `mutmut` on `metrics.py`.

- [x] **0. Setup:** package skeleton, `pyproject.toml`, `uv.lock`, generated `requirements.txt`, `.python-version`, `.gitignore`, `.gitattributes`, pre-commit hooks, `Makefile`, CI workflow, pull request template, synthetic fixtures, `DECISIONS.md`, `CHANGELOG.md`, `CITATION.cff`, data download instructions, `make check-data` with checksums, database schema, provenance module. CI green on an empty-but-wired pipeline.
- [ ] **1. Data:** DRIVE loader adapter with pathology metadata, validation, folds, preprocessing, patch sampling, the leakage audit query 04 (moved from phase 3, D-012), and their tests.
- [ ] **2. Model and training:** U-Net, losses, resumable CV training with database logging and run manifests, `colab_runner.ipynb`, the smoke integration test, and the resumability tests.
- [ ] **3. Evaluation:** sliding-window inference, all metrics, calibration, thin/thick sensitivity, and SQL queries 01–03 and 05 (04 lands in phase 1).
- [ ] **4. v0.1.0 ship point.** A complete, honest first version that can be shared while later phases continue:
  - Reported baseline runs from a clean tree; `make snapshot`; `make tables`.
  - Hero figure, best/worst figures, training curves, reliability diagram, thin/thick chart.
  - README first screen, results, clinical data considerations, governance, and reproduce sections, with later sections (external validation, uncertainty, R report, anomaly module) listed briefly under "In progress" rather than left as empty headings.
  - `MODEL_CARD.md` first version, badges, social preview image, and `docs/REPO_SETTINGS.md`.
  - Final check of every document against section 2's rules; then ask the owner to tag `v0.1.0`. **Stop here and tell the owner the repo is ready to share.**
- [ ] **5. Ablations:** `no_clahe` and `dice_only` runs, query 03, and a rerun-variance measurement.
- [ ] **6. Frozen model and uncertainty:** `make freeze` with the `frozen_models` record committed; TTA uncertainty, risk–coverage, query 08, the uncertainty panel, and their tests.
- [ ] **7. External validation:** verify STARE and CHASE_DB1 terms (log in `DECISIONS.md`), loader adapters with patient grouping, FOV generation and its validation against DRIVE, zero-shot evaluation of the frozen model, queries 07 and 09, and external figures.
- [ ] **8. R report and results site:** `analysis.qmd` (including external validation with the cluster bootstrap, and uncertainty analyses), `renv`, R hooks, CI render check, the Quarto site, and `pages.yml`.
- [ ] **9. Anomaly module:** profiles, stenosis injection, LSTM autoencoder, baseline, evaluation, and query 06.
- [ ] **10. Presentation and docs:** the GIF, the Mermaid diagram, all README sections filled in (removing "In progress"), updated MODEL_CARD, vascular extension note, and one-pager PDF.
- [ ] **11. Optional:** official DRIVE test-set leaderboard submission with the frozen model, reported separately.
- [ ] **12. Release:** `Dockerfile`; fresh-clone `make reproduce` check (inside Docker); a pass through section 2's rules against every document; final reported runs from a clean tree; a new release snapshot; tag `v1.0.0` (after asking) and cite that tag in the README.
