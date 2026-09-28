# DRIVE project design spec

Design spec for the DRIVE project. Section numbers match the original CLAUDE.md and are referenced from code, docs, and DECISIONS.md.

## 1. Project in one paragraph

A self-directed project on **retinal blood vessel segmentation** using the public **DRIVE** dataset. A small U-Net (TensorFlow/Keras) segments vessels in color fundus photographs. Evaluation is done with image-level cross-validation, **external validation** on independent retinal datasets the model never saw, and **per-pixel uncertainty maps** that show where the model is likely wrong, all reported honestly, including weaknesses. Around the model sit three supporting layers, each with a real job: **SQL** (SQLite) records every run and metric and audits for data leakage, **R** performs the statistical analysis of results, and an **LSTM autoencoder** detects simulated stenosis-like narrowings in vessel width profiles, as a bridge toward vascular imaging problems such as CT angiography.

## 2. Non-negotiable rules

The non-negotiable rules are in CLAUDE.md.

## 3. Audiences

- **Non-technical readers** (e.g. recruiters, managers) read only the README's first screen and look at the figures. It must be understandable in about 15 seconds.
- **Technical reviewers** (healthcare data scientists) will check rigor, metrics, leakage controls, and honesty. Everything below the first screen is for them.

## 4. Dataset facts (verified; do not contradict)

- **Source:** https://drive.grand-challenge.org/ (registration required). Cite: Staal J, Abràmoff MD, Niemeijer M, Viergever MA, van Ginneken B. *Ridge-based vessel segmentation in color images of the retina.* IEEE Transactions on Medical Imaging 23(4):501–509, 2004. doi:10.1109/TMI.2004.825627.
- 40 color fundus images from a diabetic retinopathy screening program in the Netherlands; 33 show no sign of diabetic retinopathy, 7 show mild early signs. Canon CR5 non-mydriatic 3CCD camera, 45° field of view, captured at 768×584 pixels and cropped around the FOV to 565×584 (W×H), 8 bits per channel. **Every image has been JPEG compressed** by the dataset providers; compression can blur the finest vessels, so note it under limitations.
- Official split: 20 training (ids 21–40) and 20 test (ids 1–20). Every image has a circular field-of-view (FOV) mask.
- **Test-set vessel annotations are withheld** on the official site; predictions can be submitted there for scoring (Dice within the FOV mask). Only the 20 training images have labels we can use.
- **Images with abnormalities, as listed on the official site** (retrieved 2026-09-26, D-015): training images **25**, **26**, and **32**, and test images 03, 08, 14, and 17. Store these notes word for word as image metadata (section 8), and don't turn them into diagnoses beyond what the site states.
  ```
  25_training: pigment epithelium changes, probably butterfly maculopathy with pigmented scar in fovea, or choroidiopathy, no diabetic retinopathy or other vascular abnormalities.
  26_training: background diabetic retinopathy, pigmentary epithelial atrophy, atrophy around optic disk
  32_training: background diabetic retinopathy
  03_test: background diabetic retinopathy
  08_test: pigment epithelium changes, pigmented scar in fovea, or choroidiopathy, no diabetic retinopathy or other vascular abnormalities
  14_test: background diabetic retinopathy
  17_test: background diabetic retinopathy
  ```
- Annotators were instructed to mark pixels they were at least 70% certain were vessel. Labels are therefore inherently uncertain at thin-vessel boundaries.
- **Patients:** images were randomly selected from a screening population of 400 diabetic subjects aged 25–90. No patient identifiers are published. Do **not** claim each image is a different patient. Use: "DRIVE does not publish patient identifiers; we treat each image as an independent subject, which we cannot verify."
- **License:** no explicit license is published for DRIVE. Do not redistribute images. Showing a few example images in figures, with attribution, is acceptable.
- Vessel pixels are roughly 1 in 8 pixels. Compute the exact within-FOV fraction from the data and quote the computed value.
- **Expected file layout** (verify after download; the loader must fail loudly on mismatch):
  ```
  data/DRIVE/training/images/21_training.tif ... 40_training.tif
  data/DRIVE/training/1st_manual/21_manual1.gif ...
  data/DRIVE/training/mask/21_training_mask.gif ...
  data/DRIVE/test/images/01_test.tif ... 20_test.tif
  data/DRIVE/test/mask/01_test_mask.gif ...
  ```
  The data root is configurable via the `DRIVE_DIR` environment variable.

### External validation datasets (verify before use)

The facts below are as commonly reported in the literature. Before downloading either dataset, verify its official source, access terms, and license, and record them in `docs/DECISIONS.md`. If a dataset's terms don't permit this use, drop it and tell the user.

- **STARE** (Hoover et al., 2000): 20 fundus images at 700×605 pixels, about half with pathology; vessel annotations from two observers (use the first, Hoover's, as reference and state this). No FOV masks are distributed; generate them (section 5).
- **CHASE_DB1** (Owen et al., 2009; Fraz et al., 2012): 28 images from **both eyes of 14 children**, 999×960 pixels; two observers (use the first as reference). No FOV masks are distributed. Because each child contributes two images, **every analysis on CHASE_DB1 must group by child**, not by image.
- Add a loader adapter per dataset (section 6), with the same fail-loudly validation as DRIVE, and add their checksums to `data/CHECKSUMS.sha256`.

## 5. Evaluation design (ask before changing)

- **5-fold cross-validation over the 20 labeled images, split by whole image** (never by patch). Each fold: 4 held-out test images; of the remaining 16, 2 are validation (early stopping and threshold choice) and 14 are training. The abnormal training images 25, 26, and 32 go to three different test folds (D-016). Folds are deterministic from the seed and stored in the database.
- Every labeled image receives exactly one **out-of-fold** prediction. All headline metrics are computed on these 20 out-of-fold predictions.
- **Threshold:** chosen per fold to maximize Dice on that fold's validation images, then applied unchanged to its held-out images. Record every threshold and the rule used.
- **All metrics are computed inside the FOV mask only.**
- **Frozen final model:** after cross-validation is complete, train one final model on all 20 labeled DRIVE images, using the median best epoch across folds as the fixed training length and the median of the fold thresholds as its threshold. Record the checkpoint's SHA-256 and the commit in `docs/DECISIONS.md`, and commit that record **before** any external data is evaluated. From then on the model and threshold never change.
- **External validation (zero-shot):** apply the frozen model to STARE and CHASE_DB1 with no retraining, fine-tuning, or threshold changes. Preprocessing matches DRIVE's, plus one documented, data-independent step: rescale each image so its FOV diameter matches DRIVE's (about 540 px), so vessels appear at a similar scale. Generate FOV masks by thresholding the red channel with a fixed rule chosen on DRIVE, and validate that rule against DRIVE's official masks (report the agreement). Report the same metrics as for DRIVE, alongside the internal cross-validation results, and describe the drop as a measurement of dataset shift. For CHASE_DB1, bootstrap CIs resample **children**, not images.
- **Optional, clearly separated:** submit the frozen model's predictions on the official DRIVE test set to the grand-challenge leaderboard. Report that score separately and label it as the official test score.

### Metrics

Per image, then summarized (mean, SD, median, min, max, and bootstrap 95% CIs across images):

- Dice/F1, sensitivity (recall), specificity, precision.
- **AUC-ROC and AUC-PR.** Explain that AUC-ROC is inflated by the easy background under class imbalance, and that AUC-PR is more informative.
- Accuracy, with an explicit note on why it's misleading here: a model predicting "no vessel" everywhere scores near the majority-class rate.
- **Calibration:** Brier score and a reliability diagram from pooled out-of-fold probabilities.
- **Pathology subgroup:** out-of-fold metrics for training images 25, 26, and 32 versus the other 17, reported descriptively (per image, not as a statistical test; n = 3 is far too small for inference). Say so plainly.
- **Thin versus thick vessel sensitivity:** skeletonize the ground truth, estimate vessel width with a distance transform, bin skeleton pixels by width (e.g. ≤2 px radius as thin; tune the bins on training data and document them), and report sensitivity per bin.

### Uncertainty

- **Method:** test-time augmentation (TTA) over the 8 flip/rotation symmetries, with probabilities mapped back to the original orientation. Per-pixel uncertainty is the predictive entropy of the averaged probability (also store the standard deviation across augmentations). Monte Carlo dropout is an optional second method; if added, compare the two.
- **Primary predictions stay single-pass** so headline metrics are unaffected. The TTA-averaged prediction is reported separately as a secondary result.
- **Analyses** (within FOV, on out-of-fold predictions and on each external dataset):
  - How well uncertainty identifies error pixels: AUC-ROC of uncertainty for separating misclassified from correct pixels.
  - **Risk–coverage curve:** Dice on the retained pixels as the most uncertain X% are set aside for human review (X from 0 to 20%). This is the "refer uncertain cases to a person" framing a clinical team understands.
  - Per-image mean uncertainty versus per-image Dice (Spearman correlation), and whether mean uncertainty rises on the external datasets relative to DRIVE (a candidate signal for detecting dataset shift).

## 6. Repository layout

```
.
├── CLAUDE.md
├── README.md
├── MODEL_CARD.md
├── CHANGELOG.md              # Keep a Changelog format, one entry per release
├── CITATION.cff              # how to cite this repo (and every dataset paper)
├── LICENSE                   # code license only; the data has its own terms
├── pyproject.toml            # package metadata, deps, ruff/mypy/pytest/coverage config
├── uv.lock                   # locked Python environment (source of truth)
├── requirements.txt          # generated from uv.lock for pip users; never hand-edited
├── .python-version           # pinned Python version
├── .pre-commit-config.yaml
├── .gitattributes
├── .gitignore
├── Makefile                  # canonical entry points (make help)
├── Dockerfile                # CPU reproduction environment (Python + R + Quarto)
├── .github/
│   ├── workflows/ci.yml      # lint, type-check, tests, smoke run, lockfile check
│   ├── workflows/pages.yml   # builds the results site from the committed release snapshot
│   └── pull_request_template.md
├── configs/                  # one YAML per experiment variant
│   ├── baseline.yaml
│   ├── no_clahe.yaml
│   ├── dice_only.yaml
│   ├── final.yaml            # frozen final model trained on all 20 DRIVE images
│   ├── external.yaml         # external validation settings (datasets, FOV rule, rescaling)
│   └── smoke.yaml            # tiny config for CI on synthetic data
├── src/retinal_vessels/      # installable package (pip/uv install -e .)
│   ├── __init__.py           # exposes __version__
│   ├── config.py             # typed, validated config loaded from YAML
│   ├── data.py               # common sample type, validation, fold construction
│   ├── datasets/             # one loader adapter per dataset
│   │   ├── drive.py
│   │   ├── stare.py
│   │   └── chase.py          # carries child ID for patient-level grouping
│   ├── fov.py                # FOV mask generation for datasets without masks
│   ├── preprocess.py         # green channel, CLAHE, normalization
│   ├── patches.py            # random patch sampling + augmentation (tf.data)
│   ├── model.py              # small U-Net
│   ├── losses.py             # Dice, BCE+Dice
│   ├── train.py              # CLI: runs full k-fold CV for one config
│   ├── predict.py            # sliding-window full-image inference
│   ├── uncertainty.py        # test-time augmentation, entropy, risk–coverage
│   ├── external.py           # CLI: zero-shot evaluation of the frozen model
│   ├── metrics.py            # FOV-masked metrics, calibration, width-stratified sensitivity
│   ├── db.py                 # SQLite connection, schema creation, write helpers
│   ├── figures.py            # all figure generation
│   ├── provenance.py         # run manifest: git state, data checksum, env versions
│   ├── utils.py              # seeding, logging setup
│   └── anomaly/
│       ├── profiles.py       # skeleton → branches → width/intensity/curvature sequences
│       ├── stenosis.py       # synthetic stenosis injection into masks
│       ├── model.py          # LSTM autoencoder
│       ├── baseline.py       # rolling z-score detector
│       └── evaluate.py       # CLI: detection metrics, logs to DB
├── sql/
│   ├── schema.sql
│   └── queries/              # numbered, commented, runnable .sql files
├── R/
│   ├── analysis.qmd          # Quarto statistical report
│   └── renv.lock
├── scripts/
│   ├── check_data.py         # sanity check + dataset stats
│   ├── make_tables.py        # DB → markdown tables used in README
│   ├── make_social_preview.py  # 1280×640 repo preview image from the hero figure
│   ├── make_gif.py           # animated prediction overlay for the README
│   ├── snapshot_db.py        # export reported runs to results/release/ (metrics only)
│   └── submit_official.py    # optional leaderboard predictions
├── notebooks/
│   ├── results_walkthrough.ipynb   # reads artifacts only; trains nothing
│   └── colab_runner.ipynb    # thin Colab driver: clone tag, sync env, run make targets
├── tests/
│   ├── fixtures/             # synthetic DRIVE-shaped data generator (no real data in CI)
│   ├── unit/
│   └── integration/          # end-to-end smoke run on synthetic data
├── docs/
│   ├── DECISIONS.md          # log of design decisions and why
│   ├── vascular_extension.md # how this extends to CTA / ultrasound
│   └── one_pager.md          # one-page summary, rendered to PDF
├── figures/                  # committed final figures
├── site/                     # Quarto website source for GitHub Pages (report + results)
├── results/                  # experiments.db, predictions (gitignored); tables/ and release/ committed
├── data/                     # gitignored; data/CHECKSUMS.sha256 is committed
└── models/                   # gitignored; checkpoints are reproduced, not committed
```

## 7. Segmentation pipeline spec

- **Preprocessing:** green channel → CLAHE (OpenCV; record clip limit and tile size in config) → scale to [0,1] → per-image standardization within the FOV. Every step can be toggled from config to support ablations.
- **Patches:** random 64×64 patches (configurable) sampled from training images, with centers constrained to the FOV. Augmentation: flips, 90° rotations, small brightness/contrast jitter. Implement with `tf.data`, seeded.
- **Model:** a small U-Net (e.g. 3–4 levels, base 16–32 filters, batch norm, dropout). Keep it trainable on a free Colab GPU in minutes per fold and runnable on a laptop CPU with a smaller config.
- **Loss:** BCE + Dice (baseline). Dice-only is an ablation.
- **Training:** Adam, early stopping on validation-image Dice (computed on full validation images, within FOV, not patches), saving the best checkpoint per fold. Log training curves per fold. **Cross-validation is resumable** (section 16): each fold is committed to the database as complete only after its checkpoint, threshold, metrics, and history are all written, and a restarted run skips completed folds.
- **Inference:** sliding window with overlap and averaged probabilities, masked to the FOV.
- **Ablations:** `baseline`, `no_clahe`, `dice_only`. Each is a full 5-fold CV run with an identical fold assignment, so R can make paired per-image comparisons. Don't add more variants without a reason logged in `DECISIONS.md`.

## 8. SQL layer (SQLite at `results/experiments.db`)

Purpose: the system of record for experiments and the governance audit trail. Created from `sql/schema.sql` by `retinal_vessels.db`. Every training and evaluation run writes to it.

Minimum tables (refine columns as needed; keep them in `schema.sql`):

- `runs` — run_id, variant, config (JSON text), config_hash, seed, git_commit, git_dirty, data_hash, python/TF versions, device, gpu_type (e.g. `Tesla T4`, or `cpu`), compute_platform (`colab` / `local` / other), deterministic_ops, started_at, finished_at, is_reported (true only for clean-tree runs used in documents).
- `images` — dataset (`drive` / `stare` / `chase`), image_id, patient_id (child ID for CHASE_DB1; null where unknown), split, has_abnormality, abnormality_note (verbatim from the official site), fov_pixels, fov_source (`official` / `generated`), vessel_fraction_in_fov, has_labels.
- `fold_assignments` — run_id, fold, image_id, role (`train` / `val` / `test`).
- `fold_status` — run_id, fold, status (`running` / `complete`), checkpoint_sha256, started_at, completed_at, attempts. Used for resuming interrupted runs.
- `thresholds` — run_id, fold, threshold, selection_rule, val_dice.
- `per_image_metrics` — run_id, dataset, image_id, fold (null for external), prediction_mode (`single` / `tta`), dice, sensitivity, specificity, precision, accuracy, auc_roc, auc_pr, brier, thin_sensitivity, thick_sensitivity, predicted_vessel_fraction, mean_uncertainty, uncertainty_error_auc.
- `risk_coverage` — run_id, dataset, image_id, fraction_referred, dice_retained.
- `frozen_models` — model_id, run_id, checkpoint_sha256, threshold, n_epochs, git_commit, frozen_at. External evaluations must reference a row here.
- `training_history` — run_id, fold, epoch, train_loss, val_dice.
- `anomaly_runs`, `anomaly_detections` — for the LSTM autoencoder module (section 10).

Queries in `sql/queries/`, each with a header comment stating its question:

- `01_fold_summary.sql` — metrics by fold and variant.
- `02_worst_images.sql` — lowest-Dice images across variants (feeds the failure-mode figures).
- `03_variant_comparison.sql` — per-image paired deltas between variants.
- `04_leakage_audit.sql` — returns rows only if any image has more than one role within a fold, or appears as `test` in more than one fold of a run. **Must return zero rows; a pytest test enforces this.**
- `05_threshold_log.sql` — threshold chosen per fold and variant, with rationale.
- `06_anomaly_recall_by_severity.sql` — detection recall by stenosis severity, LSTM autoencoder versus baseline.
- `07_internal_vs_external.sql` — headline metrics for DRIVE out-of-fold versus each external dataset, from the frozen model.
- `08_uncertainty_vs_error.sql` — per-image mean uncertainty versus Dice, by dataset.
- `09_frozen_model_audit.sql` — returns rows if any external evaluation used a model or threshold not recorded in `frozen_models`, or was run before its freeze time. **Must return zero rows; a pytest test enforces this.**

Don't store images, patches, or arrays in the database.

## 9. R layer (`R/analysis.qmd`)

Purpose: statistical inference on results. It reads `results/experiments.db` through `DBI` + `RSQLite`. Reproducible with `renv`. It renders to HTML (and PDF if a TeX install is available) under `reports/`.

Contents:

- **Bootstrap 95% CIs** (resampling images) for headline metrics per variant.
- **Bland–Altman plot:** predicted versus ground-truth vessel density per image (bias, limits of agreement). Explain in one sentence why this is a standard way to compare measurement methods.
- **Paired variant comparisons:** Wilcoxon signed-rank tests on per-image Dice across the 20 out-of-fold images (baseline versus each ablation), with effect sizes (median paired difference and CI). State plainly that n = 20 limits power, and don't over-interpret p-values.
- **Pathology subgroup:** per-image metrics for images 25, 26, and 32 plotted against the distribution of the other 17. Descriptive only.
- **External validation:** internal versus external distributions of Dice, sensitivity, and AUC-PR, with bootstrap CIs; for CHASE_DB1, a **cluster bootstrap by child**. Describe the drop plainly as dataset shift.
- **Uncertainty:** risk–coverage curves by dataset; Spearman correlation of per-image mean uncertainty with Dice; whether mean uncertainty is higher on the external datasets.
- **Optional:** a mixed-effects model (`lme4`) of Dice with variant as a fixed effect and image as a random effect, if it adds something beyond the paired tests.
- Export the key plots to `figures/` for use in the README.

Use tidyverse style and `ggplot2`. The report must run end to end with one command (see section 13).

## 10. LSTM autoencoder module (`src/retinal_vessels/anomaly/`)

Purpose: detect local vessel narrowings (a simulated analogue of stenosis) from width profiles along vessel centerlines. This is the bridge to vascular imaging, where width along a centerline is the core signal for stenosis grading.

1. **Profiles (`profiles.py`):** skeletonize a vessel mask (`skimage`), split it into branches between junctions and endpoints (`skan`), and order the pixels along each branch. At each centerline point, extract: width (2× distance-transform radius), green-channel intensity, and local curvature. Drop branches shorter than the window length. Cut branches into fixed-length sliding windows (e.g. 32 points, configurable), normalized per feature using training statistics only.
2. **Synthetic stenosis (`stenosis.py`):** choose a branch segment of a given length, erode the *mask* locally to reduce width by a severity (e.g. 30%, 50%, 70%) with a smooth taper, then re-extract profiles from the modified mask. Record the ground-truth location. Injecting into the mask rather than the profile keeps the whole pipeline honest. Add a test confirming the injection reduces the measured width at the target location.
3. **Model (`model.py`):** Keras LSTM autoencoder (LSTM encoder → RepeatVector → LSTM decoder → TimeDistributed Dense), trained only on unmodified profiles from each fold's training images, following the same CV folds as segmentation. Anomaly score = per-window reconstruction error; the alert threshold is set on validation images (e.g. the 99th percentile of normal scores).
4. **Baseline (`baseline.py`):** rolling z-score of width along the branch, with its threshold chosen the same way. The LSTM autoencoder must be compared against it.
5. **Evaluation (`evaluate.py`):** on held-out images with injected stenoses, report recall by severity, false alarms per 1,000 centerline points, and precision. Run the detection twice: once on **ground-truth masks** (best case) and once on **predicted masks** (realistic case, showing how segmentation errors propagate to a downstream measurement). Log everything to the database.
6. **Framing:** retinal vessels with simulated narrowings, not real stenoses; nothing is clinically validated. If the LSTM autoencoder doesn't beat the baseline, report that as a finding.

## 11. Figures (`figures/`, generated by `retinal_vessels.figures` or R)

- **Hero figure** for the README's first screen: original / ground truth / prediction / error map (false positives and false negatives in distinct, colorblind-safe colors) for one representative image. Pick the image closest to median Dice, not the best one, and say so in the caption.
- Best and worst images (by out-of-fold Dice), with a short note on why the worst ones fail (e.g. optic disc boundary, pathology, thin vessels, low contrast). Mark images 25, 26, and 32 wherever per-image results are plotted.
- Training curves per fold.
- Reliability diagram.
- Thin versus thick vessel sensitivity bar chart.
- Bland–Altman plot and variant comparison plot (from R).
- **Uncertainty panel:** image / prediction / uncertainty map / error map for the same image, showing whether uncertainty sits where the errors are.
- **Risk–coverage curves** for DRIVE and each external dataset.
- **External validation:** one example per external dataset (image / ground truth / prediction / error map), and a chart comparing internal and external metrics.
- **Animated GIF** (`scripts/make_gif.py`): 5–8 images cycling through fundus image → predicted vessels overlaid, with a small attribution caption. Optimize to under 1 MB so it passes the large-file hook; if that isn't achievable, ask the owner rather than raising the limit.
- **Social preview image** (`scripts/make_social_preview.py`): 1280×640 PNG built from the hero figure, with the project title. Uploading it is a manual step in the GitHub repo settings; remind the owner.
- Anomaly detection: an example branch width profile with an injected stenosis and the reconstruction error; recall versus severity for the LSTM autoencoder and the baseline.

Every figure has a caption in the README stating what it shows and which images it uses.

## 12. Documentation deliverables

### README.md, in this order

1. **First screen:** title ("Retinal vessel segmentation on DRIVE"); one row of at most five badges (CI status, coverage, Python version, license, and a static "not for clinical use" badge); a 3–4 sentence plain-English summary; the hero figure or GIF; 2–3 headline numbers with CIs pulled from `results/tables/` (include the external-validation result once it exists); and a link to the results site.
2. **How it fits together:** a Mermaid diagram (GitHub renders it natively) showing DRIVE → preprocessing → U-Net cross-validation → SQLite database → R report and results site, plus the frozen model → external validation and uncertainty, and the anomaly module branching from the vessel masks. Keep it under about 12 nodes.
3. Results: the metrics table, spread across images, calibration, thin versus thick sensitivity, the pathology subgroup (descriptive), and ablations. Include the note on accuracy and the comparability caveat.
4. **Does it generalize?** External validation results on STARE and CHASE_DB1, the freeze-before-evaluate procedure, and an honest reading of the drop.
5. **Where is it likely wrong?** The uncertainty panel, the risk–coverage curves, and what they suggest about referring uncertain regions for human review.
6. **Beyond segmentation:** the SQL audit trail (show the leakage and frozen-model audit queries and their zero-row results), the R statistical report, and the LSTM autoencoder anomaly module with its honest results.
7. **Clinical data considerations** (brief and concrete): patient-level splitting and why it matters (and DRIVE's lack of patient IDs); class imbalance; label variability (the 70% certainty instruction; one annotator for training labels); de-identification and HIPAA in real settings; asymmetric costs of false negatives (missed vessels or narrowings) versus false positives; dataset shift (cameras, populations, disease prevalence; DRIVE is one screening program in one country), now pointing to the measured external-validation drop rather than only describing it.
8. **Model governance:** threshold selection rule, calibration, the frozen-model procedure, uncertainty as a referral signal, run manifest and database, leakage audits, tests, and known failure modes.
9. Reproduce: the release tag that produced the reported numbers, `make setup` and `make reproduce`, the Docker alternative, hardware used, runtime, and measured rerun variance.
10. Limitations (including JPEG compression of the source images, a single annotator for training labels, and only three training images with abnormalities), citations for every dataset, and data access (no data in the repo).

### MODEL_CARD.md

Model details; intended use (research and learning; **not for clinical use**); out-of-scope uses; data; evaluation design; metrics with CIs; external validation results; uncertainty behavior; calibration; failure modes with figure references; limitations; ethical considerations; and a separate section for the anomaly module.

### docs/vascular_extension.md

About one page on how this work extends to vascular imaging: 3D CTA (volumetric U-Nets, anisotropic voxels, memory limits, contrast-phase variation, centerline extraction in 3D); ultrasound (speckle, operator dependence, temporal sequences, Doppler waveforms); where autoencoder-based anomaly detection fits (flagging unusual scans, sequences, or waveforms for human review); and what would be needed to validate any of this clinically.

### docs/one_pager.md → PDF

One page for non-technical readers: the hero figure, three sentences on what was built, the headline numbers, the tools used (Python/TensorFlow, SQL, R), and the repo link.

### Results site (GitHub Pages)

A small Quarto website in `site/` combining the R statistical report with the key results and figures. It is built by `.github/workflows/pages.yml` from the committed release snapshot `results/release/experiments_<tag>.db`, which contains **only metrics and metadata from reported runs** (no images, predictions, or arrays), exported by `scripts/snapshot_db.py`. This also lets anyone rerun the R analysis without GPUs or data. Enabling Pages is a manual step in the repo settings; remind the owner.

### GitHub repository presentation

Draft these in `docs/REPO_SETTINGS.md` for the owner to apply manually (Claude Code can't change repo settings):
- **Description:** one line, e.g. "U-Net vessel segmentation on DRIVE with cross-validation, external validation, uncertainty maps, a SQL audit trail, R statistics, and LSTM-autoencoder narrowing detection. Learning project; not for clinical use."
- **Topics:** `medical-imaging`, `image-segmentation`, `retinal-imaging`, `u-net`, `tensorflow`, `uncertainty-quantification`, `model-governance`, `r`, `sqlite`, `reproducible-research`.
- **Social preview image:** the output of `scripts/make_social_preview.py`.
- **Website field:** the GitHub Pages URL.
- **Branch protection on `main`:** require CI to pass before merging.

## 13. Commands

The `Makefile` is the canonical interface; raw commands are listed for reference. Keep both current as entry points are built, and keep `make help` self-documenting.

```bash
# environment
make setup            # uv sync --locked; renv::restore(); pre-commit install
make lock             # uv lock + regenerate requirements.txt (the only way deps change)

# data
make check-data       # validate DRIVE (and external datasets if present); write/verify data/CHECKSUMS.sha256

# experiments (each is a full 5-fold CV run logged to results/experiments.db)
make train VARIANT=baseline      # python -m retinal_vessels.train --config configs/baseline.yaml
make ablations                   # no_clahe + dice_only
make freeze                      # train the final model on all 20 DRIVE images; record it in frozen_models
make uncertainty                 # TTA uncertainty on out-of-fold predictions
make external                    # python -m retinal_vessels.external --config configs/external.yaml
make anomaly                     # python -m retinal_vessels.anomaly.evaluate --config configs/baseline.yaml

# outputs
make tables           # DB → results/tables/*.md
make figures
make report           # quarto render R/analysis.qmd
make snapshot         # export reported runs to results/release/experiments_<tag>.db
make site             # render the Quarto results site locally (CI deploys it)
make presentation     # GIF + social preview image
make reproduce        # everything above, in order, from a clean clone + downloaded data

# quality
make lint             # ruff, ruff format --check, mypy, sqlfluff, lintr
make test             # pytest with coverage
make smoke            # end-to-end run on synthetic data with configs/smoke.yaml
make ci               # lint + test + smoke: exactly what CI runs
```

All training entry points must also run on Colab through `notebooks/colab_runner.ipynb` (section 16).

## 14. Engineering standards

### Python

- Installable package under `src/retinal_vessels/` (src layout), Python version pinned in `.python-version`, dependencies managed with `uv`. `uv.lock` is the source of truth; `requirements.txt` is generated from it for pip users and CI checks that the two agree. Colab installs from `uv.lock` (D-011).
- Type hints everywhere; `mypy --strict` on the package. NumPy-style docstrings on every public function and class, explaining *why* where it isn't obvious, including array shapes and dtypes.
- Small, single-purpose modules with pure functions where possible; I/O at the edges. No logic in notebooks; the notebook only reads artifacts and displays them.
- All settings come from `configs/*.yaml`, loaded into typed, validated config objects by `retinal_vessels.config`. Unknown or missing keys are errors. No magic numbers in code.
- Use the `logging` module (configured once in `retinal_vessels.utils`), never `print`, in library code. CLIs log run IDs and output paths.
- Seed everything through `retinal_vessels.utils.set_seed`. Record whether deterministic GPU ops were enabled.
- Fail loudly on bad inputs (shape, dtype, empty mask, missing file, config mismatch) with specific error messages. Never silently continue.
- Ruff for lint and format (line length 100); `nbstripout` on notebooks.

### SQL

- `sqlfluff` with the SQLite dialect. Uppercase keywords, snake_case identifiers, explicit column lists (no `SELECT *` in committed queries), and a header comment on every query stating the question it answers.
- Schema changes go through `sql/schema.sql` with a `schema_version` table; never alter the database by hand.

### R

- Tidyverse style enforced by `styler` and `lintr`. Package versions locked with `renv`. The Quarto report sets a seed and prints `sessionInfo()` at the end.

### Tests (`tests/`)

- Tests never touch real DRIVE data. `tests/fixtures/` generates small synthetic DRIVE-shaped images, masks, and labels, so CI runs anywhere.
- Coverage is reported on every run; target at least 85% on non-training modules. Don't write hollow tests to hit a number.
- Required unit tests:
  - Metrics on tiny hand-made arrays, with known answers, including that pixels outside the FOV are ignored.
  - Folds are disjoint and cover all 20 images exactly once as `test`.
  - The leakage audit query returns zero rows on a populated test database, and returns rows on a deliberately leaky one.
  - The patch sampler returns the right shapes, with centers inside the FOV.
  - Stenosis injection reduces measured width at the target location, and leaves other branches unchanged.
  - Sliding-window inference returns an output the same shape as the input.
  - Config loading rejects unknown and missing keys.
  - The run manifest captures the commit, dirty flag, and data checksum.
  - TTA maps every augmented prediction back to the original orientation (an asymmetric synthetic image round-trips exactly), and uncertainty is zero when all augmentations agree.
  - Generated FOV masks agree with DRIVE's official masks above a documented threshold.
  - CHASE_DB1 loading assigns both eyes of a child the same patient ID, and the cluster bootstrap resamples children, not images.
  - Resumability: interrupting a synthetic CV run after fold 2 and restarting it completes folds 3–5 only, and yields the same fold assignments and database state as an uninterrupted run.
  - A fold marked `running` but never completed (simulated crash) is retrained on restart, not skipped.
  - Pathology metadata for images 25, 26, and 32 loads correctly.
  - The frozen-model audit query returns zero rows on a valid database, and returns rows when an external evaluation references an unfrozen model.
- Required integration test: the `smoke` config trains, predicts, evaluates, writes to a temporary database, and produces tables end to end on synthetic data in under about two minutes on CPU.

### Continuous integration (`.github/workflows/ci.yml`)

On every push and pull request: `make lint`, `make test`, `make smoke`, a check that `requirements.txt` matches `uv.lock`, and a check that the R report renders against the fixture database. `main` must always be green.

## 15. Git and commit discipline

Git and commit discipline is in docs/CONTRIBUTING.md.

## 16. Compute

- **Development, tests, and the smoke run:** the owner's laptop, on CPU. On Windows with an NVIDIA GPU, TensorFlow GPU support requires WSL2; on a Mac, use CPU rather than the Metal plugin unless there's a clear reason.
- **Reported GPU runs:** Google Colab (paid plan, T4 selected for every reported run, D-013), driven from VS Code through the official Colab extension (Select Kernel → Colab → New Colab Server). The code runs on Colab's machine, not the laptop, so every session is treated as a fresh, disposable environment.
- **Colab workflow** (implemented in `notebooks/colab_runner.ipynb`, which contains no project logic, only these steps):
  1. Clone the repo and check out the **tagged commit** being run. Never edit code in the Colab session; edit locally, commit, push, and pull. This keeps reported runs on clean, tagged commits (section 2).
  2. Install uv and the locked environment with `uv sync --locked`, exactly as CI does, and print the versions, including the GPU (`nvidia-smi`). See D-011.
  3. Place the DRIVE data on the Colab machine (the extension's upload feature; Google Drive mounting isn't supported natively in the extension), then run `make check-data` so the checksums must match before any training.
  4. Run the `make` targets.
  5. Before the session ends, download `results/experiments.db`, `results/<run_id>/` manifests, and checkpoints to the laptop, and verify the checkpoint SHA-256 values against the database.
- **Resumable runs:** Colab sessions can disconnect. Cross-validation must be restartable with the same command: completed folds (per `fold_status`) are skipped, incomplete ones are retrained from scratch. Write the database and checkpoints after each fold, and make downloading them after each fold easy, so a disconnect costs at most one fold.
- **Same hardware for comparable runs:** every run records `gpu_type` and `compute_platform`. All reported segmentation runs (every variant and the frozen model) must use the same GPU type; if Colab assigns something else, record it and don't mix it into reported comparisons. If a different GPU is ever used, say so in the README.
- **Fallbacks, if Colab becomes unworkable:** Colab Pro, Kaggle Notebooks (DRIVE uploaded as a private dataset), or a paid cloud GPU running the `Dockerfile`. Any change of platform goes in `docs/DECISIONS.md`.

## 17. Reproducibility and provenance

- **Environment:** Python locked by `uv.lock` plus `.python-version`; R by `renv.lock`; the Quarto version recorded in the README. The `Dockerfile` builds a CPU environment that can run `make reproduce` end to end; record its base image digest.
- **Data provenance:** `make check-data` computes SHA-256 checksums of every DRIVE file and compares them to the committed `data/CHECKSUMS.sha256`, wherever the data lives. It writes that file only when it is missing and `--init` is passed, and never changes an existing one (D-014). A mismatch is an error. Each run stores an aggregate data hash.
- **Run manifest:** every run writes to the `runs` table and to `results/<run_id>/manifest.json`: run ID, config and config hash, seed, git commit, dirty-tree flag, data hash, Python/TensorFlow/CUDA versions, device, deterministic-ops flag, and start and end times.
- **Artifact lineage:** every table in `results/tables/` and every figure is generated by a script from the database, and records the run IDs it came from (in a caption, footnote, or sidecar file). Regenerating them must be one command (`make tables figures report`).
- **Determinism:** document what is and isn't bit-for-bit reproducible (GPU nondeterminism) and, in the README, how much metrics vary across reruns with the same seed, measured rather than assumed.
- **Citation and licensing:** `CITATION.cff` for the repo, with the DRIVE, STARE, and CHASE_DB1 papers referenced; `LICENSE` covers the code only (ask the owner which license before adding it), and the README explains that DRIVE has its own terms.

## 18. Build phases

Build phases and their status are in PROGRESS.md.
