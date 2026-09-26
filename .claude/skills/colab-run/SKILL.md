---
name: colab-run
description: Steps for a reported GPU run on Google Colab. Use when preparing, running, or recovering a Colab training or evaluation session.
---

Reported GPU runs happen on Colab (free tier, T4) through the VS Code Colab
extension. The session is disposable. `notebooks/colab_runner.ipynb` holds
only these steps and no project logic.

1. Clone the repo and check out the tagged commit being run. Never edit code
   in the Colab session. Edit locally, commit, push, and pull.
2. Install from `requirements.txt` and print versions, including `nvidia-smi`.
3. Upload the data to the Colab machine, then run `make check-data`. The
   checksums must match before any training.
4. Run the `make` targets.
5. Before the session ends, download `results/experiments.db`, the
   `results/<run_id>/` manifests, and the checkpoints. Verify the checkpoint
   SHA-256 values against the database.

Cross-validation is resumable. Rerunning the same command skips folds marked
complete in `fold_status` and retrains incomplete ones, so download after each
fold and a disconnect costs at most one fold.

All reported runs must use the same GPU type. If Colab assigns something
else, record it and keep it out of reported comparisons. See SPEC section 16
for fallbacks.
