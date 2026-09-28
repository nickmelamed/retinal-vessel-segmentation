---
name: colab-run
description: Steps for a reported GPU run on Google Colab. Use when preparing, running, or recovering a Colab training or evaluation session.
---

Reported GPU runs happen on Colab (paid plan, always a T4, D-013) through the
VS Code Colab extension. The session is disposable, and
`notebooks/colab_runner.ipynb` holds only these steps and no project logic.
The owner runs the cells. Guide them one step at a time and ask for each
cell's output before moving on. Name cells by their heading or contents,
never by an index. Once the run is on the laptop, `/report-run` takes over.

## Before the session

1. The commit to run must be tagged and pushed, with the owner's approval
   (a release candidate such as `v0.1.0-rc.1`, D-022). Any fix after that
   means a new commit, a new rc tag, and a fresh run. A run is never patched.
2. The owner opens the notebook in VS Code, where it opens in the Jupyter
   editor, and picks Select Kernel, Colab, New Colab Server, with a T4.
3. In the first code cell, set `REF` to the tag. That cell only holds
   settings, and the owner's local edits to it are never committed.

## In the session

4. Run the checkout cell. It must show the tagged commit, and
   `git status --short` must print nothing.
5. Run the install cells. The `PATH` cell also sets `MPLBACKEND=Agg`, since
   Colab's inline plotting backend makes TensorFlow fail to import.
6. Run the environment check. Before any training, confirm all of these:
   - TensorFlow imports and lists one GPU
   - `nvidia-smi` shows a Tesla T4
   - `COLAB_RELEASE_TAG` is set, not `not set`

   If anything is off, stop and tell the owner. Nothing from that session is
   reported.
7. Upload the data. In VS Code's Explorer, right-click `data/DRIVE` and
   choose Upload to Colab. It lands at `/content/DRIVE`. Then run the data
   cell. `make check-data` must pass against the committed checksums.
8. Run the training cell. In the first lines, check for
   `Started run <run_id>` and for no unimplemented-determinism error. If one
   appears, stop the cell, and the owner decides what happens next. The
   `meta_optimizer.cc:967] layout failed` lines are harmless (D-022).
9. Each fold ends with `Fold N complete`. The notebook cannot run another
   cell while training runs, so save mid-run from the Colab terminal
   (command palette, Colab: Open Terminal):
   `cd /content/repo && zip -qr /content/results.zip results`, and zip any
   new checkpoint on its own (next step). A restarted training cell resumes
   the same run, skipping complete folds.

## After the run

10. When the cell logs `Run <run_id> finished`, ask for its last lines and
    check that every fold completed. In a scratch cell, zip the outputs in
    pieces. The extension cannot download a file over about 512 MB once it
    is encoded, and each checkpoint is about 90 MB.

    ```
    %cd /content/repo
    !du -sh results models
    !ls -lh models/*/
    !zip -qr /content/results.zip results
    !for f in models/*/fold_*.keras; do n=$(basename "$f" .keras); zip -qj "/content/model_${n}.zip" "$f"; done
    !ls -lh /content/*.zip
    ```

11. Check that every zip is well under the limit, then have the owner
    download each one to `~/Downloads`. The Colab Contents view opens from
    the command palette (View: Open View..., then Colab, Contents). Refresh
    it, then right-click a file and choose Download....
12. Keep the session connected until `/report-run` confirms the files are
    intact and match the database.

All reported runs use the same GPU type. If Colab assigns something else,
record it and keep the run out of reported comparisons. See SPEC section 16
for fallbacks.
