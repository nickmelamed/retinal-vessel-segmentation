---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
  - "tests/**/*.py"
---

# Python standards

- Installable package under `src/retinal_vessels/`. `uv.lock` is the source of
  truth for dependencies, and `requirements.txt` is generated from it for Colab.
- Type hints everywhere, and `mypy --strict` passes on the package.
- NumPy-style docstrings on public functions and classes, stating array shapes
  and dtypes where they matter (see writing-style.md for tone).
- Small, single-purpose modules. Pure functions where possible, I/O at the
  edges. No logic in notebooks.
- All settings come from `configs/*.yaml` through `retinal_vessels.config`.
  Unknown or missing keys are errors. No magic numbers.
- `logging` (configured once in `retinal_vessels.utils`), never `print`, in
  library code. CLIs log run IDs and output paths.
- Seed through `retinal_vessels.utils.set_seed` and record whether
  deterministic GPU ops were on.
- Fail loudly on bad inputs (shape, dtype, empty mask, missing file, config
  mismatch) with a specific message. Never continue silently.
- Ruff for lint and format, line length 100.
