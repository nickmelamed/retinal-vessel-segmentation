# Retinal vessel segmentation on DRIVE: a self-directed project

A learning project on public data. A small U-Net segments blood vessels in the
DRIVE color fundus photographs and is evaluated with image-level
cross-validation. Later phases add external validation on other retinal
datasets, per-pixel uncertainty maps, a SQL audit trail, statistics in R, and
an LSTM autoencoder that detects simulated narrowings in vessel width profiles.

Not for clinical use.

## Status

The project is being built in phases, listed in PROGRESS.md. The package
skeleton, data checks, database schema, run provenance, and CI are in place.
There are no model results yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/), which installs Python `3.13` from
`.python-version`.

```bash
make setup        # install the locked environment and git hooks
make check-data   # verify the downloaded data against its checksums
make ci           # lint, tests, smoke run, and lockfile check
make help         # list every target
```

## Data

No data is included in this repository. docs/DATA.md explains how to get DRIVE
from its official source (https://drive.grand-challenge.org/) and where to put
it. DRIVE has its own terms, separate from this code's license, and its images
are not redistributed here.

Staal J, Abràmoff MD, Niemeijer M, Viergever MA, van Ginneken B. *Ridge-based vessel segmentation in color images of the retina.* IEEE Transactions on Medical Imaging 23(4):501–509, 2004. doi:10.1109/TMI.2004.825627. <!-- numbers: ok -->

## License

The code is under the MIT license (see LICENSE). The license does not cover
any dataset.
