# Getting the data

No dataset files are in this repository. Download them from the official
source, place them under `data/`, and check them with `make check-data`.

## DRIVE

Source: https://drive.grand-challenge.org/ (registration required).

Cite Staal J, Abràmoff MD, Niemeijer M, Viergever MA, van Ginneken B. *Ridge-based vessel segmentation in color images of the retina.* IEEE Transactions on Medical Imaging 23(4):501–509, 2004. doi:10.1109/TMI.2004.825627.

No explicit license is published for DRIVE. Do not redistribute the images.

After downloading, the files must be laid out like this:

```
data/DRIVE/training/images/21_training.tif ... 40_training.tif
data/DRIVE/training/1st_manual/21_manual1.gif ...
data/DRIVE/training/mask/21_training_mask.gif ...
data/DRIVE/test/images/01_test.tif ... 20_test.tif
data/DRIVE/test/mask/01_test_mask.gif ...
```

To keep the data somewhere else, set `DRIVE_DIR` to the directory that
contains `training/` and `test/`:

```bash
export DRIVE_DIR=/path/to/DRIVE
```

## Checking the download

```bash
make check-data
```

This hashes every file under the DRIVE directory and compares it with
`data/CHECKSUMS.sha256`, which is committed. The comparison is always against
the committed file, even when `DRIVE_DIR` points somewhere else. Any missing,
changed, or extra file is listed by name and the command fails. Hidden files
such as `.DS_Store` are ignored. Every run records a hash of the verified
checksums, so results can be traced to the exact data they used.

STARE and CHASE_DB1 are added in phase 7, after their terms of use are
checked.
