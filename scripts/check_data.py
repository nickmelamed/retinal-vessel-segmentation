"""Verify the datasets on disk against the committed ``data/CHECKSUMS.sha256``.

The DRIVE directory defaults to ``data/DRIVE`` in this repository and can be
moved with the ``DRIVE_DIR`` environment variable. The checksum file always
defaults to the committed one, wherever the data lives. ``--init`` writes a
checksum file that does not exist yet. An existing file is never changed.

Once the checksums pass, every image is loaded and validated, and its
``images`` row (FOV size, within-FOV vessel fraction, abnormality note) is
written to the experiments database. A stored row that no longer matches
the data is an error.
"""

import argparse
import logging
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from retinal_vessels.data import Sample
from retinal_vessels.datasets.drive import LayoutError, drive_dir, load_drive
from retinal_vessels.db import connect, create_schema, image_record, write_images
from retinal_vessels.provenance import check_or_write_checksums, data_hash
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("check_data")

REPO = Path(__file__).resolve().parents[1]


def log_stats(split: str, samples: list[Sample]) -> None:
    """Log image counts, abnormal images, and the pooled within-FOV vessel fraction."""
    abnormal = [s.image_id for s in samples if s.has_abnormality]
    notes = f"abnormality notes on {', '.join(abnormal)}" if abnormal else "no abnormality notes"
    logger.info("DRIVE %s: %d images, %s", split, len(samples), notes)
    counts = [
        (int((s.label & s.fov).sum()), int(s.fov.sum())) for s in samples if s.label is not None
    ]
    if not counts:
        return
    vessel = sum(v for v, _ in counts)
    fov = sum(f for _, f in counts)
    per_image = [v / f for v, f in counts]
    logger.info(
        "DRIVE %s: within-FOV vessel fraction %.4f pooled over %d images "
        "(%d of %d FOV pixels), per image %.4f to %.4f",
        split,
        vessel / fov,
        len(counts),
        vessel,
        fov,
        min(per_image),
        max(per_image),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--checksums", type=Path, default=REPO / "data" / "CHECKSUMS.sha256")
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument(
        "--init", action="store_true", help="write the checksum file if it does not exist"
    )
    args = parser.parse_args(argv)
    setup_logging()

    drive = drive_dir(args.data_root / "DRIVE")
    if not drive.is_dir():
        logger.error("DRIVE directory not found at %s. See docs/DATA.md.", drive)
        return 1

    try:
        written, report = check_or_write_checksums(
            args.checksums, {"DRIVE": drive}, write_missing=args.init
        )
    except FileNotFoundError as err:
        logger.error("%s", err)
        return 1
    if not report.ok:
        for problem in report.problems():
            logger.error(problem)
        logger.error("%d problem(s) against %s", len(report.problems()), args.checksums)
        return 1
    action = "Wrote" if written else "Verified"
    logger.info("%s %d files. Data hash %s", action, len(report.checked), data_hash(report.checked))

    try:
        samples = {split: load_drive(drive, split) for split in ("training", "test")}
        with closing(connect(args.db)) as conn:
            create_schema(conn)
            records = [image_record(s) for split in samples.values() for s in split]
            added = write_images(conn, records)
    except (LayoutError, ValueError) as err:
        logger.error("%s", err)
        return 1
    except sqlite3.IntegrityError as err:
        logger.error("images table rejected a row in %s: %s", args.db, err)
        return 1
    for split, split_samples in samples.items():
        log_stats(split, split_samples)
    logger.info(
        "images table in %s: %d rows added, %d already present",
        args.db,
        added,
        len(records) - added,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
