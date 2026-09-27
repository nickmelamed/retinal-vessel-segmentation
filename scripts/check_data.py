"""Verify the datasets on disk against the committed ``data/CHECKSUMS.sha256``.

The DRIVE directory defaults to ``data/DRIVE`` in this repository and can be
moved with the ``DRIVE_DIR`` environment variable. The checksum file always
defaults to the committed one, wherever the data lives. ``--init`` writes a
checksum file that does not exist yet. An existing file is never changed.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from retinal_vessels.provenance import check_or_write_checksums, data_hash
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("check_data")

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--checksums", type=Path, default=REPO / "data" / "CHECKSUMS.sha256")
    parser.add_argument(
        "--init", action="store_true", help="write the checksum file if it does not exist"
    )
    args = parser.parse_args(argv)
    setup_logging()

    drive_dir = Path(os.environ.get("DRIVE_DIR", args.data_root / "DRIVE"))
    if not drive_dir.is_dir():
        logger.error("DRIVE directory not found at %s. See docs/DATA.md.", drive_dir)
        return 1

    try:
        written, report = check_or_write_checksums(
            args.checksums, {"DRIVE": drive_dir}, write_missing=args.init
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
