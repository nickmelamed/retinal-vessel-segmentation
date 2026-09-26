"""Verify the datasets on disk against ``data/CHECKSUMS.sha256``.

The DRIVE directory defaults to ``data/DRIVE`` and can be moved with the
``DRIVE_DIR`` environment variable. If the checksum file does not exist yet,
it is written from the files on disk. An existing file is never changed.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from retinal_vessels.provenance import check_or_write_checksums, data_hash
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("check_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--checksums", type=Path, default=None, help="defaults to <data-root>/CHECKSUMS.sha256"
    )
    args = parser.parse_args(argv)
    setup_logging()

    manifest = args.checksums or args.data_root / "CHECKSUMS.sha256"
    drive_dir = Path(os.environ.get("DRIVE_DIR", args.data_root / "DRIVE"))
    if not drive_dir.is_dir():
        logger.error("DRIVE directory not found at %s. See docs/DATA.md.", drive_dir)
        return 1

    written, report = check_or_write_checksums(manifest, {"DRIVE": drive_dir})
    if not report.ok:
        for problem in report.problems():
            logger.error(problem)
        logger.error("%d problem(s) against %s", len(report.problems()), manifest)
        return 1
    action = "Wrote" if written else "Verified"
    logger.info("%s %d files. Data hash %s", action, len(report.checked), data_hash(report.checked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
