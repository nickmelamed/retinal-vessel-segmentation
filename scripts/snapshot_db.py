"""Export the reported runs to ``results/release/experiments_<tag>.db``, metrics only.

Run this while preparing a release (``/release``). The snapshot is
committed, so the R report and the results site can be rebuilt from it
without the data or a GPU (SPEC section 12). See
``retinal_vessels.reporting.write_snapshot`` for what it holds.
"""

import argparse
import logging
import sys
from pathlib import Path

from retinal_vessels.reporting import ReportError, write_snapshot
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("snapshot_db")

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="the release tag, like v0.1.0")
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results" / "release")
    args = parser.parse_args(argv)
    setup_logging()

    if not args.db.is_file():
        logger.error("database not found at %s", args.db)
        return 1
    try:
        write_snapshot(args.db, args.results_dir, args.out_dir, args.tag)
    except ReportError as err:
        logger.error("%s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
