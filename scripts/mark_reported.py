"""Mark a finished run as reported, so its numbers can appear in documents.

Run this on the laptop after ``make verify-checkpoints`` and ``make evaluate``
(SPEC section 17). It checks the run against ``configs/reporting.yaml`` and
the rules in ``retinal_vessels.reporting``, lists every failed check, and
changes nothing unless all of them pass. The run id is required, since
reporting the wrong run by default would be easy to miss.
"""

import argparse
import logging
import sys
from contextlib import closing
from pathlib import Path

from retinal_vessels.config import ConfigError, load_reporting_config
from retinal_vessels.db import connect
from retinal_vessels.reporting import ReportError, mark_reported
from retinal_vessels.train import OutputDirs
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("mark_reported")

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "reporting.yaml")
    args = parser.parse_args(argv)
    setup_logging()

    if not args.db.is_file():
        logger.error("database not found at %s", args.db)
        return 1
    try:
        required = load_reporting_config(args.config).reported_runs
    except ConfigError as err:
        logger.error("%s", err)
        return 1
    dirs = OutputDirs(results=args.results_dir, models=args.models_dir)
    with closing(connect(args.db)) as conn:
        try:
            mark_reported(conn, args.run_id, required, dirs)
        except ReportError as err:
            logger.error("%s", err)
            return 1
    logger.info("Run %s is marked reported in %s", args.run_id, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
