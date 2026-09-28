"""Write the Markdown tables in ``results/tables`` from the reported runs.

Documents copy their numbers from these files, so only reported runs are
used. The smoke test and ``/scratch-train`` pass ``--allow-unreported``,
which marks every file as a development run and cannot write to
``results/tables``.
"""

import argparse
import logging
import sys
from contextlib import closing
from pathlib import Path

from retinal_vessels.config import ConfigError, load_reporting_config
from retinal_vessels.db import connect
from retinal_vessels.tables import TableError, reported_run_ids, write_tables
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("make_tables")

REPO = Path(__file__).resolve().parents[1]
REPORTED_TABLES = REPO / "results" / "tables"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--out-dir", type=Path, default=REPORTED_TABLES)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "reporting.yaml")
    parser.add_argument(
        "--allow-unreported",
        metavar="RUN_ID",
        help="tabulate this unreported run as a development run, never into results/tables",
    )
    args = parser.parse_args(argv)
    setup_logging()

    # check_numbers.py reads every file under results/tables, so a development
    # run may not write anywhere inside it.
    out = args.out_dir.resolve()
    if args.allow_unreported and REPORTED_TABLES.resolve() in (out, *out.parents):
        logger.error("unreported runs never go into %s. Pass another --out-dir.", REPORTED_TABLES)
        return 1
    if not args.db.is_file():
        logger.error("database not found at %s", args.db)
        return 1
    try:
        config = load_reporting_config(args.config).tables
    except ConfigError as err:
        logger.error("%s", err)
        return 1
    with closing(connect(args.db)) as conn:
        run_ids = [args.allow_unreported] if args.allow_unreported else reported_run_ids(conn)
        try:
            write_tables(
                conn,
                run_ids,
                args.results_dir,
                args.out_dir,
                config,
                args.db,
                development=bool(args.allow_unreported),
            )
        except TableError as err:
            logger.error("%s", err)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
