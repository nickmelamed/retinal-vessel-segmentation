"""Check a run's checkpoint files against the SHA-256 values in ``fold_status``.

Run this on the laptop after downloading a Colab run's database and
checkpoints (SPEC section 16, step 5). Without ``--run-id`` it checks the
most recent run. Every complete fold must have a checkpoint whose hash
matches, and any missing or mismatched file is an error.
"""

import argparse
import logging
import sys
from contextlib import closing
from pathlib import Path

from retinal_vessels.db import connect
from retinal_vessels.provenance import sha256_file
from retinal_vessels.train import OutputDirs
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("verify_checkpoints")

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--run-id", help="defaults to the most recent run")
    args = parser.parse_args(argv)
    setup_logging()

    if not args.db.is_file():
        logger.error("database not found at %s", args.db)
        return 1
    with closing(connect(args.db)) as conn:
        run_id = args.run_id
        if run_id is None:
            row = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
            run_id = row[0] if row else None
        if run_id is None:
            logger.error("no runs in %s", args.db)
            return 1
        folds = conn.execute(
            "SELECT fold, checkpoint_sha256 FROM fold_status "
            "WHERE run_id = ? AND status = 'complete' ORDER BY fold",
            (run_id,),
        ).fetchall()
    if not folds:
        logger.error("run %s has no complete folds in %s", run_id, args.db)
        return 1

    dirs = OutputDirs(results=REPO / "results", models=args.models_dir)
    problems = 0
    for fold, expected in folds:
        path = dirs.checkpoint(run_id, fold)
        if not path.is_file():
            logger.error("fold %d: checkpoint missing at %s", fold, path)
            problems += 1
        elif sha256_file(path) != expected:
            logger.error("fold %d: checksum mismatch for %s", fold, path)
            problems += 1
    if problems:
        logger.error("run %s: %d of %d checkpoints failed", run_id, problems, len(folds))
        return 1
    logger.info("run %s: %d checkpoints match the database", run_id, len(folds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
