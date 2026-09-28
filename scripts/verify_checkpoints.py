"""Check a run's checkpoint files against the SHA-256 values in ``fold_status``.

Run this on the laptop after downloading a Colab run's database and
checkpoints (SPEC section 16, step 5). Without ``--run-id`` it checks the
most recent run. Every complete fold must have a checkpoint whose hash
matches, and any missing or mismatched file is an error.

The run must also be finished with every fold complete, since a partial
run would otherwise pass on the few folds it has. Pass
``--allow-unfinished`` to check a run that is still training.
"""

import argparse
import json
import logging
import sys
from contextlib import closing
from pathlib import Path

from retinal_vessels.db import connect
from retinal_vessels.reporting import checkpoint_problems
from retinal_vessels.train import OutputDirs
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("verify_checkpoints")

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--run-id", help="defaults to the most recent run")
    parser.add_argument(
        "--allow-unfinished", action="store_true", help="check a run that is still training"
    )
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
        run = conn.execute(
            "SELECT config, finished_at FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        folds = conn.execute(
            "SELECT fold, checkpoint_sha256 FROM fold_status "
            "WHERE run_id = ? AND status = 'complete' ORDER BY fold",
            (run_id,),
        ).fetchall()
    if run is None:
        logger.error("no run %s in %s", run_id, args.db)
        return 1
    n_folds = json.loads(run[0]).get("folds", {}).get("n_folds")
    if not isinstance(n_folds, int):
        logger.error("run %s has no folds.n_folds in its stored config", run_id)
        return 1
    finished = run[1] is not None
    logger.info(
        "run %s: %d of %d folds complete, %s",
        run_id,
        len(folds),
        n_folds,
        "finished" if finished else "not finished",
    )
    if not folds:
        logger.error("run %s has no complete folds in %s", run_id, args.db)
        return 1
    if not args.allow_unfinished and (not finished or len(folds) != n_folds):
        logger.error(
            "run %s is not finished. Finish it, or pass --allow-unfinished to check "
            "the folds it has.",
            run_id,
        )
        return 1

    dirs = OutputDirs(results=REPO / "results", models=args.models_dir)
    with closing(connect(args.db)) as conn:
        problems = checkpoint_problems(conn, run_id, dirs)
    for problem in problems:
        logger.error("%s", problem)
    if problems:
        logger.error("run %s: %d of %d checkpoints failed", run_id, len(problems), len(folds))
        return 1
    logger.info("run %s: %d checkpoints match the database", run_id, len(folds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
