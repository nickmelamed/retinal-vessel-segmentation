"""Decide whether a finished run can back reported numbers, and mark it reported.

Only reported runs reach the tables, figures, and documents (SPEC section 17).
A run qualifies once it finished every fold from a clean, tagged commit on
the reported hardware, was evaluated from that same commit, its checkpoints
match the database, and the leakage audit is empty. Every failed check is
listed, so one attempt shows everything that is wrong.
"""

import json
import sqlite3

from retinal_vessels.config import ReportedRunsConfig
from retinal_vessels.db import fold_statuses, run_query, set_reported
from retinal_vessels.evaluate import EVALUATION_NAME
from retinal_vessels.provenance import sha256_file
from retinal_vessels.train import OutputDirs

LEAKAGE_AUDIT = "04_leakage_audit"


class ReportError(RuntimeError):
    """A run that cannot be marked reported."""


def checkpoint_problems(conn: sqlite3.Connection, run_id: str, dirs: OutputDirs) -> list[str]:
    """Return a problem for each complete fold whose checkpoint is missing or has the wrong hash."""
    rows = conn.execute(
        "SELECT fold, checkpoint_sha256 FROM fold_status "
        "WHERE run_id = ? AND status = 'complete' ORDER BY fold",
        (run_id,),
    ).fetchall()
    problems = []
    for fold, expected in rows:
        path = dirs.checkpoint(run_id, fold)
        if not path.is_file():
            problems.append(f"fold {fold}: checkpoint missing at {path}")
        elif sha256_file(path) != expected:
            problems.append(f"fold {fold}: checksum mismatch for {path}")
    return problems


def _evaluation_problems(run_id: str, commit: str, dirs: OutputDirs) -> list[str]:
    path = dirs.results / run_id / EVALUATION_NAME
    if not path.is_file():
        return [f"{path} is missing. Run make evaluate at the run's commit first."]
    summary = json.loads(path.read_text(encoding="utf-8"))
    code = summary.get("code", {})
    problems = []
    if summary.get("run_id") != run_id:
        problems.append(f"{path} belongs to run {summary.get('run_id')}")
    if code.get("commit") != commit or code.get("dirty") is not False:
        problems.append(
            f"{path} was written by commit {code.get('commit')} (dirty={code.get('dirty')}), "
            f"not by a clean checkout of the run's commit {commit}"
        )
    return problems


def reportability_problems(
    conn: sqlite3.Connection, run_id: str, required: ReportedRunsConfig, dirs: OutputDirs
) -> list[str]:
    """Return every reason ``run_id`` cannot be reported. An empty list means it can."""
    run = conn.execute(
        "SELECT config, finished_at, git_commit, git_dirty, git_tag, gpu_type, compute_platform "
        "FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        return [f"no run {run_id} in the database"]
    config, finished_at, commit, dirty, tag, gpu_type, platform = run

    problems = []
    n_folds = json.loads(config)["folds"]["n_folds"]
    complete = [f for f, s in fold_statuses(conn, run_id).items() if s == "complete"]
    if finished_at is None or len(complete) != n_folds:
        problems.append(f"the run is not finished: {len(complete)} of {n_folds} folds complete")
    if dirty:
        problems.append("the run trained from a dirty tree")
    if tag is None:
        problems.append("the run's commit had no tag when it trained")
    if gpu_type != required.gpu_type or platform != required.compute_platform:
        problems.append(
            f"the run used {gpu_type} on {platform}, and reported runs use "
            f"{required.gpu_type} on {required.compute_platform}"
        )

    tested = conn.execute(
        "SELECT COUNT(DISTINCT image_id) FROM fold_assignments WHERE run_id = ? AND role = 'test'",
        (run_id,),
    ).fetchone()[0]
    rows, evaluated = conn.execute(
        "SELECT COUNT(*), COUNT(brier) FROM per_image_metrics "
        "WHERE run_id = ? AND prediction_mode = 'single'",
        (run_id,),
    ).fetchone()
    if rows != tested or evaluated != rows:
        problems.append(
            f"{evaluated} of {tested} out-of-fold images are evaluated ({rows} rows stored)"
        )

    problems += _evaluation_problems(run_id, commit, dirs)
    problems += checkpoint_problems(conn, run_id, dirs)
    leaks = run_query(conn, LEAKAGE_AUDIT)
    if leaks:
        problems.append(f"the leakage audit returned {len(leaks)} rows, for example {leaks[0]}")
    return problems


def mark_reported(
    conn: sqlite3.Connection, run_id: str, required: ReportedRunsConfig, dirs: OutputDirs
) -> None:
    """Set ``is_reported`` for ``run_id`` once every check passes.

    Raises ``ReportError`` listing each failed check, and changes nothing then.
    """
    problems = reportability_problems(conn, run_id, required, dirs)
    if problems:
        raise ReportError(f"run {run_id} cannot be reported:\n  " + "\n  ".join(problems))
    set_reported(conn, run_id)
