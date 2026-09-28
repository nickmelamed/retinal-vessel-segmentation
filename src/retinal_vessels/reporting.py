"""Decide whether a finished run can back reported numbers, and mark it reported.

Only reported runs reach the tables, figures, and documents (SPEC section 17).
A run qualifies once it finished every fold from a clean, tagged commit on
the reported hardware, was evaluated from that same commit, its checkpoints
match the database, and the leakage audit is empty. Every failed check is
listed, so one attempt shows everything that is wrong.
"""

import json
import logging
import os
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from retinal_vessels.config import ReportedRunsConfig
from retinal_vessels.db import create_schema, fold_statuses, run_query, set_reported
from retinal_vessels.evaluate import EVALUATION_NAME
from retinal_vessels.provenance import data_hash, read_checksums, sha256_file
from retinal_vessels.train import OutputDirs

logger = logging.getLogger(__name__)

LEAKAGE_AUDIT = "04_leakage_audit"
DATASET = "drive"
# The dataset's top-level directory in data/CHECKSUMS.sha256.
MANIFEST_DATASET = "DRIVE"


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
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        return [f"{path} cannot be read: {err}"]
    if not isinstance(summary, dict) or not isinstance(summary.get("code"), dict):
        return [f"{path} is not an evaluation summary"]
    code = summary["code"]
    problems = []
    if summary.get("run_id") != run_id:
        problems.append(f"{path} belongs to run {summary.get('run_id')}")
    if code.get("commit") != commit or code.get("dirty") is not False:
        problems.append(
            f"{path} was written by commit {code.get('commit')} (dirty={code.get('dirty')}), "
            f"not by a clean checkout of the run's commit {commit}"
        )
    return problems


def _data_problems(data_hash_: str, checksums: Path) -> list[str]:
    try:
        entries = read_checksums(checksums)
    except (OSError, ValueError) as err:
        return [f"cannot read the committed checksums at {checksums}: {err}"]
    # A run's hash covers only the DRIVE entries it was verified against, and
    # the manifest also holds the external datasets once they are added.
    expected = data_hash(
        {rel: d for rel, d in entries.items() if rel.split("/", 1)[0] == MANIFEST_DATASET}
    )
    if data_hash_ != expected:
        return [f"the run's data hash does not match the checksums in {checksums}"]
    return []


def reportability_problems(
    conn: sqlite3.Connection,
    run_id: str,
    required: ReportedRunsConfig,
    dirs: OutputDirs,
    checksums: Path,
) -> list[str]:
    """Return every reason ``run_id`` cannot be reported. An empty list means it can.

    ``checksums`` is the committed checksum manifest, which the run's data
    hash must match.
    """
    run = conn.execute(
        "SELECT config, finished_at, git_commit, git_dirty, git_tag, gpu_type, compute_platform, "
        "deterministic_ops, data_hash FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        return [f"no run {run_id} in the database"]
    config, finished_at, commit, dirty, tag, gpu_type, platform, deterministic, data = run

    problems = []
    stored = json.loads(config)
    n_folds = stored["folds"]["n_folds"]
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
    # Determinism is a config choice, so the record must match what was asked.
    asked = stored["training"]["deterministic_ops"]
    if bool(deterministic) != asked:
        problems.append(
            f"the run recorded deterministic_ops={bool(deterministic)}, "
            f"but its config asked for {asked}"
        )
    problems += _data_problems(data, checksums)

    # Compare sets of images, since matching counts could hide a row for an
    # image the run never tested next to a missing one.
    tested = set(
        conn.execute(
            "SELECT DISTINCT dataset, image_id FROM fold_assignments "
            "WHERE run_id = ? AND role = 'test'",
            (run_id,),
        ).fetchall()
    )
    stored_rows = {
        (dataset, image_id): brier is not None
        for dataset, image_id, brier in conn.execute(
            "SELECT dataset, image_id, brier FROM per_image_metrics "
            "WHERE run_id = ? AND prediction_mode = 'single'",
            (run_id,),
        )
    }
    evaluated = {key for key, done in stored_rows.items() if done} & tested
    # SPEC section 5 gives every labeled image exactly one out-of-fold
    # prediction. The leakage audit catches an image tested twice, and this
    # catches one never tested.
    labeled = set(
        conn.execute(
            "SELECT dataset, image_id FROM images WHERE dataset = ? AND has_labels = 1",
            (DATASET,),
        ).fetchall()
    )
    never = sorted(image_id for (_, image_id) in labeled - tested)
    if never:
        problems.append(f"the run never tested labeled images {never}")
    if evaluated != tested:
        problems.append(f"{len(evaluated)} of {len(tested)} out-of-fold images are evaluated")
    untested = sorted(image_id for (_, image_id) in set(stored_rows) - tested)
    if untested:
        problems.append(f"metric rows exist for images the run never tested: {untested}")

    problems += _evaluation_problems(run_id, commit, dirs)
    problems += checkpoint_problems(conn, run_id, dirs)
    leaks = run_query(conn, LEAKAGE_AUDIT)
    if leaks:
        problems.append(f"the leakage audit returned {len(leaks)} rows, for example {leaks[0]}")
    return problems


def mark_reported(
    conn: sqlite3.Connection,
    run_id: str,
    required: ReportedRunsConfig,
    dirs: OutputDirs,
    checksums: Path,
) -> None:
    """Set ``is_reported`` for ``run_id`` once every check passes.

    Raises ``ReportError`` listing each failed check, and changes nothing then.
    """
    problems = reportability_problems(conn, run_id, required, dirs, checksums)
    if problems:
        raise ReportError(f"run {run_id} cannot be reported:\n  " + "\n  ".join(problems))
    set_reported(conn, run_id)


# Tables copied whole into a snapshot. They describe the data, not a run.
SNAPSHOT_WHOLE = ("images",)
# Filled by create_schema in the new database.
SNAPSHOT_SKIP = ("schema_version",)
# A release tag also names the snapshot file, so it must be a safe file name.
RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$")


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA main.table_info({table})")]


def _copy_reported(conn: sqlite3.Connection) -> dict[str, int]:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM main.sqlite_master WHERE type = 'table' ORDER BY name"
        )
        if row[0] not in SNAPSHOT_SKIP
    ]
    copied = {}
    for table in tables:
        columns = _columns(conn, table)
        names = ", ".join(columns)
        if table in SNAPSHOT_WHOLE:
            where = ""
        elif "run_id" in columns:
            where = " WHERE run_id IN (SELECT run_id FROM src.runs WHERE is_reported = 1)"
        else:
            # A new table must be placed in one group or the other on purpose.
            raise ReportError(f"the snapshot has no rule for table {table}")
        cursor = conn.execute(
            f"INSERT INTO main.{table} ({names}) SELECT {names} FROM src.{table}{where}"
        )
        copied[table] = cursor.rowcount
    return copied


def write_snapshot(source: Path, results_dir: Path, out_dir: Path, tag: str) -> Path:
    """Export the reported runs of ``source`` to ``out_dir/experiments_<tag>.db``.

    The snapshot has the full schema, with every row of ``images`` and the
    rows of reported runs from every other table, so the R report and the
    results site can be rebuilt without the data or a GPU (SPEC section 12).
    It holds metrics and metadata only. Each run's ``evaluation.json`` is
    copied to ``out_dir/<tag>/<run_id>/``, since the reliability table and
    fold edges live only there. An existing snapshot is never replaced.
    Raises ``ReportError`` if there is no reported run or a check fails.
    """
    if not RELEASE_TAG.match(tag):
        raise ReportError(f"{tag!r} is not a release tag like v0.1.0 or v0.1.0-rc.1")
    path = out_dir / f"experiments_{tag}.db"
    if path.exists() or (out_dir / tag).exists():
        raise ReportError(f"a snapshot for {tag} already exists in {out_dir}")
    with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as src:
        reported = [r[0] for r in src.execute("SELECT run_id FROM runs WHERE is_reported = 1")]
    if not reported:
        raise ReportError(f"{source} has no reported runs. Run make mark-reported first.")
    evaluations = {run_id: results_dir / run_id / EVALUATION_NAME for run_id in reported}
    missing = [str(p) for p in evaluations.values() if not p.is_file()]
    if missing:
        raise ReportError(f"reported runs are missing their evaluation summaries: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        # URI filenames let the source be attached read-only, so writing the
        # snapshot can never change the database it copies from.
        with closing(sqlite3.connect(partial.resolve().as_uri(), uri=True)) as conn:
            create_schema(conn)
            # Rows are copied table by table, so references are checked once at the end.
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("ATTACH DATABASE ? AS src", (f"{source.resolve().as_uri()}?mode=ro",))
            with conn:
                copied = _copy_reported(conn)
            conn.execute("DETACH DATABASE src")
            dangling = conn.execute("PRAGMA foreign_key_check").fetchall()
            if dangling:
                raise ReportError(f"the snapshot has rows with dangling references: {dangling}")
            leaks = run_query(conn, LEAKAGE_AUDIT)
            if leaks:
                raise ReportError(f"the leakage audit returned {len(leaks)} rows in the snapshot")
            conn.execute("VACUUM")
        for run_id, evaluation in evaluations.items():
            target = out_dir / tag / run_id / EVALUATION_NAME
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(evaluation, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        shutil.rmtree(out_dir / tag, ignore_errors=True)
        raise
    os.replace(partial, path)
    logger.info("Wrote snapshot %s with runs %s, rows %s", path, reported, copied)
    return path
