import importlib.util
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest

from retinal_vessels.db import SCHEMA_VERSION, schema_version, set_reported
from retinal_vessels.reporting import ReportError, write_snapshot
from tests.fixtures.database import RUN, insert
from tests.fixtures.evaluated_run import REPO, RUN_ID, TAG, Finished

SCRIPT = REPO / "scripts" / "snapshot_db.py"
RUN_TABLES = (
    "runs",
    "fold_assignments",
    "fold_status",
    "thresholds",
    "per_image_metrics",
    "training_history",
)
MAX_BYTES = 1_000_000


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("snapshot_db", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


@pytest.fixture
def reported(reportable: Finished) -> Finished:
    """The reportable run, marked reported, next to an unreported run with its own rows."""
    set_reported(reportable.conn, RUN_ID)
    insert(reportable.conn, "runs", RUN)
    insert(
        reportable.conn,
        "fold_assignments",
        {"run_id": RUN["run_id"], "fold": 1, "dataset": "drive", "image_id": "21", "role": "test"},
    )
    reportable.conn.commit()
    return reportable


def snapshot(finished: Finished, tmp_path: Path, tag: str = TAG) -> Path:
    return write_snapshot(finished.db, finished.dirs.results, tmp_path / "release", tag)


def rows(
    conn: sqlite3.Connection, table: str, run_id: str | None = None
) -> list[tuple[object, ...]]:
    columns = ", ".join(r[1] for r in conn.execute(f"PRAGMA table_info({table})"))
    where = "" if run_id is None else f" WHERE run_id = '{run_id}'"
    return conn.execute(f"SELECT {columns} FROM {table}{where} ORDER BY 1, 2, 3").fetchall()


def test_copies_every_row_of_the_reported_run(reported: Finished, tmp_path: Path) -> None:
    path = snapshot(reported, tmp_path)
    assert path == tmp_path / "release" / f"experiments_{TAG}.db"
    with closing(sqlite3.connect(path)) as snap:
        for table in RUN_TABLES:
            expected = rows(reported.conn, table, RUN_ID)
            assert expected, table
            assert rows(snap, table) == expected, table
        assert rows(snap, "images") == rows(reported.conn, "images")


def test_leaves_out_unreported_runs(reported: Finished, tmp_path: Path) -> None:
    with closing(sqlite3.connect(snapshot(reported, tmp_path))) as snap:
        assert snap.execute("SELECT run_id FROM runs").fetchall() == [(RUN_ID,)]
        assert snap.execute(
            "SELECT COUNT(*) FROM fold_assignments WHERE run_id = ?", (RUN["run_id"],)
        ).fetchone() == (0,)


def test_keeps_the_schema_version(reported: Finished, tmp_path: Path) -> None:
    with closing(sqlite3.connect(snapshot(reported, tmp_path))) as snap:
        assert schema_version(snap) == SCHEMA_VERSION


def test_copies_the_evaluation_summary(reported: Finished, tmp_path: Path) -> None:
    snapshot(reported, tmp_path)
    source = reported.dirs.results / RUN_ID / "evaluation.json"
    copy = tmp_path / "release" / TAG / RUN_ID / "evaluation.json"
    assert copy.read_bytes() == source.read_bytes()


def test_stays_under_the_large_file_limit(reported: Finished, tmp_path: Path) -> None:
    # The pre-commit hook rejects files over 1 MB, and the snapshot is committed.
    assert snapshot(reported, tmp_path).stat().st_size < MAX_BYTES


def test_refuses_without_a_reported_run(reportable: Finished, tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="no reported runs"):
        snapshot(reportable, tmp_path)
    assert not (tmp_path / "release").exists()


def test_never_replaces_a_snapshot(reported: Finished, tmp_path: Path) -> None:
    path = snapshot(reported, tmp_path)
    before = path.read_bytes()
    with pytest.raises(ReportError, match="already exists"):
        snapshot(reported, tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("tag", ["0.1.0", "v0.1", "v0.1.0/../x", "latest"])
def test_refuses_a_tag_that_is_not_a_release(reported: Finished, tmp_path: Path, tag: str) -> None:
    with pytest.raises(ReportError, match="not a release tag"):
        snapshot(reported, tmp_path, tag)


def test_refuses_a_reported_run_without_its_summary(reported: Finished, tmp_path: Path) -> None:
    (reported.dirs.results / RUN_ID / "evaluation.json").unlink()
    with pytest.raises(ReportError, match="missing their evaluation summaries"):
        snapshot(reported, tmp_path)


def test_refuses_a_reference_to_an_unreported_run(reported: Finished, tmp_path: Path) -> None:
    # A frozen model trained by an unreported run would be left out, and the
    # reported run pointing at it would dangle.
    insert(
        reported.conn,
        "frozen_models",
        {
            "model_id": "m1",
            "run_id": RUN["run_id"],
            "checkpoint_sha256": "ab" * 32,
            "threshold": 0.5,
            "n_epochs": 10,
            "git_commit": "abc",
            "frozen_at": RUN["started_at"],
        },
    )
    reported.conn.execute(
        "UPDATE runs SET frozen_model_id = 'm1', applied_threshold = 0.5 WHERE run_id = ?",
        (RUN_ID,),
    )
    reported.conn.commit()
    with pytest.raises(ReportError, match="dangling references"):
        snapshot(reported, tmp_path)
    release = tmp_path / "release"
    assert list(release.iterdir()) == []


def test_cli_writes_the_snapshot(reported: Finished, tmp_path: Path) -> None:
    out = tmp_path / "release"
    args = ["--tag", TAG, "--db", str(reported.db), "--results-dir", str(reported.dirs.results)]
    assert load_script().main([*args, "--out-dir", str(out)]) == 0
    assert (out / f"experiments_{TAG}.db").is_file()
    assert load_script().main([*args, "--out-dir", str(out)]) == 1


def test_cli_needs_a_database(tmp_path: Path) -> None:
    assert load_script().main(["--tag", TAG, "--db", str(tmp_path / "absent.db")]) == 1
