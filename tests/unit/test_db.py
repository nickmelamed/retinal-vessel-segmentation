import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from retinal_vessels.db import SCHEMA_VERSION, connect, create_schema, schema_version

TABLES = {
    "schema_version",
    "images",
    "frozen_models",
    "runs",
    "fold_assignments",
    "fold_status",
    "thresholds",
    "per_image_metrics",
    "risk_coverage",
    "training_history",
}

RUN: dict[str, Any] = {
    "run_id": "r1",
    "variant": "baseline",
    "config": "{}",
    "config_hash": "c",
    "seed": 0,
    "git_commit": "abc",
    "git_dirty": 0,
    "data_hash": "d",
    "python_version": "3.13",
    "tensorflow_version": "2.21",
    "device": "cpu",
    "gpu_type": "cpu",
    "compute_platform": "local",
    "deterministic_ops": 0,
    "started_at": "2026-09-26T00:00:00Z",
}

IMAGE: dict[str, Any] = {
    "dataset": "drive",
    "image_id": "21",
    "split": "training",
    "has_abnormality": 0,
    "fov_pixels": 100,
    "fov_source": "official",
    "vessel_fraction_in_fov": 0.1,
    "has_labels": 1,
}


def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(tmp_path / "experiments.db")
    create_schema(c)
    yield c
    c.close()


def tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r[0] for r in rows}


def test_creates_every_table_and_records_the_version(conn: sqlite3.Connection) -> None:
    assert tables(conn) == TABLES
    assert schema_version(conn) == SCHEMA_VERSION


def test_create_schema_is_idempotent(conn: sqlite3.Connection) -> None:
    insert(conn, "runs", RUN)
    create_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone() == (1,)
    assert conn.execute("SELECT run_id FROM runs").fetchall() == [("r1",)]


def test_empty_database_has_no_version(tmp_path: Path) -> None:
    with closing(connect(tmp_path / "new.db")) as c:
        assert schema_version(c) is None


def test_rejects_a_database_at_another_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (99, '2026-09-26T00:00:00Z')"
    )
    with pytest.raises(RuntimeError, match="version 99"):
        create_schema(conn)


def test_rejects_a_schema_file_with_the_wrong_version(tmp_path: Path) -> None:
    schema = tmp_path / "schema.sql"
    schema.write_text(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT);"
        "INSERT INTO schema_version VALUES (7, 'x');"
    )
    with (
        closing(connect(tmp_path / "e.db")) as c,
        pytest.raises(RuntimeError, match="sets version 7"),
    ):
        create_schema(c, schema)


def test_missing_schema_file_fails_loudly(tmp_path: Path) -> None:
    with closing(connect(tmp_path / "e.db")) as c, pytest.raises(FileNotFoundError):
        create_schema(c, tmp_path / "nope.sql")


def test_foreign_keys_are_enforced(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        insert(
            conn,
            "thresholds",
            {"run_id": "nope", "fold": 1, "threshold": 0.5, "selection_rule": "max val dice"},
        )


FINISHED = {"finished_at": "2026-09-26T01:00:00Z"}


# Each row breaks exactly one rule, and ``match`` names the constraint that
# must reject it, so a row cannot pass because some other check fired first.
@pytest.mark.parametrize(
    ("table", "row", "match"),
    [
        ("images", {**IMAGE, "dataset": "hrf"}, "dataset IN"),
        ("images", {**IMAGE, "fov_source": "guessed"}, "fov_source IN"),
        ("images", {**IMAGE, "has_labels": 0}, "has_labels = 1"),
        ("images", {**IMAGE, "has_abnormality": 1}, "has_abnormality = 1"),
        (
            "images",
            {**IMAGE, "has_abnormality": 1, "abnormality_note": ""},
            "has_abnormality = 1",
        ),
        (
            "images",
            {**IMAGE, "abnormality_note": "background diabetic retinopathy"},
            "has_abnormality = 1",
        ),
        (
            "runs",
            {**RUN, **FINISHED, "git_tag": "v0.1.0", "git_dirty": 1, "is_reported": 1},
            "git_dirty = 0",
        ),
        ("runs", {**RUN, **FINISHED, "is_reported": 1}, "git_tag IS NOT NULL"),
        ("runs", {**RUN, "git_tag": "v0.1.0", "is_reported": 1}, "finished_at IS NOT NULL"),
        ("runs", {**RUN, "seed": "zero"}, "runs.seed"),
    ],
)
def test_check_constraints_reject_bad_rows(
    conn: sqlite3.Connection, table: str, row: dict[str, Any], match: str
) -> None:
    with pytest.raises(sqlite3.IntegrityError, match=match):
        insert(conn, table, row)


def test_fold_rows_are_checked(conn: sqlite3.Connection) -> None:
    insert(conn, "runs", RUN)
    insert(conn, "images", IMAGE)
    base = {"run_id": "r1", "fold": 1, "dataset": "drive", "image_id": "21"}
    insert(conn, "fold_assignments", {**base, "role": "test"})
    with pytest.raises(sqlite3.IntegrityError):
        insert(conn, "fold_assignments", {**base, "role": "holdout"})
    with pytest.raises(sqlite3.IntegrityError):
        insert(conn, "fold_assignments", {**base, "role": "test"})


def test_a_fold_cannot_be_complete_without_its_checkpoint(conn: sqlite3.Connection) -> None:
    insert(conn, "runs", RUN)
    row = {"run_id": "r1", "fold": 1, "started_at": "2026-09-26T00:00:00Z"}
    insert(conn, "fold_status", {**row, "status": "running"})
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE fold_status SET status = 'complete' WHERE run_id = 'r1'")
    conn.execute(
        "UPDATE fold_status SET status = 'complete', checkpoint_sha256 = 'x',"
        " completed_at = '2026-09-26T01:00:00Z' WHERE run_id = 'r1'"
    )


def test_metrics_must_lie_in_range(conn: sqlite3.Connection) -> None:
    insert(conn, "runs", RUN)
    insert(conn, "images", IMAGE)
    row = {
        "run_id": "r1",
        "dataset": "drive",
        "image_id": "21",
        "fold": 1,
        "prediction_mode": "single",
    }
    with pytest.raises(sqlite3.IntegrityError):
        insert(conn, "per_image_metrics", {**row, "dice": 1.5})
    insert(conn, "per_image_metrics", {**row, "dice": 0.8})


def test_connect_creates_the_results_directory(tmp_path: Path) -> None:
    path = tmp_path / "results" / "nested" / "experiments.db"
    with closing(connect(path)):
        pass
    assert path.is_file()


def test_a_finished_clean_tagged_run_can_be_reported(conn: sqlite3.Connection) -> None:
    finished = {"git_tag": "v0.1.0", "finished_at": "2026-09-26T01:00:00Z"}
    insert(conn, "runs", {**RUN, **finished, "is_reported": 1})
    assert conn.execute("SELECT is_reported FROM runs").fetchall() == [(1,)]


def test_an_unfinished_run_cannot_be_reported(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="finished_at IS NOT NULL"):
        insert(conn, "runs", {**RUN, "git_tag": "v0.1.0", "is_reported": 1})


def test_external_runs_record_model_and_threshold_together(conn: sqlite3.Connection) -> None:
    insert(conn, "runs", RUN)
    frozen = {
        "model_id": "m1",
        "run_id": "r1",
        "checkpoint_sha256": "c",
        "threshold": 0.5,
        "n_epochs": 30,
        "git_commit": "abc",
        "frozen_at": "2026-09-26T02:00:00Z",
    }
    insert(conn, "frozen_models", frozen)
    pairing = "frozen_model_id IS NULL"
    with pytest.raises(sqlite3.IntegrityError, match=pairing):
        insert(conn, "runs", {**RUN, "run_id": "e1", "frozen_model_id": "m1"})
    with pytest.raises(sqlite3.IntegrityError, match=pairing):
        insert(conn, "runs", {**RUN, "run_id": "e2", "applied_threshold": 0.5})
    insert(conn, "runs", {**RUN, "run_id": "e3", "frozen_model_id": "m1", "applied_threshold": 0.5})
    with pytest.raises(sqlite3.IntegrityError, match="applied_threshold BETWEEN"):
        insert(
            conn, "runs", {**RUN, "run_id": "e4", "frozen_model_id": "m1", "applied_threshold": 1.5}
        )


def test_folds_count_from_one_and_zero_is_the_frozen_model(conn: sqlite3.Connection) -> None:
    insert(conn, "runs", RUN)
    insert(conn, "images", IMAGE)
    for table, row in [
        ("fold_assignments", {"dataset": "drive", "image_id": "21", "role": "test"}),
        ("fold_status", {"status": "running", "started_at": "2026-09-26T00:00:00Z"}),
        ("thresholds", {"threshold": 0.5, "selection_rule": "max val dice"}),
        ("per_image_metrics", {"dataset": "drive", "image_id": "21", "prediction_mode": "single"}),
    ]:
        with pytest.raises(sqlite3.IntegrityError, match="fold >= 1"):
            insert(conn, table, {"run_id": "r1", "fold": 0, **row})
    insert(conn, "training_history", {"run_id": "r1", "fold": 0, "epoch": 0, "train_loss": 0.3})


@pytest.mark.parametrize(
    "started_at",
    [
        "2026-09-26T12:00:00+00:00",
        "2026-09-26T12:00:00.123Z",
        "2026-09-26 12:00:00",
        "2026-13-01T00:00:00Z",
        "t0",
    ],
)
def test_timestamps_must_be_whole_second_utc(conn: sqlite3.Connection, started_at: str) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="started_at"):
        insert(conn, "runs", {**RUN, "started_at": started_at})


def test_an_abnormal_image_is_stored_with_its_note(conn: sqlite3.Connection) -> None:
    note = "background diabetic retinopathy"
    insert(
        conn, "images", {**IMAGE, "image_id": "32", "has_abnormality": 1, "abnormality_note": note}
    )
    rows = conn.execute("SELECT image_id, abnormality_note FROM images").fetchall()
    assert rows == [("32", note)]
