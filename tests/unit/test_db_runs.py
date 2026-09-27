import json
import sqlite3
from dataclasses import replace
from typing import Any

import pytest

from retinal_vessels.data import make_folds
from retinal_vessels.db import (
    EpochRecord,
    FoldRecord,
    ImageMetrics,
    complete_fold,
    find_resumable_run,
    finish_run,
    fold_statuses,
    insert_run,
    start_fold,
    stored_fold_rows,
    write_fold_assignments,
)
from retinal_vessels.metrics import BinaryMetrics
from retinal_vessels.provenance import RunManifest
from tests.fixtures.database import insert

T0 = "2026-09-27T10:00:00Z"
T1 = "2026-09-27T11:00:00Z"
ENV = {
    "python_version": "3.13.1",
    "tensorflow_version": "2.21.0",
    "cuda_version": None,
    "device": "cpu",
    "gpu_type": "cpu",
    "compute_platform": "local",
}
MANIFEST = RunManifest(
    run_id="run-a",
    variant="baseline",
    config={"b": 1, "a": [1, 2]},
    config_hash="hash",
    seed=7,
    git_commit="c0ffee",
    git_dirty=False,
    git_tag=None,
    data_hash="data",
    deterministic_ops=True,
    started_at=T0,
    environment=ENV,
)
METRICS = BinaryMetrics(
    dice=0.8,
    sensitivity=0.75,
    specificity=0.97,
    precision=None,
    accuracy=0.95,
    predicted_vessel_fraction=0.12,
)


def image_row(image_id: str) -> dict[str, Any]:
    return {
        "dataset": "drive",
        "image_id": image_id,
        "split": "training",
        "has_abnormality": 0,
        "fov_pixels": 10,
        "fov_source": "official",
        "has_labels": 1,
    }


@pytest.fixture
def db(conn: sqlite3.Connection) -> sqlite3.Connection:
    for image_id in ("21", "22"):
        insert(conn, "images", image_row(image_id))
    insert_run(conn, MANIFEST)
    return conn


def record(fold: int = 1, **overrides: Any) -> FoldRecord:
    fields: dict[str, Any] = {
        "run_id": "run-a",
        "fold": fold,
        "checkpoint_sha256": "ab" * 32,
        "threshold": 0.42,
        "selection_rule": "rule",
        "val_dice": 0.81,
        "history": [EpochRecord(1, 0.9, 0.7), EpochRecord(2, 0.6, 0.81)],
        "test_metrics": [
            ImageMetrics("drive", "21", METRICS),
            ImageMetrics("drive", "22", METRICS),
        ],
    }
    return FoldRecord(**{**fields, **overrides})


def counts(conn: sqlite3.Connection, fold: int) -> tuple[int, int, int]:
    return tuple(  # type: ignore[return-value]
        conn.execute(f"SELECT COUNT(*) FROM {t} WHERE fold = ?", (fold,)).fetchone()[0]
        for t in ("thresholds", "per_image_metrics", "training_history")
    )


def test_insert_run_writes_the_manifest(db: sqlite3.Connection) -> None:
    row = db.execute(
        "SELECT variant, config, config_hash, seed, git_dirty, git_tag, cuda_version, "
        "deterministic_ops, started_at, finished_at, is_reported FROM runs"
    ).fetchone()
    assert row == (
        "baseline",
        json.dumps({"a": [1, 2], "b": 1}),
        "hash",
        7,
        0,
        None,
        None,
        1,
        T0,
        None,
        0,
    )


def test_finds_the_unfinished_matching_run(db: sqlite3.Connection) -> None:
    key = ("baseline", "hash", "c0ffee", "data")
    assert find_resumable_run(db, *key) == "run-a"
    for i, changed in enumerate(key):
        other = list(key)
        other[i] = changed + "-other"
        assert find_resumable_run(db, *other) is None
    db.execute("UPDATE runs SET finished_at = ?", (T1,))
    assert find_resumable_run(db, *key) is None


def test_several_matching_runs_are_an_error(db: sqlite3.Connection) -> None:
    insert_run(db, replace(MANIFEST, run_id="run-b"))
    with pytest.raises(RuntimeError, match="run-a.*run-b"):
        find_resumable_run(db, "baseline", "hash", "c0ffee", "data")


def test_stored_fold_rows_round_trip(db: sqlite3.Connection) -> None:
    ids = [str(i) for i in range(21, 41)]
    for image_id in ids[2:]:
        insert(db, "images", image_row(image_id))
    folds = make_folds(ids, [], n_folds=5, n_val=2, seed=1)
    write_fold_assignments(db, "run-a", "drive", folds)
    rows = stored_fold_rows(db, "run-a")
    assert len(rows) == 100
    assert rows == sorted(rows)
    assert stored_fold_rows(db, "run-z") == []


def test_start_fold_inserts_running(db: sqlite3.Connection) -> None:
    assert start_fold(db, "run-a", 1, T0) == 1
    assert fold_statuses(db, "run-a") == {1: "running"}


def test_restarting_a_running_fold_clears_partial_rows(db: sqlite3.Connection) -> None:
    start_fold(db, "run-a", 1, T0)
    start_fold(db, "run-a", 2, T0)
    # Rows an interrupted attempt might have left behind, in fold 1 only.
    insert(
        db, "thresholds", {"run_id": "run-a", "fold": 1, "threshold": 0.5, "selection_rule": "r"}
    )
    insert(
        db,
        "training_history",
        {"run_id": "run-a", "fold": 1, "epoch": 1, "train_loss": 1.0},
    )
    insert(
        db,
        "per_image_metrics",
        {
            "run_id": "run-a",
            "dataset": "drive",
            "image_id": "21",
            "fold": 1,
            "prediction_mode": "single",
        },
    )
    insert(
        db, "thresholds", {"run_id": "run-a", "fold": 2, "threshold": 0.5, "selection_rule": "r"}
    )
    assert start_fold(db, "run-a", 1, T1) == 2
    assert counts(db, 1) == (0, 0, 0)
    assert counts(db, 2) == (1, 0, 0)
    assert db.execute("SELECT attempts, started_at FROM fold_status WHERE fold = 1").fetchone() == (
        2,
        T1,
    )
    assert start_fold(db, "run-a", 1, T1) == 3


def test_complete_fold_writes_everything(db: sqlite3.Connection) -> None:
    start_fold(db, "run-a", 1, T0)
    complete_fold(db, record(), T1)
    assert fold_statuses(db, "run-a") == {1: "complete"}
    assert counts(db, 1) == (1, 2, 2)
    assert db.execute(
        "SELECT checkpoint_sha256, completed_at FROM fold_status WHERE fold = 1"
    ).fetchone() == ("ab" * 32, T1)
    assert db.execute("SELECT threshold, selection_rule, val_dice FROM thresholds").fetchone() == (
        0.42,
        "rule",
        0.81,
    )
    assert db.execute(
        "SELECT dice, sensitivity, specificity, precision_score, accuracy, "
        "predicted_vessel_fraction, prediction_mode, auc_roc FROM per_image_metrics "
        "WHERE image_id = '21'"
    ).fetchone() == (0.8, 0.75, 0.97, None, 0.95, 0.12, "single", None)
    with pytest.raises(ValueError, match="already complete"):
        start_fold(db, "run-a", 1, T1)


def test_complete_fold_is_all_or_nothing(db: sqlite3.Connection) -> None:
    start_fold(db, "run-a", 1, T0)
    # An image missing from the images table fails its foreign key after the
    # threshold and history inserts have run.
    bad = record(test_metrics=[ImageMetrics("drive", "99", METRICS)])
    with pytest.raises(sqlite3.IntegrityError):
        complete_fold(db, bad, T1)
    assert counts(db, 1) == (0, 0, 0)
    assert fold_statuses(db, "run-a") == {1: "running"}


@pytest.mark.parametrize("started", [False, True])
def test_complete_fold_needs_a_running_fold(db: sqlite3.Connection, started: bool) -> None:
    if started:
        start_fold(db, "run-a", 1, T0)
        complete_fold(db, record(), T1)
    with pytest.raises(ValueError, match="not running"):
        complete_fold(db, record(), T1)


def test_complete_fold_needs_history(db: sqlite3.Connection) -> None:
    start_fold(db, "run-a", 1, T0)
    with pytest.raises(ValueError, match="no training history"):
        complete_fold(db, record(history=[]), T1)


def test_finish_run_needs_every_started_fold_complete(db: sqlite3.Connection) -> None:
    start_fold(db, "run-a", 1, T0)
    start_fold(db, "run-a", 2, T0)
    complete_fold(db, record(), T1)
    with pytest.raises(ValueError, match=r"\[2\]"):
        finish_run(db, "run-a", T1)
    # Each image has one out-of-fold row per run, so fold 2 brings no test rows here.
    complete_fold(db, record(fold=2, test_metrics=[]), T1)
    finish_run(db, "run-a", T1)
    assert db.execute("SELECT finished_at FROM runs").fetchone() == (T1,)
