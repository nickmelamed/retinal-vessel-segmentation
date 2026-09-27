import sqlite3
from pathlib import Path

import pytest

from retinal_vessels.data import Fold, make_folds
from retinal_vessels.datasets.drive import load_drive
from retinal_vessels.db import (
    QUERIES_DIR,
    image_record,
    run_query,
    write_fold_assignments,
    write_images,
)
from tests.fixtures.database import RUN, insert

AUDIT = "04_leakage_audit"


@pytest.fixture
def folds(conn: sqlite3.Connection, synthetic_data_root: Path) -> list[Fold]:
    """Write the synthetic training images and a valid run r1, and return its folds."""
    samples = load_drive(synthetic_data_root / "DRIVE", "training")
    write_images(conn, [image_record(s) for s in samples])
    abnormal = {s.image_id for s in samples if s.has_abnormality}
    result = make_folds([s.image_id for s in samples], abnormal, n_folds=5, n_val=2, seed=0)
    insert(conn, "runs", RUN)
    write_fold_assignments(conn, "r1", "drive", result)
    return result


def leaks(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    # SQLite leaves the order of group_concat unspecified, so compare the
    # detail column as a set.
    return [(*row[:4], set(str(row[4]).split(","))) for row in run_query(conn, AUDIT)]


def test_query_has_a_header_stating_its_question() -> None:
    text = (QUERIES_DIR / f"{AUDIT}.sql").read_text()
    assert text.startswith("-- Question:")


def test_valid_folds_return_zero_rows(conn: sqlite3.Connection, folds: list[Fold]) -> None:
    assert run_query(conn, AUDIT) == []


def test_several_valid_runs_return_zero_rows(conn: sqlite3.Connection, folds: list[Fold]) -> None:
    # Each run puts every image in the test role once. Mixing runs up would
    # make every image look like a test image in two folds.
    insert(conn, "runs", {**RUN, "run_id": "r2"})
    ids = [i for f in folds for i in f.test]
    write_fold_assignments(conn, "r2", "drive", make_folds(ids, set(), 5, 2, seed=1))
    assert run_query(conn, AUDIT) == []


def test_two_roles_in_one_fold_is_a_leak(conn: sqlite3.Connection, folds: list[Fold]) -> None:
    leaked = folds[0].train[0]
    insert(conn, "fold_assignments", {**_row(1, leaked), "role": "val"})
    assert leaks(conn) == [
        ("r1", "drive", leaked, "more than one role in fold 1", {"train", "val"})
    ]


def test_test_in_two_folds_is_a_leak(conn: sqlite3.Connection, folds: list[Fold]) -> None:
    leaked = folds[0].test[0]
    # Move the image's fold 2 row to the test role, so it keeps one role per
    # fold and only the second rule can catch it.
    conn.execute(
        "UPDATE fold_assignments SET role = 'test' "
        "WHERE run_id = 'r1' AND fold = 2 AND image_id = ?",
        (leaked,),
    )
    assert leaks(conn) == [("r1", "drive", leaked, "test in more than one fold", {"1", "2"})]


def test_every_leak_is_reported(conn: sqlite3.Connection, folds: list[Fold]) -> None:
    for fold in folds:
        insert(conn, "fold_assignments", {**_row(fold.number, fold.test[0]), "role": "train"})
    rows = run_query(conn, AUDIT)
    assert len(rows) == 5
    assert {r[2] for r in rows} == {f.test[0] for f in folds}


def test_missing_query_raises(conn: sqlite3.Connection, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="query not found"):
        run_query(conn, AUDIT, queries_dir=tmp_path)


def _row(fold: int, image_id: str) -> dict[str, object]:
    return {"run_id": "r1", "fold": fold, "dataset": "drive", "image_id": image_id}
