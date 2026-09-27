import sqlite3
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from retinal_vessels.data import Sample, make_folds
from retinal_vessels.db import (
    IMAGE_COLUMNS,
    image_record,
    write_fold_assignments,
    write_images,
)
from tests.fixtures.database import RUN, insert

SHAPE = (6, 5)
IDS = [str(i) for i in range(21, 41)]


def sample(image_id: str = "21", **overrides: Any) -> Sample:
    fov = np.zeros(SHAPE, dtype=bool)
    fov[1:5, 1:4] = True  # 12 pixels
    label = np.zeros(SHAPE, dtype=bool)
    label[2, 1:4] = True  # 3 pixels inside the FOV
    label[0, 0] = True  # 1 pixel outside it
    fields: dict[str, Any] = {
        "dataset": "drive",
        "image_id": image_id,
        "split": "training",
        "image": np.zeros((*SHAPE, 3), dtype=np.uint8),
        "fov": fov,
        "label": label,
    }
    return Sample(**{**fields, **overrides})


def test_record_counts_vessels_inside_the_fov_only() -> None:
    record = image_record(sample())
    assert record.fov_pixels == 12
    assert record.vessel_fraction_in_fov == 3 / 12
    assert (record.has_labels, record.has_abnormality) == (1, 0)
    assert record.abnormality_note is None
    assert record.fov_source == "official"


def test_record_without_labels_has_no_fraction() -> None:
    record = image_record(sample(label=None, split="test"))
    assert record.vessel_fraction_in_fov is None
    assert record.has_labels == 0


def test_record_carries_the_note_and_source() -> None:
    record = image_record(sample(abnormality_note="note", patient_id="c1"), fov_source="generated")
    assert (record.has_abnormality, record.abnormality_note) == (1, "note")
    assert (record.patient_id, record.fov_source) == ("c1", "generated")


def stored(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
    cols = ", ".join(IMAGE_COLUMNS)
    return conn.execute(f"SELECT {cols} FROM images ORDER BY image_id").fetchall()


def test_write_images_inserts_every_column(conn: sqlite3.Connection) -> None:
    records = [image_record(sample("21")), image_record(sample("25", abnormality_note="n"))]
    assert write_images(conn, records) == 2
    assert stored(conn) == [
        ("drive", "21", None, "training", 0, None, 12, "official", 0.25, 1),
        ("drive", "25", None, "training", 1, "n", 12, "official", 0.25, 1),
    ]


def test_write_images_is_idempotent(conn: sqlite3.Connection) -> None:
    records = [image_record(sample("21")), image_record(sample("22"))]
    write_images(conn, records)
    assert write_images(conn, records) == 0
    assert write_images(conn, [*records, image_record(sample("23"))]) == 1
    assert len(stored(conn)) == 3


def test_write_images_rejects_changed_data(conn: sqlite3.Connection) -> None:
    record = image_record(sample("21"))
    write_images(conn, [record])
    changed = replace(record, vessel_fraction_in_fov=0.3)
    with pytest.raises(ValueError, match="drive 21 differs from the data"):
        write_images(conn, [changed])
    assert stored(conn)[0][8] == 0.25


def test_write_images_rolls_back_on_a_bad_row(conn: sqlite3.Connection) -> None:
    good = image_record(sample("21"))
    bad = replace(image_record(sample("22")), has_abnormality=1)
    with pytest.raises(sqlite3.IntegrityError):
        write_images(conn, [good, bad])
    assert stored(conn) == []


def test_write_fold_assignments_stores_every_role(conn: sqlite3.Connection) -> None:
    write_images(conn, [image_record(sample(i)) for i in IDS])
    insert(conn, "runs", RUN)
    folds = make_folds(IDS, {"25", "26", "32"}, n_folds=5, n_val=2, seed=0)
    write_fold_assignments(conn, "r1", "drive", folds)
    counts = conn.execute(
        "SELECT fold, role, COUNT(*) FROM fold_assignments GROUP BY fold, role ORDER BY fold, role"
    ).fetchall()
    assert counts == [
        (f, r, n) for f in range(1, 6) for r, n in (("test", 4), ("train", 14), ("val", 2))
    ]
    tests = conn.execute(
        "SELECT image_id FROM fold_assignments WHERE fold = 2 AND role = 'test' ORDER BY image_id"
    ).fetchall()
    assert [t[0] for t in tests] == list(folds[1].test)


def test_write_fold_assignments_needs_known_images(conn: sqlite3.Connection) -> None:
    insert(conn, "runs", RUN)
    folds = make_folds(IDS, set(), n_folds=5, n_val=2, seed=0)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        write_fold_assignments(conn, "r1", "drive", folds)
    assert conn.execute("SELECT COUNT(*) FROM fold_assignments").fetchone() == (0,)
