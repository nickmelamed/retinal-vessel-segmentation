import sqlite3
from dataclasses import replace
from typing import Any

import pytest

from retinal_vessels.db import (
    QUERIES_DIR,
    EpochRecord,
    EvaluationRecord,
    FoldRecord,
    ImageMetrics,
    complete_fold,
    fill_evaluation,
    finish_run,
    insert_run,
    run_query,
    start_fold,
)
from retinal_vessels.metrics import BinaryMetrics
from retinal_vessels.provenance import RunManifest
from tests.fixtures.database import insert

T0 = "2026-09-27T10:00:00Z"
T1 = "2026-09-27T11:00:00Z"
MANIFEST = RunManifest(
    run_id="base",
    variant="baseline",
    config={},
    config_hash="hash",
    seed=7,
    git_commit="c0ffee",
    git_dirty=False,
    git_tag=None,
    data_hash="data",
    deterministic_ops=True,
    started_at=T0,
    environment={
        "python_version": "3.13.1",
        "tensorflow_version": "2.21.0",
        "cuda_version": None,
        "device": "cpu",
        "gpu_type": "cpu",
        "compute_platform": "local",
    },
)
NOTE = "synthetic note"
# Dice per image and fold for each run. "open" never finishes.
DICE = {
    "base": {1: {"21": 0.8, "22": 0.6}, 2: {"23": 0.7}},
    "ablate": {1: {"21": 0.7, "22": 0.65}, 2: {"23": 0.9}},
    "open": {1: {"21": 0.1}},
}
VARIANTS = {"base": "baseline", "ablate": "dice_only", "open": "baseline"}
THRESHOLDS = {1: 0.4, 2: 0.55}


def metrics(dice: float) -> BinaryMetrics:
    return BinaryMetrics(
        dice=dice,
        sensitivity=dice,
        specificity=0.9,
        precision=0.5,
        accuracy=0.95,
        predicted_vessel_fraction=0.1,
    )


def history(best: float) -> list[EpochRecord]:
    # The best validation Dice first appears at epoch 2 and repeats at epoch 3.
    return [
        EpochRecord(1, 0.9, best / 2),
        EpochRecord(2, 0.7, best),
        EpochRecord(3, 0.6, best),
        EpochRecord(4, 0.5, best / 3),
    ]


def assign(conn: sqlite3.Connection, run_id: str, folds: dict[int, dict[str, float]]) -> None:
    # Each fold tests its own images and trains on the rest.
    for fold, tested in folds.items():
        for image_id in ("21", "22", "23"):
            role = "test" if image_id in tested else "train"
            row = {"run_id": run_id, "fold": fold, "dataset": "drive", "image_id": image_id}
            insert(conn, "fold_assignments", {**row, "role": role})


@pytest.fixture
def db(conn: sqlite3.Connection) -> sqlite3.Connection:
    for image_id in ("21", "22", "23"):
        abnormal = image_id == "23"
        insert(
            conn,
            "images",
            {
                "dataset": "drive",
                "image_id": image_id,
                "split": "training",
                "has_abnormality": int(abnormal),
                "abnormality_note": NOTE if abnormal else None,
                "fov_pixels": 10,
                "fov_source": "official",
                "has_labels": 1,
            },
        )
    for run_id, folds in DICE.items():
        insert_run(conn, replace(MANIFEST, run_id=run_id, variant=VARIANTS[run_id]))
        assign(conn, run_id, folds)
        for fold, images in folds.items():
            start_fold(conn, run_id, fold, T0)
            record = FoldRecord(
                run_id=run_id,
                fold=fold,
                checkpoint_sha256="ab" * 32,
                threshold=THRESHOLDS[fold],
                selection_rule="rule",
                val_dice=0.8,
                history=history(0.8),
                test_metrics=[ImageMetrics("drive", i, metrics(d)) for i, d in images.items()],
            )
            complete_fold(conn, record, T1)
    finish_run(conn, "base", T1)
    finish_run(conn, "ablate", T1)
    start_fold(conn, "open", 2, T0)
    evaluated = [
        EvaluationRecord("drive", "21", 0.9, 0.8, 0.05, 0.6, None),
        EvaluationRecord("drive", "22", 0.8, 0.7, 0.07, 0.4, 0.9),
    ]
    fill_evaluation(conn, "base", evaluated)
    return conn


def query(conn: sqlite3.Connection, name: str) -> list[dict[str, Any]]:
    cursor = conn.execute((QUERIES_DIR / f"{name}.sql").read_text(encoding="utf-8"))
    columns = [c[0] for c in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    assert len(rows) == len(run_query(conn, name))
    return rows


def test_fold_summary_averages_each_finished_fold(db: sqlite3.Connection) -> None:
    rows = query(db, "01_fold_summary")
    keys = [(r["run_id"], r["fold"]) for r in rows]
    assert keys == [("base", 1), ("base", 2), ("ablate", 1), ("ablate", 2)]
    first = rows[0]
    assert (first["variant"], first["is_reported"], first["n_images"]) == ("baseline", 0, 2)
    assert first["n_evaluated"] == 2
    assert first["mean_dice"] == pytest.approx(0.7)
    assert first["mean_auc_roc"] == pytest.approx(0.85)
    assert first["mean_auc_pr"] == pytest.approx(0.75)
    assert first["mean_brier"] == pytest.approx(0.06)
    assert first["mean_thin_sensitivity"] == pytest.approx(0.5)
    # NULLs are skipped, so the thick mean covers only image 22.
    assert first["mean_thick_sensitivity"] == pytest.approx(0.9)
    assert first["mean_precision"] == pytest.approx(0.5)
    unevaluated = rows[1]
    assert (unevaluated["n_images"], unevaluated["n_evaluated"]) == (1, 0)
    assert unevaluated["mean_auc_roc"] is None


def test_worst_images_ranks_each_finished_run(db: sqlite3.Connection) -> None:
    rows = query(db, "02_worst_images")
    ranked = [(r["run_id"], r["dice_rank"], r["image_id"], r["dice"]) for r in rows]
    assert ranked == [
        ("base", 1, "22", 0.6),
        ("base", 2, "23", 0.7),
        ("base", 3, "21", 0.8),
        ("ablate", 1, "22", 0.65),
        ("ablate", 2, "21", 0.7),
        ("ablate", 3, "23", 0.9),
    ]
    abnormal = {r["image_id"] for r in rows if r["has_abnormality"]}
    assert abnormal == {"23"}
    assert rows[0]["fold"] == 1 and rows[1]["fold"] == 2


def test_worst_images_breaks_ties_by_image_id(db: sqlite3.Connection) -> None:
    db.execute("UPDATE per_image_metrics SET dice = 0.5 WHERE run_id = 'base'")
    rows = [r for r in query(db, "02_worst_images") if r["run_id"] == "base"]
    assert [r["image_id"] for r in rows] == ["21", "22", "23"]


def test_variant_comparison_pairs_images_against_the_baseline(db: sqlite3.Connection) -> None:
    rows = query(db, "03_variant_comparison")
    assert [(r["baseline_run_id"], r["other_run_id"], r["image_id"]) for r in rows] == [
        ("base", "ablate", "21"),
        ("base", "ablate", "22"),
        ("base", "ablate", "23"),
    ]
    assert [r["delta_dice"] for r in rows] == pytest.approx([-0.1, 0.05, 0.2])
    assert rows[0]["other_variant"] == "dice_only"
    assert rows[0]["delta_specificity"] == pytest.approx(0.0)
    # The baseline was evaluated and the other run was not.
    assert rows[0]["delta_auc_pr"] is None


def test_variant_comparison_needs_both_runs_finished(db: sqlite3.Connection) -> None:
    db.execute("UPDATE runs SET finished_at = NULL WHERE run_id = 'ablate'")
    assert query(db, "03_variant_comparison") == []


def test_threshold_log_covers_every_run(db: sqlite3.Connection) -> None:
    rows = query(db, "05_threshold_log")
    got = [(r["run_id"], r["fold"], r["threshold"], r["is_finished"]) for r in rows]
    assert got == [
        ("base", 1, 0.4, 1),
        ("base", 2, 0.55, 1),
        ("open", 1, 0.4, 0),
        ("ablate", 1, 0.4, 1),
        ("ablate", 2, 0.55, 1),
    ]
    assert {(r["best_epoch"], r["n_epochs"]) for r in rows} == {(2, 4)}
    assert {(r["selection_rule"], r["val_dice"]) for r in rows} == {("rule", 0.8)}


def add_finished_run(
    conn: sqlite3.Connection, run_id: str, folds: dict[int, dict[str, float]], data_hash: str
) -> None:
    insert_run(conn, replace(MANIFEST, run_id=run_id, variant="dice_only", data_hash=data_hash))
    assign(conn, run_id, folds)
    for fold, images in folds.items():
        start_fold(conn, run_id, fold, T0)
        record = FoldRecord(
            run_id=run_id,
            fold=fold,
            checkpoint_sha256="ab" * 32,
            threshold=THRESHOLDS[fold],
            selection_rule="rule",
            val_dice=0.8,
            history=history(0.8),
            test_metrics=[ImageMetrics("drive", i, metrics(d)) for i, d in images.items()],
        )
        complete_fold(conn, record, T1)
    finish_run(conn, run_id, T1)


def test_variant_comparison_pairs_only_identical_folds_and_data(db: sqlite3.Connection) -> None:
    # "shifted" held image 21 out in fold 2, where the baseline held it out in
    # fold 1. Images 22 and 23 keep their fold numbers, but the models behind
    # them trained on different images, so nothing is paired. "elsewhere" has
    # the baseline's folds but trained on other data.
    add_finished_run(db, "shifted", {1: {"22": 0.5}, 2: {"21": 0.5, "23": 0.5}}, "data")
    add_finished_run(db, "elsewhere", DICE["ablate"], "other data")
    rows = query(db, "03_variant_comparison")
    pairs = {(r["other_run_id"], r["image_id"]) for r in rows}
    assert {i for run, i in pairs if run == "shifted"} == set()
    assert {i for run, i in pairs if run == "elsewhere"} == set()
    assert {i for run, i in pairs if run == "ablate"} == {"21", "22", "23"}


def test_variant_comparison_notices_a_changed_validation_split(db: sqlite3.Connection) -> None:
    # Same test folds, but one image moved from training to validation.
    db.execute(
        "UPDATE fold_assignments SET role = 'val' "
        "WHERE run_id = 'ablate' AND fold = 1 AND image_id = '23'"
    )
    assert query(db, "03_variant_comparison") == []
