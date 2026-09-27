import json
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from retinal_vessels import evaluate
from retinal_vessels.config import load_config
from retinal_vessels.data import Fold, Sample, make_folds
from retinal_vessels.datasets.drive import load_drive
from retinal_vessels.db import (
    EpochRecord,
    FoldRecord,
    ImageMetrics,
    complete_fold,
    connect,
    create_schema,
    finish_run,
    image_record,
    insert_run,
    start_fold,
    write_fold_assignments,
    write_images,
)
from retinal_vessels.evaluate import EvaluationError, evaluate_run, latest_finished_run
from retinal_vessels.metrics import auc_roc, binarize, binary_metrics, reliability, thin_edge
from retinal_vessels.provenance import (
    ChecksumReport,
    RunManifest,
    check_or_write_checksums,
    data_hash,
)
from retinal_vessels.train import OutputDirs, save_predictions

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "20260927T120000Z-eval"
THRESHOLD = 0.5
T0 = "2026-09-27T12:00:00Z"
T1 = "2026-09-27T13:00:00Z"
ENV = {
    "python_version": "3.13.1",
    "tensorflow_version": "2.21.0",
    "cuda_version": None,
    "device": "cpu",
    "gpu_type": "cpu",
    "compute_platform": "local",
}
EVALUATED = "auc_roc, auc_pr, brier, thin_sensitivity, thick_sensitivity"


@pytest.fixture(autouse=True)
def restore_root_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # main() installs the package's log handler on the root logger.
    monkeypatch.delenv("DRIVE_DIR", raising=False)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


@dataclass
class Finished:
    conn: sqlite3.Connection
    db: Path
    samples: list[Sample]
    report: ChecksumReport
    dirs: OutputDirs
    folds: list[Fold]
    probabilities: dict[str, np.ndarray]

    def evaluate(self) -> Path:
        return evaluate_run(self.conn, RUN_ID, self.samples, self.report, self.dirs)

    def rows(self) -> list[tuple[object, ...]]:
        return self.conn.execute(
            f"SELECT image_id, {EVALUATED} FROM per_image_metrics ORDER BY image_id"
        ).fetchall()

    def sample(self, image_id: str) -> Sample:
        return next(s for s in self.samples if s.image_id == image_id)


def fake_probability(sample: Sample, rng: np.random.Generator) -> np.ndarray:
    assert sample.label is not None
    noisy = np.where(sample.label, 0.7, 0.2) + rng.normal(0, 0.2, sample.label.shape)
    return np.where(sample.fov, np.clip(noisy, 0, 1), 0).astype(np.float32)


@pytest.fixture
def finished(synthetic_data_root: Path, tmp_path: Path) -> Iterator[Finished]:
    config = load_config(REPO / "configs" / "smoke.yaml")
    drive = synthetic_data_root / "DRIVE"
    _, report = check_or_write_checksums(
        synthetic_data_root / "CHECKSUMS.sha256", {"DRIVE": drive}, write_missing=True
    )
    samples = load_drive(drive, "training")
    db = synthetic_data_root / "experiments.db"
    conn = connect(db)
    create_schema(conn)
    write_images(conn, [image_record(s) for s in samples])
    insert_run(
        conn,
        RunManifest(
            run_id=RUN_ID,
            variant=config.variant,
            config=config.as_dict(),
            config_hash="hash",
            seed=config.seed,
            git_commit="c0ffee",
            git_dirty=False,
            git_tag=None,
            data_hash=data_hash(report.checked),
            deterministic_ops=True,
            started_at=T0,
            environment=ENV,
        ),
    )
    folds = make_folds(
        [s.image_id for s in samples],
        {s.image_id for s in samples if s.has_abnormality},
        n_folds=config.folds.n_folds,
        n_val=config.folds.n_val,
        seed=config.folds.seed,
    )
    write_fold_assignments(conn, RUN_ID, "drive", folds)
    dirs = OutputDirs(results=synthetic_data_root / "results", models=tmp_path / "models")
    rng = np.random.default_rng(0)
    by_id = {s.image_id: s for s in samples}
    probabilities = {}
    for fold in folds:
        start_fold(conn, RUN_ID, fold.number, T0)
        probs = {i: fake_probability(by_id[i], rng) for i in fold.test}
        save_predictions(dirs, RUN_ID, fold.number, probs)
        probabilities.update(probs)
        metrics = []
        for image_id, prob in probs.items():
            s = by_id[image_id]
            assert s.label is not None
            m = binary_metrics(binarize(prob, THRESHOLD), s.label, s.fov)
            metrics.append(ImageMetrics("drive", image_id, m))
        record = FoldRecord(
            run_id=RUN_ID,
            fold=fold.number,
            checkpoint_sha256="ab" * 32,
            threshold=THRESHOLD,
            selection_rule="rule",
            val_dice=0.5,
            history=[EpochRecord(1, 0.5, 0.5)],
            test_metrics=metrics,
        )
        complete_fold(conn, record, T1)
    finish_run(conn, RUN_ID, T1)
    yield Finished(conn, db, samples, report, dirs, folds, probabilities)
    conn.close()


def test_fills_every_evaluated_column(finished: Finished) -> None:
    assert all(v is None for row in finished.rows() for v in row[1:])
    finished.evaluate()
    rows = finished.rows()
    assert len(rows) == 20
    for _, roc, pr, brier, thin, thick in rows:
        assert roc is not None and pr is not None and brier is not None
        assert thin is not None or thick is not None
    first = finished.sample(rows[0][0])
    assert first.label is not None
    expected = auc_roc(finished.probabilities[first.image_id], first.label, first.fov)
    assert rows[0][1] == expected


def test_writes_the_run_summary(finished: Finished) -> None:
    path = finished.evaluate()
    assert path == finished.dirs.results / RUN_ID / "evaluation.json"
    summary = json.loads(path.read_text())
    assert summary["run_id"] == RUN_ID
    assert summary["evaluation"] == {"reliability_bins": 10, "thin_quantile": 0.5}
    assert [f["fold"] for f in summary["folds"]] == [1, 2, 3, 4, 5]

    fold = finished.folds[0]
    tuning = [finished.sample(i) for i in (*fold.train, *fold.val)]
    labels = [s.label for s in tuning if s.label is not None]
    edge = thin_edge(labels, [s.fov for s in tuning], 0.5)
    assert summary["folds"][0]["thin_edge"] == edge
    assert summary["folds"][0]["threshold"] == THRESHOLD

    ids = sorted(finished.probabilities)
    samples = [finished.sample(i) for i in ids]
    table = reliability(
        [finished.probabilities[i] for i in ids],
        [s.label for s in samples if s.label is not None],
        [s.fov for s in samples],
        10,
    )
    assert summary["reliability"]["counts"] == list(table.counts)
    assert summary["reliability"]["pooled_brier"] == pytest.approx(table.pooled_brier)
    assert sum(table.counts) == sum(int(s.fov.sum()) for s in samples)
    assert not list(path.parent.glob("*.partial"))


def test_rerunning_gives_the_same_results(finished: Finished) -> None:
    path = finished.evaluate()
    rows, text = finished.rows(), path.read_text()
    finished.evaluate()
    assert finished.rows() == rows
    assert path.read_text() == text


def test_rewrites_the_manifest_from_the_database(finished: Finished) -> None:
    manifest = finished.dirs.results / RUN_ID / "manifest.json"
    assert not manifest.exists()
    finished.evaluate()
    assert json.loads(manifest.read_text())["finished_at"] == T1


def assert_nothing_written(finished: Finished, match: str) -> None:
    with pytest.raises(EvaluationError, match=match):
        finished.evaluate()
    assert all(v is None for row in finished.rows() for v in row[1:])
    assert not (finished.dirs.results / RUN_ID / "evaluation.json").exists()


def test_refuses_an_unfinished_run(finished: Finished) -> None:
    finished.conn.execute("UPDATE runs SET finished_at = NULL")
    assert_nothing_written(finished, "5 of 5 folds complete")


def test_refuses_a_run_missing_a_fold(finished: Finished) -> None:
    finished.conn.execute("DELETE FROM fold_status WHERE fold = 5")
    assert_nothing_written(finished, "4 of 5 folds complete")


def test_refuses_an_unknown_run(finished: Finished) -> None:
    with pytest.raises(EvaluationError, match="no run 'other'"):
        evaluate_run(finished.conn, "other", finished.samples, finished.report, finished.dirs)


def first_test_file(finished: Finished) -> Path:
    return finished.dirs.predictions(RUN_ID) / f"{finished.folds[0].test[0]}.npy"


def test_refuses_a_missing_prediction(finished: Finished) -> None:
    first_test_file(finished).unlink()
    assert_nothing_written(finished, "is missing")


def test_refuses_a_changed_prediction(finished: Finished) -> None:
    path = first_test_file(finished)
    np.save(path, np.load(path) * np.float32(0.5))
    assert_nothing_written(finished, "does not match")


def test_refuses_a_fold_without_hashes(finished: Finished) -> None:
    finished.dirs.prediction_hashes(RUN_ID, 2).unlink()
    assert_nothing_written(finished, "fold 2 has no prediction hashes")


def test_refuses_hashes_for_other_images(finished: Finished) -> None:
    fold = finished.folds[0]
    kept = {i: finished.probabilities[i] for i in fold.test[:-1]}
    save_predictions(finished.dirs, RUN_ID, fold.number, kept)
    assert_nothing_written(finished, "but fold 1 tests")


def test_refuses_swapped_predictions_with_matching_hashes(finished: Finished) -> None:
    # Hashes written after a swap pass, so only the recomputed metrics catch it.
    fold = finished.folds[0]
    a, b = fold.test[:2]
    swapped = {i: finished.probabilities[i] for i in fold.test}
    swapped[a], swapped[b] = swapped[b], swapped[a]
    save_predictions(finished.dirs, RUN_ID, fold.number, swapped)
    assert_nothing_written(finished, "does not reproduce the metrics stored")


def test_refuses_a_prediction_of_the_wrong_dtype(finished: Finished) -> None:
    fold = finished.folds[0]
    probs = {i: finished.probabilities[i] for i in fold.test}
    probs[fold.test[0]] = probs[fold.test[0]].astype(np.float64)
    save_predictions(finished.dirs, RUN_ID, fold.number, probs)
    assert_nothing_written(finished, "holds float64")


def test_refuses_data_that_differs_from_the_run(finished: Finished) -> None:
    checked = dict(finished.report.checked)
    first = next(iter(checked))
    checked[first] = "0" * 64
    finished.report = replace(finished.report, checked=checked)
    assert_nothing_written(finished, "does not match the data")


def test_latest_finished_run(finished: Finished) -> None:
    assert latest_finished_run(finished.conn) == RUN_ID
    finished.conn.execute("UPDATE runs SET finished_at = NULL")
    assert latest_finished_run(finished.conn) is None


def cli(finished: Finished, root: Path, *extra: str) -> int:
    return evaluate.main(
        [
            "--data-root",
            str(root),
            "--checksums",
            str(root / "CHECKSUMS.sha256"),
            "--db",
            str(finished.db),
            "--results-dir",
            str(finished.dirs.results),
            *extra,
        ]
    )


def test_main_evaluates_the_latest_finished_run(
    finished: Finished, synthetic_data_root: Path
) -> None:
    finished.conn.commit()
    assert cli(finished, synthetic_data_root) == 0
    assert (finished.dirs.results / RUN_ID / "evaluation.json").is_file()
    assert all(row[1] is not None for row in finished.rows())


def test_main_reports_failures(
    finished: Finished, synthetic_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert cli(finished, synthetic_data_root, "--run-id", "other") == 1
    assert "no run 'other'" in caplog.text
    finished.conn.execute("UPDATE runs SET finished_at = NULL")
    finished.conn.commit()
    assert cli(finished, synthetic_data_root) == 1
    assert "no finished run" in caplog.text
    assert cli(finished, synthetic_data_root / "missing") == 1


def test_main_needs_the_database(synthetic_data_root: Path, tmp_path: Path) -> None:
    assert evaluate.main(["--db", str(tmp_path / "none.db")]) == 1
