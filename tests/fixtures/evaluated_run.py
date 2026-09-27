"""A finished, trained-looking run on synthetic data, for tests that evaluate or report it.

The run has fake probabilities in place of a model's, so it needs no
TensorFlow, but every row and file is what ``train`` would have written.
"""

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

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
from retinal_vessels.evaluate import evaluate_run
from retinal_vessels.metrics import binarize, binary_metrics
from retinal_vessels.provenance import (
    ChecksumReport,
    GitState,
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
# The fixture's run trained from a clean tree at this commit.
CODE = GitState(commit="c0ffee", dirty=False, tag=None)


@dataclass
class Finished:
    conn: sqlite3.Connection
    db: Path
    samples: list[Sample]
    report: ChecksumReport
    dirs: OutputDirs
    folds: list[Fold]
    probabilities: dict[str, np.ndarray]

    def evaluate(self, code: GitState = CODE) -> Path:
        return evaluate_run(self.conn, RUN_ID, self.samples, self.report, self.dirs, code)

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
