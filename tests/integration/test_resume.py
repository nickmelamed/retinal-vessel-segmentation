"""Resumable cross-validation on synthetic data (SPEC sections 14 and 16).

One uninterrupted smoke run is the reference. Every other run must end with
exactly the same fold assignments, thresholds, metrics, history, and
predictions, however it was interrupted. That holds bit for bit only because
the smoke config turns on deterministic ops and each fold's seeds depend
only on (seed, fold) and (seed, fold, epoch).
"""

import logging
import shutil
import sqlite3
import subprocess
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from retinal_vessels import train
from retinal_vessels.config import Config, load_config
from retinal_vessels.data import Fold, Sample
from retinal_vessels.datasets.drive import load_drive
from retinal_vessels.db import (
    FoldRecord,
    connect,
    create_schema,
    fold_statuses,
    image_record,
    write_images,
)
from retinal_vessels.provenance import (
    ChecksumReport,
    compute_checksums,
    format_checksums,
)
from tests.fixtures.synthetic_drive import write_synthetic_drive

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
ENV = {
    "python_version": "3.13.1",
    "tensorflow_version": "2.21.0",
    "cuda_version": None,
    "device": "cpu",
    "gpu_type": "cpu",
    "compute_platform": "local",
}
# Rows that must match between runs, without run ids and timestamps.
COMPARED = {
    "fold_assignments": "fold, dataset, image_id, role",
    "thresholds": "fold, threshold, selection_rule, val_dice",
    "per_image_metrics": "fold, image_id, dice, sensitivity, specificity, precision_score, "
    "accuracy, predicted_vessel_fraction",
    "training_history": "fold, epoch, train_loss, val_dice",
}


class Interrupted(Exception):
    pass


@dataclass(frozen=True)
class Setup:
    config: Config
    samples: list[Sample]
    report: ChecksumReport
    data_root: Path
    repo: Path


@dataclass(frozen=True)
class Outcome:
    db: Path
    dirs: train.OutputDirs
    run_id: str


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> Setup:
    root = tmp_path_factory.mktemp("data")
    drive = write_synthetic_drive(root)
    entries = compute_checksums({"DRIVE": drive})
    (root / "CHECKSUMS.sha256").write_text(format_checksums(entries))
    # A throwaway repository, so the git state these runs record is clean
    # whatever state the working copy running the tests is in.
    repo = tmp_path_factory.mktemp("repo")
    (repo / "code.py").write_text("x = 1\n")
    for args in (
        ["init", "-q"],
        ["add", "code.py"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)
    return Setup(
        config=load_config(REPO / "configs" / "smoke.yaml"),
        samples=load_drive(drive, "training"),
        report=ChecksumReport(checked=entries),
        data_root=root,
        repo=repo,
    )


def run(setup: Setup, where: Path, **kwargs: Any) -> Outcome:
    db = where / "experiments.db"
    dirs = train.OutputDirs(results=where / "results", models=where / "models")
    with closing(connect(db)) as conn:
        create_schema(conn)
        write_images(conn, [image_record(s) for s in setup.samples])
        kwargs.setdefault("env", ENV)
        kwargs.setdefault("repo", setup.repo)
        run_id = train.run_cv(conn, setup.config, setup.samples, setup.report, dirs, **kwargs)
    return Outcome(db, dirs, run_id)


@pytest.fixture(scope="module")
def reference(setup: Setup, tmp_path_factory: pytest.TempPathFactory) -> Outcome:
    return run(setup, tmp_path_factory.mktemp("reference"))


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    trained: list[int] = []
    real = train.train_fold

    def recording(run_id: str, fold: Fold, *args: Any) -> FoldRecord:
        trained.append(fold.number)
        return real(run_id, fold, *args)

    monkeypatch.setattr(train, "train_fold", recording)
    yield trained


def rows(outcome: Outcome) -> dict[str, list[tuple[Any, ...]]]:
    with closing(sqlite3.connect(outcome.db)) as conn:
        return {
            table: conn.execute(
                f"SELECT {cols} FROM {table} WHERE run_id = ? ORDER BY {cols}",
                (outcome.run_id,),
            ).fetchall()
            for table, cols in COMPARED.items()
        }


def assert_same_results(a: Outcome, b: Outcome) -> None:
    assert rows(a) == rows(b)
    pa, pb = a.dirs.predictions(a.run_id), b.dirs.predictions(b.run_id)
    names = sorted(p.name for p in pa.iterdir())
    assert len(names) == 20
    assert names == sorted(p.name for p in pb.iterdir())
    for name in names:
        np.testing.assert_array_equal(np.load(pa / name), np.load(pb / name))


def fold_rows(outcome: Outcome) -> list[tuple[int, str, int]]:
    with closing(sqlite3.connect(outcome.db)) as conn:
        return conn.execute(
            "SELECT fold, status, attempts FROM fold_status WHERE run_id = ? ORDER BY fold",
            (outcome.run_id,),
        ).fetchall()


def test_reference_run_is_complete(reference: Outcome) -> None:
    got = rows(reference)
    assert len(got["fold_assignments"]) == 100
    assert len(got["thresholds"]) == 5
    assert len(got["per_image_metrics"]) == 20
    assert fold_rows(reference) == [(f, "complete", 1) for f in range(1, 6)]
    with closing(sqlite3.connect(reference.db)) as conn:
        finished = conn.execute("SELECT finished_at FROM runs").fetchone()[0]
    assert finished is not None
    for fold in range(1, 6):
        assert reference.dirs.checkpoint(reference.run_id, fold).is_file()


def test_interrupted_after_fold_2_resumes_at_fold_3(
    setup: Setup,
    reference: Outcome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spy: list[int],
) -> None:
    recording = train.train_fold

    def crash_on_fold_3(run_id: str, fold: Fold, *args: Any) -> FoldRecord:
        if fold.number == 3:
            raise Interrupted
        return recording(run_id, fold, *args)

    monkeypatch.setattr(train, "train_fold", crash_on_fold_3)
    with pytest.raises(Interrupted):
        run(setup, tmp_path)
    assert spy == [1, 2]

    monkeypatch.setattr(train, "train_fold", recording)
    resumed = run(setup, tmp_path)
    assert spy == [1, 2, 3, 4, 5]
    with closing(sqlite3.connect(resumed.db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone() == (1,)
    assert fold_rows(resumed) == [
        (1, "complete", 1),
        (2, "complete", 1),
        (3, "complete", 2),
        (4, "complete", 1),
        (5, "complete", 1),
    ]
    assert_same_results(reference, resumed)


def copy_of(reference: Outcome, where: Path) -> Path:
    db = where / "experiments.db"
    shutil.copy(reference.db, db)
    return db


def test_a_fold_left_running_is_retrained(
    setup: Setup, reference: Outcome, tmp_path: Path, spy: list[int]
) -> None:
    db = copy_of(reference, tmp_path)
    # A crash after start_fold but before complete_fold leaves only the
    # running status, since complete_fold writes everything at once.
    with closing(connect(db)) as conn, conn:
        conn.execute("UPDATE runs SET finished_at = NULL")
        for table in ("thresholds", "per_image_metrics", "training_history"):
            conn.execute(f"DELETE FROM {table} WHERE fold = 4")
        conn.execute(
            "UPDATE fold_status SET status = 'running', checkpoint_sha256 = NULL, "
            "completed_at = NULL WHERE fold = 4"
        )
    shutil.copytree(reference.dirs.results, tmp_path / "results")
    shutil.copytree(reference.dirs.models, tmp_path / "models")
    resumed = run(setup, tmp_path)
    assert resumed.run_id == reference.run_id
    assert spy == [4]
    assert fold_rows(resumed)[3] == (4, "complete", 2)
    assert_same_results(reference, resumed)


def unfinished_copy(reference: Outcome, where: Path) -> Path:
    db = copy_of(reference, where)
    with closing(connect(db)) as conn, conn:
        conn.execute("UPDATE runs SET finished_at = NULL")
    return db


def test_changed_fold_assignments_refuse_to_resume(
    setup: Setup, reference: Outcome, tmp_path: Path, spy: list[int]
) -> None:
    db = unfinished_copy(reference, tmp_path)
    with closing(connect(db)) as conn, conn:
        conn.execute(
            "UPDATE fold_assignments SET role = 'train' "
            "WHERE fold = 1 AND role = 'val' AND image_id = "
            "(SELECT MIN(image_id) FROM fold_assignments WHERE fold = 1 AND role = 'val')"
        )
    with pytest.raises(train.ResumeError, match="different fold assignments"):
        run(setup, tmp_path)
    assert spy == []


def test_a_different_gpu_refuses_to_resume(
    setup: Setup, reference: Outcome, tmp_path: Path, spy: list[int]
) -> None:
    unfinished_copy(reference, tmp_path)
    with pytest.raises(train.ResumeError, match="gpu_type"):
        run(setup, tmp_path, env={**ENV, "device": "gpu", "gpu_type": "Tesla T4"})
    assert spy == []


def test_new_starts_a_fresh_run(setup: Setup, reference: Outcome, tmp_path: Path) -> None:
    unfinished_copy(reference, tmp_path)
    with closing(connect(tmp_path / "experiments.db")) as conn:
        fresh = train.open_run(
            conn,
            setup.config,
            [],
            setup.report,
            train.OutputDirs(tmp_path / "results", tmp_path / "models"),
            setup.repo,
            ENV,
            new=True,
        )
        assert fresh != reference.run_id
        assert fold_statuses(conn, fresh) == {}
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone() == (2,)


def test_several_unfinished_matches_refuse_to_resume(
    setup: Setup, reference: Outcome, tmp_path: Path, spy: list[int]
) -> None:
    db = unfinished_copy(reference, tmp_path)
    with closing(connect(db)) as conn, conn:
        conn.execute(
            "INSERT INTO runs SELECT 'another', variant, config, config_hash, seed, git_commit, "
            "git_dirty, git_tag, data_hash, python_version, tensorflow_version, cuda_version, "
            "device, gpu_type, compute_platform, deterministic_ops, frozen_model_id, "
            "applied_threshold, started_at, finished_at, is_reported FROM runs"
        )
    with pytest.raises(train.ResumeError, match="several unfinished runs"):
        run(setup, tmp_path)
    assert spy == []


def test_a_dirty_tree_never_resumes(
    setup: Setup, reference: Outcome, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Code edited after an interruption must not finish a run whose row
    # records the clean commit it started on (SPEC section 17).
    unfinished_copy(reference, tmp_path)
    dirty = tmp_path / "repo"
    shutil.copytree(setup.repo, dirty)
    (dirty / "code.py").write_text("x = 2\n")
    with closing(connect(tmp_path / "experiments.db")) as conn:
        with caplog.at_level(logging.WARNING, logger="retinal_vessels.train"):
            fresh = train.open_run(
                conn,
                setup.config,
                [],
                setup.report,
                train.OutputDirs(tmp_path / "results", tmp_path / "models"),
                dirty,
                ENV,
                new=False,
            )
        assert fresh != reference.run_id
        assert conn.execute("SELECT git_dirty FROM runs WHERE run_id = ?", (fresh,)).fetchone() == (
            1,
        )
    assert "dirty" in caplog.text


def restored_copy(reference: Outcome, where: Path) -> None:
    unfinished_copy(reference, where)
    shutil.copytree(reference.dirs.results, where / "results")
    shutil.copytree(reference.dirs.models, where / "models")


def test_a_missing_prediction_refuses_to_resume(
    setup: Setup, reference: Outcome, tmp_path: Path, spy: list[int]
) -> None:
    # Restoring only the database on a new machine must not let a run finish
    # with out-of-fold predictions missing for its complete folds.
    restored_copy(reference, tmp_path)
    next(iter((tmp_path / "results" / reference.run_id / "predictions").glob("*.npy"))).unlink()
    with pytest.raises(train.ResumeError, match="prediction"):
        run(setup, tmp_path)
    assert spy == []


@pytest.mark.parametrize("change", ["delete", "alter"])
def test_a_missing_or_changed_checkpoint_refuses_to_resume(
    setup: Setup, reference: Outcome, tmp_path: Path, spy: list[int], change: str
) -> None:
    restored_copy(reference, tmp_path)
    checkpoint = tmp_path / "models" / reference.run_id / "fold_2.keras"
    if change == "delete":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"other weights")
    with pytest.raises(train.ResumeError, match="fold 2 checkpoint"):
        run(setup, tmp_path)
    assert spy == []


def test_a_fully_restored_run_resumes_with_nothing_to_train(
    setup: Setup, reference: Outcome, tmp_path: Path, spy: list[int]
) -> None:
    restored_copy(reference, tmp_path)
    assert run(setup, tmp_path).run_id == reference.run_id
    assert spy == []
