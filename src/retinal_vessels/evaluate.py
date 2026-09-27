"""Evaluate a finished cross-validation run from its saved probabilities (SPEC section 5).

Usage: ``python -m retinal_vessels.evaluate --run-id <run_id>``.

Training stores each held-out image's probabilities and its confusion-matrix
metrics. This step fills in the rest without retraining: AUC-ROC, AUC-PR,
the Brier score, and thin and thick vessel sensitivity per image, plus a
pooled reliability table in ``results/<run_id>/evaluation.json``. Each fold's
thin/thick edge comes from its own training and validation labels (D-021).

Before computing anything, it checks that the probability files match the
hashes each fold recorded, that the data matches the run's data hash, and
that each image's recomputed confusion metrics equal the stored ones.
Rerunning it gives the same values.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from retinal_vessels.config import Config
from retinal_vessels.data import Sample
from retinal_vessels.datasets.drive import LayoutError, drive_dir, load_drive
from retinal_vessels.db import (
    EvaluationRecord,
    connect,
    create_schema,
    fill_evaluation,
    fold_statuses,
    fold_thresholds,
    run_manifest,
    stored_fold_rows,
    stored_image_metrics,
)
from retinal_vessels.metrics import (
    ReliabilityTable,
    auc_roc,
    average_precision,
    binarize,
    binary_metrics,
    brier,
    reliability,
    thin_edge,
    width_sensitivity,
)
from retinal_vessels.provenance import (
    ChecksumReport,
    check_or_write_checksums,
    data_hash,
    read_checksums,
    sha256_file,
    write_manifest,
)
from retinal_vessels.train import DATASET, REPO, OutputDirs
from retinal_vessels.utils import setup_logging

logger = logging.getLogger(__name__)

EVALUATION_NAME = "evaluation.json"


class EvaluationError(RuntimeError):
    """A run that cannot be evaluated, or whose saved outputs do not match its record."""


@dataclass(frozen=True)
class FoldSummary:
    """A fold's threshold and thin/thick edge, and the skeleton pixels in each bin."""

    fold: int
    threshold: float
    thin_edge: float
    n_thin: int
    n_thick: int


@dataclass(frozen=True)
class RunEvaluation:
    """What ``evaluation.json`` holds besides the per-image rows in the database."""

    run_id: str
    evaluation: dict[str, object]
    folds: list[FoldSummary]
    reliability: ReliabilityTable


def _stored_config(conn: sqlite3.Connection, run_id: str) -> tuple[Config, str, str | None]:
    try:
        manifest = run_manifest(conn, run_id)
    except KeyError as err:
        raise EvaluationError(str(err)) from err
    try:
        config = Config.model_validate_json(json.dumps(manifest.config))
    except ValidationError as err:
        raise EvaluationError(f"run {run_id} has a config this code cannot read:\n{err}") from err
    return config, manifest.data_hash, manifest.finished_at


def _check_finished(conn: sqlite3.Connection, run_id: str, n_folds: int, finished: bool) -> None:
    statuses = fold_statuses(conn, run_id)
    complete = sorted(f for f, s in statuses.items() if s == "complete")
    if not finished or complete != list(range(1, n_folds + 1)):
        raise EvaluationError(
            f"run {run_id} is not finished ({len(complete)} of {n_folds} folds complete). "
            "Finish training before evaluating it."
        )


def _roles(conn: sqlite3.Connection, run_id: str) -> dict[int, dict[str, list[str]]]:
    roles: dict[int, dict[str, list[str]]] = {}
    for fold, image_id, role in stored_fold_rows(conn, run_id):
        roles.setdefault(fold, {"train": [], "val": [], "test": []})[role].append(image_id)
    return roles


def _check_hashes(dirs: OutputDirs, run_id: str, fold: int, test: Sequence[str]) -> None:
    path = dirs.prediction_hashes(run_id, fold)
    if not path.is_file():
        raise EvaluationError(f"fold {fold} has no prediction hashes at {path}")
    listed = read_checksums(path)
    expected = {f"{image_id}.npy" for image_id in test}
    if set(listed) != expected:
        raise EvaluationError(
            f"{path} lists {sorted(listed)}, but fold {fold} tests {sorted(expected)}"
        )
    for name, digest in sorted(listed.items()):
        file = dirs.predictions(run_id) / name
        if not file.is_file():
            raise EvaluationError(f"prediction file {file} is missing")
        if sha256_file(file) != digest:
            raise EvaluationError(f"prediction file {file} does not match {path}")


def _load_probability(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    prob: np.ndarray = np.load(path)
    if prob.dtype != np.float32 or prob.shape != shape:
        raise EvaluationError(f"{path} holds {prob.dtype} {prob.shape}, expected float32 {shape}")
    return prob


def evaluate_run(
    conn: sqlite3.Connection,
    run_id: str,
    samples: Sequence[Sample],
    report: ChecksumReport,
    dirs: OutputDirs,
) -> Path:
    """Evaluate every out-of-fold prediction of ``run_id`` and return the path of its summary.

    ``samples`` are the labeled DRIVE training images and ``report`` their
    checksum verification, which must match the run's data hash. Raises
    ``EvaluationError`` before writing anything if any check fails. Rewrites
    the run's ``manifest.json`` from the database first.
    """
    config, run_data_hash, finished_at = _stored_config(conn, run_id)
    _check_finished(conn, run_id, config.folds.n_folds, finished_at is not None)
    write_manifest(run_manifest(conn, run_id), dirs.results)
    if not report.ok or data_hash(report.checked) != run_data_hash:
        raise EvaluationError(
            f"the data does not match the data run {run_id} trained on. "
            "Evaluate against the same checksummed data."
        )

    by_id: Mapping[str, Sample] = {s.image_id: s for s in samples}
    thresholds = fold_thresholds(conn, run_id)
    stored = stored_image_metrics(conn, run_id, DATASET)
    records: list[EvaluationRecord] = []
    summaries: list[FoldSummary] = []
    pooled: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold, roles in sorted(_roles(conn, run_id).items()):
        _check_hashes(dirs, run_id, fold, roles["test"])
        tuning = [by_id[i] for i in roles["train"] + roles["val"]]
        edge = thin_edge(
            [_label(s) for s in tuning], [s.fov for s in tuning], config.evaluation.thin_quantile
        )
        threshold = thresholds[fold]
        n_thin = n_thick = 0
        for image_id in roles["test"]:
            sample = by_id[image_id]
            label = _label(sample)
            path = dirs.predictions(run_id) / f"{image_id}.npy"
            prob = _load_probability(path, label.shape)
            prediction = binarize(prob, threshold)
            if stored.get(image_id) != (fold, binary_metrics(prediction, label, sample.fov)):
                raise EvaluationError(
                    f"{path} does not reproduce the metrics stored for image {image_id} "
                    f"in fold {fold}"
                )
            width = width_sensitivity(prediction, label, sample.fov, edge)
            n_thin, n_thick = n_thin + width.n_thin, n_thick + width.n_thick
            records.append(
                EvaluationRecord(
                    dataset=DATASET,
                    image_id=image_id,
                    auc_roc=auc_roc(prob, label, sample.fov),
                    auc_pr=average_precision(prob, label, sample.fov),
                    brier=brier(prob, label, sample.fov),
                    thin_sensitivity=width.thin,
                    thick_sensitivity=width.thick,
                )
            )
            pooled.append((prob, label, sample.fov))
        summaries.append(FoldSummary(fold, threshold, edge, n_thin, n_thick))
    if len(records) != len(stored):
        raise EvaluationError(
            f"run {run_id} stores {len(stored)} image rows, but its folds test {len(records)}"
        )

    table = reliability(
        [p for p, _, _ in pooled],
        [t for _, t, _ in pooled],
        [f for _, _, f in pooled],
        config.evaluation.reliability_bins,
    )
    fill_evaluation(conn, run_id, records)
    summary = RunEvaluation(run_id, config.evaluation.model_dump(), summaries, table)
    path = dirs.results / run_id / EVALUATION_NAME
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)
    logger.info("Evaluated %d images of run %s. Summary in %s", len(records), run_id, path)
    return path


def _label(sample: Sample) -> np.ndarray:
    if sample.label is None:
        raise EvaluationError(f"image {sample.image_id} has no label")
    return sample.label


def latest_finished_run(conn: sqlite3.Connection) -> str | None:
    """Return the most recent finished run id, or None if there is none."""
    row = conn.execute("SELECT MAX(run_id) FROM runs WHERE finished_at IS NOT NULL").fetchone()
    return None if row is None or row[0] is None else str(row[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="defaults to the most recent finished run")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--checksums", type=Path, default=REPO / "data" / "CHECKSUMS.sha256")
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args(argv)
    setup_logging()

    if not args.db.is_file():
        logger.error("database not found at %s", args.db)
        return 1
    try:
        drive = drive_dir(args.data_root / "DRIVE")
        _, report = check_or_write_checksums(args.checksums, {"DRIVE": drive})
        samples = load_drive(drive, "training")
    except (LayoutError, FileNotFoundError) as err:
        logger.error("%s", err)
        return 1

    # Evaluation reads only the results directory, and never a checkpoint.
    dirs = OutputDirs(results=args.results_dir, models=REPO / "models")
    with closing(connect(args.db)) as conn:
        create_schema(conn)
        run_id = args.run_id or latest_finished_run(conn)
        if run_id is None:
            logger.error("no finished run in %s", args.db)
            return 1
        try:
            evaluate_run(conn, run_id, samples, report, dirs)
        except EvaluationError as err:
            logger.error("%s", err)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
