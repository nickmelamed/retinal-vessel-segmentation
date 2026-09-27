"""Run full 5-fold cross-validation for one config (SPEC sections 5, 7, and 16).

Usage: ``python -m retinal_vessels.train --config configs/baseline.yaml``.

Each fold trains on its training images, stops early on whole-image Dice of
its validation images, and chooses its threshold on those same images. Only
then does the best checkpoint predict the fold's held-out test images, whose
probabilities are saved under ``results/<run_id>/predictions/`` and whose
metrics go to the database.

The run is resumable. Rerunning the same command continues the unfinished
run with the same variant, config, commit, and data: complete folds are
skipped and a fold left running is retrained from scratch. Pass ``--new`` to
start a fresh run instead.
"""

import argparse
import logging
import os
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retinal_vessels.config import Config, ConfigError, load_config
from retinal_vessels.data import Fold, Sample, fold_rows, make_folds
from retinal_vessels.datasets.drive import LayoutError, drive_dir, load_drive
from retinal_vessels.db import (
    EpochRecord,
    FoldRecord,
    ImageMetrics,
    complete_fold,
    connect,
    create_schema,
    find_resumable_run,
    finish_run,
    fold_statuses,
    image_record,
    insert_run,
    run_manifest,
    start_fold,
    stored_fold_rows,
    write_fold_assignments,
    write_images,
)
from retinal_vessels.losses import make_loss, pack_target
from retinal_vessels.metrics import (
    best_threshold,
    binary_metrics,
    selection_rule,
    threshold_grid,
)
from retinal_vessels.model import build_unet
from retinal_vessels.patches import make_patch_dataset
from retinal_vessels.predict import predict_image
from retinal_vessels.preprocess import preprocess
from retinal_vessels.provenance import (
    ChecksumReport,
    build_manifest,
    check_or_write_checksums,
    environment,
    new_run_id,
    sha256_file,
    utc_timestamp,
    write_manifest,
)
from retinal_vessels.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DATASET = "drive"


class ResumeError(RuntimeError):
    """An unfinished run matches this command but cannot be continued safely."""


@dataclass(frozen=True)
class Prepared:
    """One labeled image after preprocessing: (H, W) float32 image, bool label and FOV."""

    image: np.ndarray
    label: np.ndarray
    fov: np.ndarray


@dataclass(frozen=True)
class OutputDirs:
    """Where a run writes its manifest and predictions, and its checkpoints."""

    results: Path
    models: Path

    def checkpoint(self, run_id: str, fold: int) -> Path:
        return self.models / run_id / f"fold_{fold}.keras"

    def predictions(self, run_id: str) -> Path:
        return self.results / run_id / "predictions"


def derived_seed(*entropy: int) -> int:
    """Return a 32-bit seed mixed from ``entropy``, such as (seed, fold, epoch).

    Seeds for different folds and epochs never collide the way ``seed + epoch``
    would, and each depends only on its own inputs, so a resumed fold draws
    exactly what an uninterrupted one would.
    """
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def prepare(samples: Sequence[Sample], config: Config) -> dict[str, Prepared]:
    """Preprocess every labeled sample once, keyed by image id."""
    prepared = {}
    for s in samples:
        if s.label is None:
            raise ValueError(f"image {s.image_id} has no label and cannot be used for training")
        prepared[s.image_id] = Prepared(
            preprocess(s.image, s.fov, config.preprocess), s.label, s.fov
        )
    return prepared


def _stack(data: Mapping[str, Prepared], ids: Sequence[str]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.stack([getattr(data[i], name) for i in ids]) for name in ("image", "label", "fov")
    )


def train_fold(
    run_id: str, fold: Fold, data: Mapping[str, Prepared], config: Config, dirs: OutputDirs
) -> FoldRecord:
    """Train one fold, predict its test images, and return what the fold writes.

    Saves the best checkpoint by validation Dice and the test-image
    probabilities to disk. The database is left to the caller, so a crash
    here leaves the fold ``running``.
    """
    import keras

    set_seed(derived_seed(config.seed, fold.number), config.training.deterministic_ops)
    model = build_unet(config.model)
    model.compile(
        optimizer=keras.optimizers.Adam(config.training.learning_rate),
        loss=make_loss(config.loss),
    )
    images, labels, fovs = _stack(data, fold.train)
    val = [data[i] for i in fold.val]
    grid = threshold_grid(config.threshold.divisions)
    checkpoint = dirs.checkpoint(run_id, fold.number)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_name(f"{checkpoint.stem}.partial.keras")

    history: list[EpochRecord] = []
    best_dice, best_threshold_value, since_best = -1.0, 0.0, 0
    for epoch in range(1, config.training.max_epochs + 1):
        seed = derived_seed(config.seed, fold.number, epoch)
        patches = make_patch_dataset(images, labels, fovs, config.patches, seed).map(pack_target)
        loss = float(model.fit(patches, epochs=1, verbose=0, shuffle=False).history["loss"][0])
        probs = [predict_image(model, v.image, v.fov, config.inference) for v in val]
        threshold, dice = best_threshold(probs, [v.label for v in val], [v.fov for v in val], grid)
        history.append(EpochRecord(epoch=epoch, train_loss=loss, val_dice=dice))
        logger.info(
            "Fold %d epoch %d: train loss %.4f, val Dice %.4f at threshold %.2f",
            fold.number,
            epoch,
            loss,
            dice,
            threshold,
        )
        if dice > best_dice:
            best_dice, best_threshold_value, since_best = dice, threshold, 0
            model.save(partial)
            os.replace(partial, checkpoint)
        else:
            since_best += 1
            if since_best >= config.training.patience:
                logger.info("Fold %d stopped early after epoch %d", fold.number, epoch)
                break

    best = keras.models.load_model(checkpoint, compile=False)
    out = dirs.predictions(run_id)
    out.mkdir(parents=True, exist_ok=True)
    test_metrics = []
    for image_id in fold.test:
        item = data[image_id]
        prob = predict_image(best, item.image, item.fov, config.inference)
        np.save(out / f"{image_id}.npy", prob)
        metrics = binary_metrics(prob >= best_threshold_value, item.label, item.fov)
        test_metrics.append(ImageMetrics(DATASET, image_id, metrics))
    return FoldRecord(
        run_id=run_id,
        fold=fold.number,
        checkpoint_sha256=sha256_file(checkpoint),
        threshold=best_threshold_value,
        selection_rule=selection_rule(config.threshold.divisions),
        val_dice=best_dice,
        history=history,
        test_metrics=test_metrics,
    )


def _check_resumable(
    conn: sqlite3.Connection, run_id: str, folds: Sequence[Fold], env: Mapping[str, Any]
) -> None:
    stored = stored_fold_rows(conn, run_id)
    computed = sorted(fold_rows(folds))
    if stored != computed:
        raise ResumeError(
            f"run {run_id} has different fold assignments from the ones computed now. "
            "Pass --new to start a fresh run."
        )
    before = run_manifest(conn, run_id).environment
    changed = {k: (before[k], env.get(k)) for k in before if before[k] != env.get(k)}
    if changed:
        # Reported comparisons need one GPU type and runtime (SPEC section 16).
        raise ResumeError(
            f"run {run_id} started on a different environment, (then, now): {changed}. "
            "Pass --new to start a fresh run here."
        )


def open_run(
    conn: sqlite3.Connection,
    config: Config,
    folds: Sequence[Fold],
    report: ChecksumReport,
    dirs: OutputDirs,
    repo: Path,
    env: Mapping[str, Any],
    new: bool,
) -> str:
    """Return the id of the run to continue, creating and recording one if needed."""
    manifest = build_manifest(
        run_id=new_run_id(),
        variant=config.variant,
        config=config.as_dict(),
        seed=config.seed,
        deterministic_ops=config.training.deterministic_ops,
        repo=repo,
        data=report,
        env=env,
    )
    if not new:
        try:
            found = find_resumable_run(
                conn,
                manifest.variant,
                manifest.config_hash,
                manifest.git_commit,
                manifest.data_hash,
            )
        except RuntimeError as err:
            raise ResumeError(str(err)) from err
        if found is not None:
            _check_resumable(conn, found, folds, manifest.environment)
            done = sorted(f for f, s in fold_statuses(conn, found).items() if s == "complete")
            logger.info("Resuming run %s, complete folds: %s", found, done or "none")
            return found
    insert_run(conn, manifest)
    write_fold_assignments(conn, manifest.run_id, DATASET, folds)
    write_manifest(manifest, dirs.results)
    logger.info("Started run %s", manifest.run_id)
    return manifest.run_id


def run_cv(
    conn: sqlite3.Connection,
    config: Config,
    samples: Sequence[Sample],
    report: ChecksumReport,
    dirs: OutputDirs,
    *,
    repo: Path = REPO,
    env: Mapping[str, Any] | None = None,
    new: bool = False,
) -> str:
    """Train every fold that is not yet complete and return the run id.

    ``samples`` are the labeled DRIVE training images and ``report`` their
    passing checksum verification. ``env`` defaults to the current machine.
    """
    abnormal = {s.image_id for s in samples if s.has_abnormality}
    folds = make_folds(
        [s.image_id for s in samples],
        abnormal,
        n_folds=config.folds.n_folds,
        n_val=config.folds.n_val,
        seed=config.folds.seed,
    )
    run_id = open_run(
        conn, config, folds, report, dirs, repo, env if env is not None else environment(), new
    )
    data = prepare(samples, config)
    for fold in folds:
        if fold_statuses(conn, run_id).get(fold.number) == "complete":
            logger.info("Fold %d of run %s is complete, skipping", fold.number, run_id)
            continue
        attempt = start_fold(conn, run_id, fold.number, utc_timestamp())
        logger.info("Fold %d of run %s, attempt %d", fold.number, run_id, attempt)
        record = train_fold(run_id, fold, data, config, dirs)
        complete_fold(conn, record, utc_timestamp())
        logger.info(
            "Fold %d complete: threshold %.2f, val Dice %.4f. Checkpoint %s",
            fold.number,
            record.threshold,
            record.val_dice,
            dirs.checkpoint(run_id, fold.number),
        )
    finish_run(conn, run_id, utc_timestamp())
    manifest = run_manifest(conn, run_id)
    write_manifest(manifest, dirs.results)
    logger.info("Run %s finished. Predictions in %s", run_id, dirs.predictions(run_id))
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--checksums", type=Path, default=REPO / "data" / "CHECKSUMS.sha256")
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--new", action="store_true", help="start a fresh run, never resume")
    args = parser.parse_args(argv)
    setup_logging()

    try:
        config = load_config(args.config)
        drive = drive_dir(args.data_root / "DRIVE")
        _, report = check_or_write_checksums(args.checksums, {"DRIVE": drive})
        if not report.ok:
            for problem in report.problems():
                logger.error(problem)
            logger.error("Data does not match %s, not training", args.checksums)
            return 1
        samples = load_drive(drive, "training")
    except (ConfigError, LayoutError, FileNotFoundError) as err:
        logger.error("%s", err)
        return 1

    dirs = OutputDirs(results=args.results_dir, models=args.models_dir)
    with closing(connect(args.db)) as conn:
        create_schema(conn)
        write_images(conn, [image_record(s) for s in samples])
        try:
            run_id = run_cv(conn, config, samples, report, dirs, new=args.new)
        except ResumeError as err:
            logger.error("%s", err)
            return 1
    logger.info("Run %s recorded in %s", run_id, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
