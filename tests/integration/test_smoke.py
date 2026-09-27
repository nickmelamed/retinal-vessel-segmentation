"""End-to-end run on synthetic data.

Write synthetic data, record and verify its checksums through the real
``check_data.py`` CLI, which also writes the ``images`` rows. Then build the
folds, check them with the leakage audit, sample a batch of patches, and
write a run manifest. Finally, train the full smoke cross-validation through
the ``train`` CLI, as ``make train`` does.
"""

import json
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from retinal_vessels.config import load_config
from retinal_vessels.data import make_folds
from retinal_vessels.datasets.drive import load_drive
from retinal_vessels.db import (
    SCHEMA_VERSION,
    connect,
    create_schema,
    run_query,
    schema_version,
    write_fold_assignments,
)
from retinal_vessels.patches import make_patch_dataset
from retinal_vessels.preprocess import preprocess
from retinal_vessels.provenance import (
    build_manifest,
    new_run_id,
    read_checksums,
    sha256_file,
    verify_checksums,
    write_manifest,
)
from tests.fixtures.database import RUN, insert

REPO = Path(__file__).resolve().parents[2]
CHECK_DATA = REPO / "scripts" / "check_data.py"

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def _check_data(
    data_root: Path, drive_dir: Path | None = None, init: bool = False
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "DRIVE_DIR"}
    if drive_dir is not None:
        env["DRIVE_DIR"] = str(drive_dir)
    args = [
        "--data-root",
        str(data_root),
        "--checksums",
        str(data_root / "CHECKSUMS.sha256"),
        "--db",
        str(data_root.parent / "results" / "experiments.db"),
    ]
    return subprocess.run(
        [sys.executable, str(CHECK_DATA), *args, *(["--init"] if init else [])],
        capture_output=True,
        text=True,
        env=env,
    )


def test_pipeline_is_wired_end_to_end(synthetic_data_root: Path, tmp_path: Path) -> None:
    manifest_path = synthetic_data_root / "CHECKSUMS.sha256"

    missing = _check_data(synthetic_data_root)
    assert missing.returncode == 1
    assert "--init" in missing.stderr
    assert not manifest_path.exists()

    first = _check_data(synthetic_data_root, init=True)
    assert first.returncode == 0, first.stderr
    assert "Wrote 100 files" in first.stderr
    written = manifest_path.read_bytes()

    second = _check_data(synthetic_data_root)
    assert second.returncode == 0, second.stderr
    assert "Verified 100 files" in second.stderr
    assert manifest_path.read_bytes() == written

    moved = tmp_path / "elsewhere" / "DRIVE"
    moved.parent.mkdir()
    (synthetic_data_root / "DRIVE").rename(moved)
    assert _check_data(synthetic_data_root).returncode == 1
    assert _check_data(synthetic_data_root, drive_dir=moved).returncode == 0
    verified = verify_checksums(read_checksums(manifest_path), {"DRIVE": moved})
    assert verified.ok

    (moved / "test" / "images" / "05_test.tif").write_bytes(b"corrupt")
    corrupt = _check_data(synthetic_data_root, drive_dir=moved)
    assert corrupt.returncode == 1
    assert "checksum mismatch: DRIVE/test/images/05_test.tif" in corrupt.stderr
    assert manifest_path.read_bytes() == written
    corrupted = verify_checksums(read_checksums(manifest_path), {"DRIVE": moved})

    config = load_config(REPO / "configs" / "smoke.yaml")
    samples = load_drive(moved, "training")
    abnormal = {s.image_id for s in samples if s.has_abnormality}
    assert abnormal == {"25", "26", "32"}
    folds = make_folds(
        [s.image_id for s in samples],
        abnormal,
        n_folds=config.folds.n_folds,
        n_val=config.folds.n_val,
        seed=config.folds.seed,
    )
    with closing(connect(tmp_path / "results" / "experiments.db")) as conn:
        create_schema(conn)
        assert schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone() == (40,)
        insert(conn, "runs", RUN)
        write_fold_assignments(conn, RUN["run_id"], "drive", folds)
        assert run_query(conn, "04_leakage_audit") == []

    train = [s for s in samples if s.image_id in folds[0].train]
    images = np.stack([preprocess(s.image, s.fov, config.preprocess) for s in train])
    labels = np.stack([s.label for s in train if s.label is not None])
    fovs = np.stack([s.fov for s in train])
    x, y, w = next(iter(make_patch_dataset(images, labels, fovs, config.patches, config.seed)))
    size = config.patches.size
    assert x.shape == y.shape == w.shape == (config.patches.batch_size, size, size, 1)

    run = dict(
        variant=config.variant,
        config=config.as_dict(),
        seed=config.seed,
        deterministic_ops=False,
        repo=REPO,
    )
    with pytest.raises(ValueError, match="failed verification"):
        build_manifest(run_id=new_run_id(), data=corrupted, **run)  # type: ignore[arg-type]
    manifest = build_manifest(run_id=new_run_id(), data=verified, **run)  # type: ignore[arg-type]
    saved = json.loads(write_manifest(manifest, tmp_path / "results").read_text())
    assert saved["variant"] == "smoke"
    assert len(saved["git_commit"]) == 40
    assert saved["environment"]["tensorflow_version"]


def test_training_cli_runs_cross_validation(synthetic_data_root: Path, tmp_path: Path) -> None:
    assert _check_data(synthetic_data_root, init=True).returncode == 0
    results, models = tmp_path / "results", tmp_path / "models"
    db = synthetic_data_root.parent / "results" / "experiments.db"
    env = {k: v for k, v in os.environ.items() if k != "DRIVE_DIR"}
    args = [
        "--config",
        str(REPO / "configs" / "smoke.yaml"),
        "--data-root",
        str(synthetic_data_root),
        "--checksums",
        str(synthetic_data_root / "CHECKSUMS.sha256"),
        "--db",
        str(db),
        "--results-dir",
        str(results),
        "--models-dir",
        str(models),
    ]
    trained = subprocess.run(
        [sys.executable, "-m", "retinal_vessels.train", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    assert trained.returncode == 0, trained.stderr

    with closing(connect(db)) as conn:
        (run_id,) = conn.execute("SELECT run_id FROM runs").fetchone()
        assert f"Run {run_id} finished" in trained.stderr
        statuses = conn.execute("SELECT status FROM fold_status ORDER BY fold").fetchall()
        assert statuses == [("complete",)] * 5
        assert conn.execute("SELECT COUNT(*) FROM thresholds").fetchone() == (5,)
        oof = conn.execute(
            "SELECT COUNT(DISTINCT image_id) FROM per_image_metrics WHERE dice IS NOT NULL"
        ).fetchone()
        assert oof == (20,)
        assert run_query(conn, "04_leakage_audit") == []
        shas = dict(conn.execute("SELECT fold, checkpoint_sha256 FROM fold_status").fetchall())

    for fold, sha in shas.items():
        assert sha256_file(models / run_id / f"fold_{fold}.keras") == sha
    assert len(list((results / run_id / "predictions").glob("*.npy"))) == 20
    manifest = json.loads((results / run_id / "manifest.json").read_text())
    assert manifest["finished_at"] is not None
    assert manifest["config"]["variant"] == "smoke"

    again = subprocess.run(
        [sys.executable, "-m", "retinal_vessels.train", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    assert again.returncode == 0, again.stderr
    assert "Started run" in again.stderr
