import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from retinal_vessels import train
from retinal_vessels.config import load_config
from retinal_vessels.datasets.drive import load_drive
from retinal_vessels.provenance import compute_checksums, format_checksums

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "configs" / "smoke.yaml"


@pytest.fixture(autouse=True)
def restore_root_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # main() installs the package's log handler, which would otherwise stay
    # on the root logger and turn later setup_logging calls into no-ops.
    monkeypatch.delenv("DRIVE_DIR", raising=False)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


def test_derived_seeds_are_stable_and_distinct() -> None:
    assert train.derived_seed(1, 2, 3) == train.derived_seed(1, 2, 3)
    seeds = {train.derived_seed(7, fold, epoch) for fold in range(1, 6) for epoch in range(1, 50)}
    assert len(seeds) == 5 * 49
    # seed + epoch would give fold 1 epoch 2 and fold 2 epoch 1 the same draws.
    assert train.derived_seed(7, 1, 2) != train.derived_seed(7, 2, 1)
    assert 0 <= train.derived_seed(7, 1) < 2**32


def test_output_dirs() -> None:
    dirs = train.OutputDirs(results=Path("r"), models=Path("m"))
    assert dirs.checkpoint("run", 3) == Path("m/run/fold_3.keras")
    assert dirs.predictions("run") == Path("r/run/predictions")


def test_prepare_preprocesses_and_rejects_unlabeled(synthetic_data_root: Path) -> None:
    config = load_config(SMOKE)
    samples = load_drive(synthetic_data_root / "DRIVE", "training")[:2]
    prepared = train.prepare(samples, config)
    assert set(prepared) == {s.image_id for s in samples}
    first = prepared[samples[0].image_id]
    assert first.image.dtype == np.float32
    assert first.image.shape == first.label.shape == first.fov.shape
    with pytest.raises(ValueError, match="no label"):
        train.prepare([replace(samples[0], label=None)], config)


def cli(root: Path, *extra: str) -> int:
    return train.main(
        [
            "--data-root",
            str(root),
            "--checksums",
            str(root / "CHECKSUMS.sha256"),
            "--db",
            str(root / "experiments.db"),
            "--results-dir",
            str(root / "results"),
            "--models-dir",
            str(root / "models"),
            *extra,
        ]
    )


def test_main_rejects_a_bad_config(synthetic_data_root: Path, tmp_path: Path) -> None:
    assert cli(synthetic_data_root, "--config", str(tmp_path / "absent.yaml")) == 1


def test_main_needs_the_checksum_file(synthetic_data_root: Path) -> None:
    assert cli(synthetic_data_root, "--config", str(SMOKE)) == 1
    assert not (synthetic_data_root / "CHECKSUMS.sha256").exists()


def test_main_refuses_data_that_fails_its_checksums(synthetic_data_root: Path) -> None:
    drive = synthetic_data_root / "DRIVE"
    (synthetic_data_root / "CHECKSUMS.sha256").write_text(
        format_checksums(compute_checksums({"DRIVE": drive}))
    )
    (drive / "training" / "images" / "21_training.tif").write_bytes(b"corrupt")
    assert cli(synthetic_data_root, "--config", str(SMOKE)) == 1
    assert not (synthetic_data_root / "experiments.db").exists()


def test_main_reports_a_run_it_cannot_resume(
    synthetic_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = synthetic_data_root / "DRIVE"
    (synthetic_data_root / "CHECKSUMS.sha256").write_text(
        format_checksums(compute_checksums({"DRIVE": drive}))
    )

    def refuse(*args: object, **kwargs: object) -> str:
        raise train.ResumeError("several unfinished runs match")

    monkeypatch.setattr(train, "run_cv", refuse)
    assert cli(synthetic_data_root, "--config", str(SMOKE)) == 1
