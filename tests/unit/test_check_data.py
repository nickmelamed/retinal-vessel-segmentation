import importlib.util
import logging
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from retinal_vessels.db import connect
from tests.fixtures.synthetic_drive import fov_mask, training_paths

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_data.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_data = load_script()


def run(root: Path, *extra: str) -> int:
    args = ["--data-root", str(root), "--checksums", str(root / "CHECKSUMS.sha256")]
    code: int = check_data.main([*args, "--db", str(root / "experiments.db"), *extra])
    return code


@pytest.fixture(autouse=True)
def no_drive_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRIVE_DIR", raising=False)


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    # main() installs the package's log handler, which would otherwise stay
    # on the root logger and turn later setup_logging calls into no-ops.
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


def images(root: Path) -> list[tuple[str, int, float | None]]:
    with closing(connect(root / "experiments.db")) as conn:
        rows: list[tuple[str, int, float | None]] = conn.execute(
            "SELECT image_id, has_labels, vessel_fraction_in_fov FROM images ORDER BY image_id"
        ).fetchall()
    return rows


def test_writes_an_images_row_for_every_image(
    synthetic_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    assert run(synthetic_data_root, "--init") == 0
    rows = images(synthetic_data_root)
    assert len(rows) == 40
    assert all(has_labels == 0 and f is None for i, has_labels, f in rows if int(i) <= 20)
    assert all(has_labels == 1 and f is not None for i, has_labels, f in rows if int(i) > 20)
    assert "40 rows added, 0 already present" in caplog.text


def test_logs_the_pooled_fraction_as_computed(
    synthetic_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    run(synthetic_data_root, "--init")
    fov = fov_mask()
    vessel = total = 0
    for number in range(21, 41):
        _, label_path, _ = training_paths(synthetic_data_root / "DRIVE", number)
        with Image.open(label_path) as im:
            label = np.asarray(im) > 0
        vessel += int((label & fov).sum())
        total += int(fov.sum())
    expected = f"within-FOV vessel fraction {vessel / total:.4f} pooled over 20 images"
    assert expected in caplog.text
    assert f"({vessel} of {total} FOV pixels)" in caplog.text
    assert "abnormality notes on 25, 26, 32" in caplog.text
    assert "abnormality notes on 03, 08, 14, 17" in caplog.text


def test_rerunning_adds_nothing(
    synthetic_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    run(synthetic_data_root, "--init")
    first = images(synthetic_data_root)
    assert run(synthetic_data_root) == 0
    assert images(synthetic_data_root) == first
    assert "0 rows added, 40 already present" in caplog.text


def test_stored_rows_that_disagree_with_the_data_fail(
    synthetic_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run(synthetic_data_root, "--init")
    with closing(connect(synthetic_data_root / "experiments.db")) as conn, conn:
        conn.execute("UPDATE images SET fov_pixels = 1 WHERE image_id = '21'")
    assert run(synthetic_data_root) == 1
    assert "drive 21 differs from the data" in caplog.text


def test_files_that_pass_checksums_but_not_the_loader_fail(
    synthetic_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _, _, mask = training_paths(synthetic_data_root / "DRIVE", 22)
    Image.new("L", (100, 100)).save(mask, optimize=False)
    assert run(synthetic_data_root, "--init") == 1
    assert "expected 565x584 pixels" in caplog.text
    assert (
        not (synthetic_data_root / "experiments.db").exists() or images(synthetic_data_root) == []
    )


def test_checksum_failures_stop_before_the_database(
    synthetic_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run(synthetic_data_root, "--init")
    (synthetic_data_root / "experiments.db").unlink()
    (synthetic_data_root / "DRIVE" / "test" / "images" / "05_test.tif").write_bytes(b"x")
    assert run(synthetic_data_root) == 1
    assert "checksum mismatch" in caplog.text
    assert not (synthetic_data_root / "experiments.db").exists()


def test_missing_drive_directory_fails(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert run(tmp_path) == 1
    assert "DRIVE directory not found" in caplog.text
