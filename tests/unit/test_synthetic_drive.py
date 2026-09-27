from pathlib import Path

import numpy as np
from PIL import Image

from tests.fixtures.synthetic_drive import (
    HEIGHT,
    TEST_IDS,
    TRAIN_IDS,
    WIDTH,
    fov_mask,
    official_test_paths,
    training_paths,
    write_synthetic_drive,
)


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im)


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_layout_matches_drive_names(synthetic_data_root: Path) -> None:
    drive = synthetic_data_root / "DRIVE"
    expected = {p for i in TRAIN_IDS for p in training_paths(drive, i)}
    expected |= {p for i in TEST_IDS for p in official_test_paths(drive, i)}
    actual = {p for p in drive.rglob("*") if p.is_file()}
    assert actual == expected
    assert len(actual) == 100


def test_formats_match_drive(synthetic_data_root: Path) -> None:
    image_path, label_path, mask_path = training_paths(synthetic_data_root / "DRIVE", 21)
    with Image.open(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (WIDTH, HEIGHT)
    for path in (label_path, mask_path):
        with Image.open(path) as gif:
            assert gif.mode == "L"
            assert gif.size == (WIDTH, HEIGHT)
            assert set(np.unique(np.asarray(gif))) == {0, 255}


def test_labels_lie_inside_the_fov(synthetic_data_root: Path) -> None:
    drive = synthetic_data_root / "DRIVE"
    for image_id in TRAIN_IDS:
        _, label_path, mask_path = training_paths(drive, image_id)
        label = _read(label_path) > 0
        mask = _read(mask_path) > 0
        assert label.any()
        assert not (label & ~mask).any()
        assert np.array_equal(mask, fov_mask())


def test_image_is_black_outside_the_fov(synthetic_data_root: Path) -> None:
    image_path, _ = official_test_paths(synthetic_data_root / "DRIVE", 1)
    image = _read(image_path)
    assert not image[~fov_mask()].any()


def test_same_seed_gives_identical_files(tmp_path: Path) -> None:
    a = write_synthetic_drive(tmp_path / "a", train_ids=[21], test_ids=[1], seed=3)
    b = write_synthetic_drive(tmp_path / "b", train_ids=[21], test_ids=[1], seed=3)
    c = write_synthetic_drive(tmp_path / "c", train_ids=[21], test_ids=[1], seed=4)
    assert _files(a) == _files(b)
    assert _files(a) != _files(c)
