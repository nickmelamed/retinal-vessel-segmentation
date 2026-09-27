from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from retinal_vessels.datasets.drive import (
    ABNORMALITY_NOTES,
    DRIVE_DIR_ENV,
    LayoutError,
    drive_dir,
    expected_files,
    load_drive,
    validate_layout,
)
from tests.fixtures.synthetic_drive import fov_mask, official_test_paths, training_paths

# The official wording (D-015). Written out again here, not imported, so a
# change to the loader's strings fails this test.
NOTES = {
    "25": "pigment epithelium changes, probably butterfly maculopathy with pigmented scar in "
    "fovea, or choroidiopathy, no diabetic retinopathy or other vascular abnormalities.",
    "26": "background diabetic retinopathy, pigmentary epithelial atrophy, atrophy around "
    "optic disk",
    "32": "background diabetic retinopathy",
    "03": "background diabetic retinopathy",
    "08": "pigment epithelium changes, pigmented scar in fovea, or choroidiopathy, no diabetic "
    "retinopathy or other vascular abnormalities",
    "14": "background diabetic retinopathy",
    "17": "background diabetic retinopathy",
}


@pytest.fixture
def drive(synthetic_data_root: Path) -> Path:
    return synthetic_data_root / "DRIVE"


def test_expected_files_match_the_synthetic_tree(drive: Path) -> None:
    on_disk = {p for p in drive.rglob("*") if p.is_file()}
    assert expected_files(drive) == on_disk
    assert len(on_disk) == 100


def test_loads_training_images_with_labels(drive: Path) -> None:
    samples = load_drive(drive, "training")
    assert [s.image_id for s in samples] == [str(i) for i in range(21, 41)]
    first = samples[0]
    assert first.image.shape == (584, 565, 3)
    assert first.image.dtype == np.uint8
    assert first.label is not None and first.label.dtype == np.bool_
    np.testing.assert_array_equal(first.fov, fov_mask())
    assert all(s.split == "training" and s.dataset == "drive" for s in samples)


def test_test_images_have_no_labels(drive: Path) -> None:
    samples = load_drive(drive, "test")
    assert [s.image_id for s in samples] == [f"{i:02d}" for i in range(1, 21)]
    assert all(s.label is None and s.split == "test" for s in samples)


def test_pathology_metadata_for_25_26_32(drive: Path) -> None:
    samples = {s.image_id: s for s in load_drive(drive, "training")}
    for image_id in ("25", "26", "32"):
        assert samples[image_id].has_abnormality
        assert samples[image_id].abnormality_note == NOTES[image_id]
    normal = [s for i, s in samples.items() if i not in {"25", "26", "32"}]
    assert len(normal) == 17
    assert all(s.abnormality_note is None for s in normal)


def test_pathology_metadata_for_test_images(drive: Path) -> None:
    samples = {s.image_id: s for s in load_drive(drive, "test")}
    noted = {i: s.abnormality_note for i, s in samples.items() if s.has_abnormality}
    assert noted == {i: NOTES[i] for i in ("03", "08", "14", "17")}


def test_loader_notes_are_exactly_the_official_ones() -> None:
    assert ABNORMALITY_NOTES == NOTES


def test_missing_file_is_named(drive: Path) -> None:
    _, label, _ = training_paths(drive, 30)
    label.unlink()
    with pytest.raises(LayoutError, match="missing: training/1st_manual/30_manual1.gif"):
        load_drive(drive, "training")


def test_extra_file_is_named(drive: Path) -> None:
    (drive / "test" / "images" / "21_test.tif").write_bytes(b"x")
    with pytest.raises(LayoutError, match="unexpected: test/images/21_test.tif"):
        validate_layout(drive)


def test_every_problem_is_listed(drive: Path) -> None:
    official_test_paths(drive, 5)[1].unlink()
    (drive / "notes.txt").write_text("x")
    with pytest.raises(LayoutError) as err:
        validate_layout(drive)
    assert "missing: test/mask/05_test_mask.gif" in str(err.value)
    assert "unexpected: notes.txt" in str(err.value)


def test_hidden_files_are_ignored(drive: Path) -> None:
    (drive / ".DS_Store").write_bytes(b"x")
    (drive / "training" / ".cache").mkdir()
    (drive / "training" / ".cache" / "thumb.png").write_bytes(b"x")
    validate_layout(drive)


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(LayoutError, match="not found"):
        validate_layout(tmp_path / "DRIVE")


def test_wrong_size_raises(drive: Path) -> None:
    image, _, _ = training_paths(drive, 21)
    Image.new("RGB", (564, 584)).save(image)
    with pytest.raises(LayoutError, match=r"21_training.tif: expected 565x584"):
        load_drive(drive, "training")


def test_wrong_image_mode_raises(drive: Path) -> None:
    image, _ = official_test_paths(drive, 1)
    Image.new("L", (565, 584)).save(image)
    with pytest.raises(LayoutError, match="expected image mode RGB, got L"):
        load_drive(drive, "test")


def test_palette_gif_raises(drive: Path) -> None:
    _, _, mask = training_paths(drive, 22)
    # A gray palette reads back as "L", so use colors to keep the file in "P".
    palette = Image.fromarray(fov_mask().astype(np.uint8), "P")
    palette.putpalette([0, 0, 0, 255, 0, 0])
    palette.save(mask)
    with pytest.raises(LayoutError, match="expected image mode L, got P"):
        load_drive(drive, "training")


def test_non_binary_label_raises(drive: Path) -> None:
    _, label, _ = training_paths(drive, 23)
    values = np.where(fov_mask(), 255, 0).astype(np.uint8)
    values[300, 300] = 128
    Image.fromarray(values).save(label, optimize=False)
    with pytest.raises(LayoutError, match=r"also found \[128\]"):
        load_drive(drive, "training")


def test_empty_fov_raises(drive: Path) -> None:
    _, _, mask = training_paths(drive, 24)
    Image.fromarray(np.zeros((584, 565), dtype=np.uint8)).save(mask, optimize=False)
    with pytest.raises(ValueError, match="drive image '24': FOV mask is empty"):
        load_drive(drive, "training")


def test_unknown_split_raises(drive: Path) -> None:
    with pytest.raises(ValueError, match="'training' or 'test'"):
        load_drive(drive, "validation")  # type: ignore[arg-type]


def test_drive_dir_honours_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DRIVE_DIR_ENV, raising=False)
    assert drive_dir(tmp_path / "default") == tmp_path / "default"
    monkeypatch.setenv(DRIVE_DIR_ENV, str(tmp_path / "moved"))
    assert drive_dir(tmp_path / "default") == tmp_path / "moved"
