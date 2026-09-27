"""Load DRIVE (Staal et al., 2004) from its official directory layout.

The layout is checked file by file before anything is read (SPEC section 4),
so a partial or misplaced download fails with a list of what is wrong.
"""

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from retinal_vessels.data import Sample

logger = logging.getLogger(__name__)

DRIVE_DIR_ENV = "DRIVE_DIR"
WIDTH = 565
HEIGHT = 584
TRAIN_IDS = tuple(range(21, 41))
TEST_IDS = tuple(range(1, 21))
MASK_ON = 255

DriveSplit = Literal["training", "test"]

# From https://drive.grand-challenge.org/ (retrieved 2026-09-26), see D-015.
# Keep the published wording, since these describe images, not diagnoses.
ABNORMALITY_NOTES = {
    "25": (
        "pigment epithelium changes, probably butterfly maculopathy with pigmented "
        "scar in fovea, or choroidiopathy, no diabetic retinopathy or other vascular "
        "abnormalities."
    ),
    "26": (
        "background diabetic retinopathy, pigmentary epithelial atrophy, atrophy around optic disk"
    ),
    "32": "background diabetic retinopathy",
    "03": "background diabetic retinopathy",
    "08": (
        "pigment epithelium changes, pigmented scar in fovea, or choroidiopathy, "
        "no diabetic retinopathy or other vascular abnormalities"
    ),
    "14": "background diabetic retinopathy",
    "17": "background diabetic retinopathy",
}


class LayoutError(ValueError):
    """The DRIVE directory does not match the official file layout."""


def drive_dir(default: Path) -> Path:
    """Return ``$DRIVE_DIR`` if set, otherwise ``default``."""
    return Path(os.environ.get(DRIVE_DIR_ENV, default))


def image_id(number: int) -> str:
    """Return DRIVE's two-digit id for an image number, like ``"03"``."""
    return f"{number:02d}"


def _paths(root: Path, split: DriveSplit, number: int) -> tuple[Path, Path | None, Path]:
    i = image_id(number)
    base = root / split
    if split == "training":
        image = base / "images" / f"{i}_training.tif"
        label = base / "1st_manual" / f"{i}_manual1.gif"
        return image, label, base / "mask" / f"{i}_training_mask.gif"
    return base / "images" / f"{i}_test.tif", None, base / "mask" / f"{i}_test_mask.gif"


def _split_ids(split: DriveSplit) -> tuple[int, ...]:
    return TRAIN_IDS if split == "training" else TEST_IDS


def expected_files(root: Path) -> set[Path]:
    """Return every file path the official layout has under ``root``."""
    expected: set[Path] = set()
    for split in ("training", "test"):
        for number in _split_ids(split):
            expected.update(p for p in _paths(root, split, number) if p is not None)
    return expected


def validate_layout(root: Path) -> None:
    """Check that ``root`` holds exactly the official DRIVE files.

    Hidden files and directories like ``.DS_Store`` are ignored. Any other
    extra or missing file raises ``LayoutError`` listing every offending path.
    """
    if not root.is_dir():
        raise LayoutError(f"DRIVE directory not found at {root}")
    found = {
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    }
    expected = expected_files(root)
    problems = [f"missing: {p.relative_to(root)}" for p in sorted(expected - found)]
    problems += [f"unexpected: {p.relative_to(root)}" for p in sorted(found - expected)]
    if problems:
        raise LayoutError(f"{root} does not match the DRIVE layout:\n" + "\n".join(problems))


def _open(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != mode:
            raise LayoutError(f"{path}: expected image mode {mode}, got {im.mode}")
        if im.size != (WIDTH, HEIGHT):
            raise LayoutError(f"{path}: expected {WIDTH}x{HEIGHT} pixels, got {im.size}")
        return np.array(im)


def _read_binary(path: Path) -> np.ndarray:
    values = _open(path, "L")
    unexpected = sorted(set(np.unique(values).tolist()) - {0, MASK_ON})
    if unexpected:
        raise LayoutError(f"{path}: expected only 0 and {MASK_ON}, also found {unexpected}")
    return np.asarray(values == MASK_ON)


def load_drive(root: Path, split: DriveSplit) -> list[Sample]:
    """Validate the layout under ``root``, then load every image of ``split``.

    Test images carry no label, since DRIVE withholds them. Labels are
    returned as published. A few labeled pixels fall outside the FOV, so
    anything that scores or counts vessels must mask by the FOV first.
    """
    if split not in ("training", "test"):
        raise ValueError(f"DRIVE split must be 'training' or 'test', got {split!r}")
    validate_layout(root)
    samples = []
    for number in _split_ids(split):
        image_path, label_path, mask_path = _paths(root, split, number)
        i = image_id(number)
        samples.append(
            Sample(
                dataset="drive",
                image_id=i,
                split=split,
                image=_open(image_path, "RGB"),
                fov=_read_binary(mask_path),
                label=None if label_path is None else _read_binary(label_path),
                abnormality_note=ABNORMALITY_NOTES.get(i),
            )
        )
    logger.info("Loaded %d DRIVE %s images from %s", len(samples), split, root)
    return samples
