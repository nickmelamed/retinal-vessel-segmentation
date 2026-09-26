"""Write a small synthetic dataset with DRIVE's file names, sizes, and formats.

CI never sees real DRIVE images, so tests run on these instead. Each image is
a dark disc (the FOV) on black, with darker random line segments standing in
for vessels. Labels mark the segments inside the FOV. Like the real files,
images are RGB TIFFs and labels and masks are grayscale GIFs holding 0 and 255.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WIDTH = 565
HEIGHT = 584
FOV_RADIUS = 270
TRAIN_IDS = tuple(range(21, 41))
TEST_IDS = tuple(range(1, 21))
N_SEGMENTS = 25
MAX_VESSEL_WIDTH = 6
BACKGROUND_RGB = (150, 70, 30)
VESSEL_RGB = (90, 30, 10)
ON = 255


def training_paths(root: Path, image_id: int) -> tuple[Path, Path, Path]:
    """Return the image, label, and mask paths for a training id under ``root``."""
    base = root / "training"
    return (
        base / "images" / f"{image_id:02d}_training.tif",
        base / "1st_manual" / f"{image_id:02d}_manual1.gif",
        base / "mask" / f"{image_id:02d}_training_mask.gif",
    )


def official_test_paths(root: Path, image_id: int) -> tuple[Path, Path]:
    """Return the image and mask paths for a test id under ``root``."""
    base = root / "test"
    return (
        base / "images" / f"{image_id:02d}_test.tif",
        base / "mask" / f"{image_id:02d}_test_mask.gif",
    )


def fov_mask() -> np.ndarray:
    """Return the circular FOV as a bool array of shape (HEIGHT, WIDTH)."""
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    cy, cx = (HEIGHT - 1) / 2, (WIDTH - 1) / 2
    return np.asarray((yy - cy) ** 2 + (xx - cx) ** 2 <= FOV_RADIUS**2)


def _vessels(rng: np.random.Generator) -> np.ndarray:
    canvas = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(canvas)
    for _ in range(N_SEGMENTS):
        x0, x1 = rng.integers(0, WIDTH, size=2)
        y0, y1 = rng.integers(0, HEIGHT, size=2)
        width = int(rng.integers(1, MAX_VESSEL_WIDTH + 1))
        draw.line([(int(x0), int(y0)), (int(x1), int(y1))], fill=ON, width=width)
    return np.asarray(canvas) > 0


def _sample(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fov = fov_mask()
    vessels = _vessels(rng) & fov
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[fov] = BACKGROUND_RGB
    image[vessels] = VESSEL_RGB
    noise = rng.integers(-5, 6, size=image.shape)
    image = np.where(fov[..., None], np.clip(image + noise, 0, ON), 0).astype(np.uint8)
    return image, vessels, fov


def _save_gif(binary: np.ndarray, path: Path) -> None:
    # Pillow's default palette optimisation writes a two-colour "P" image;
    # without it the file reads back as "L" with 0 and 255, like DRIVE's.
    Image.fromarray(np.where(binary, ON, 0).astype(np.uint8)).save(path, optimize=False)


def write_synthetic_drive(
    root: Path,
    train_ids: Iterable[int] = TRAIN_IDS,
    test_ids: Iterable[int] = TEST_IDS,
    seed: int = 0,
) -> Path:
    """Write a DRIVE-shaped tree under ``root / "DRIVE"`` and return that directory.

    The same seed and ids always produce byte-identical files.
    """
    drive = root / "DRIVE"
    for sub in (
        "training/images",
        "training/1st_manual",
        "training/mask",
        "test/images",
        "test/mask",
    ):
        (drive / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for image_id in train_ids:
        image, vessels, fov = _sample(rng)
        image_path, label_path, mask_path = training_paths(drive, image_id)
        Image.fromarray(image).save(image_path)
        _save_gif(vessels, label_path)
        _save_gif(fov, mask_path)
    for image_id in test_ids:
        image, _, fov = _sample(rng)
        image_path, mask_path = official_test_paths(drive, image_id)
        Image.fromarray(image).save(image_path)
        _save_gif(fov, mask_path)
    return drive
