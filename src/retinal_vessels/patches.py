"""Sample augmented training patches with ``tf.data`` (SPEC section 7).

Every random choice is drawn up front from one NumPy generator seeded by
the caller, so the same seed gives the same patches in the same order even
with parallel ``tf.data`` maps.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from retinal_vessels.config import PatchesConfig

IMAGE, LABEL, FOV = 0, 1, 2
N_ROTATIONS = 4
# A flipped fundus is as plausible as the original, so flip half the time.
FLIP_PROBABILITY = 0.5


def sample_centers(fovs: Sequence[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` patch centers as an (n, 3) int64 array of (image, row, col).

    Each center picks an image uniformly, then a pixel uniformly from that
    image's FOV, so every center lies inside a FOV.
    """
    if n < 1:
        raise ValueError(f"need at least one center, got n={n}")
    if not fovs:
        raise ValueError("no FOV masks to sample from")
    coords = [np.argwhere(f) for f in fovs]
    for index, c in enumerate(coords):
        if len(c) == 0:
            raise ValueError(f"FOV mask {index} is empty")
    images = rng.integers(0, len(fovs), size=n)
    picks = np.array([rng.integers(0, len(coords[i])) for i in images])
    rows_cols = np.array([coords[i][p] for i, p in zip(images, picks, strict=True)])
    return np.column_stack([images, rows_cols]).astype(np.int64)


@dataclass(frozen=True)
class Augmentation:
    """Per-patch augmentation parameters, one entry per patch."""

    flip_rows: np.ndarray
    flip_cols: np.ndarray
    rotations: np.ndarray
    brightness: np.ndarray
    contrast: np.ndarray


def sample_augmentation(cfg: PatchesConfig, n: int, rng: np.random.Generator) -> Augmentation:
    """Draw augmentation parameters for ``n`` patches.

    Steps switched off in config get their identity value, so every patch
    goes through the same ops.
    """
    low, high = cfg.contrast_range
    return Augmentation(
        flip_rows=rng.random(n) < FLIP_PROBABILITY if cfg.flip else np.zeros(n, dtype=bool),
        flip_cols=rng.random(n) < FLIP_PROBABILITY if cfg.flip else np.zeros(n, dtype=bool),
        rotations=rng.integers(0, N_ROTATIONS, size=n) if cfg.rot90 else np.zeros(n, dtype=int),
        brightness=rng.uniform(-cfg.brightness_delta, cfg.brightness_delta, size=n),
        contrast=rng.uniform(low, high, size=n),
    )


def _validate(images: np.ndarray, labels: np.ndarray, fovs: np.ndarray) -> None:
    if images.ndim != 3:
        raise ValueError(f"images must be (N, H, W), got shape {images.shape}")
    if images.dtype != np.float32:
        raise ValueError(f"images must be float32, got {images.dtype}")
    for name, array in (("labels", labels), ("fovs", fovs)):
        if array.shape != images.shape:
            raise ValueError(f"{name} shape {array.shape} does not match images {images.shape}")
        if array.dtype != np.bool_:
            raise ValueError(f"{name} must be bool, got {array.dtype}")
    if len(images) == 0:
        raise ValueError("no images to sample patches from")


def make_patch_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    fovs: np.ndarray,
    cfg: PatchesConfig,
    seed: int,
) -> Any:
    """Return a batched ``tf.data.Dataset`` of ``cfg.per_epoch`` augmented patches.

    ``images`` is (N, H, W) float32 from ``preprocess``, and ``labels`` and
    ``fovs`` are (N, H, W) bool. Each batch is ``(x, y, w)``, all
    (batch, size, size, 1) float32. ``y`` is the vessel label and
    ``w`` is the FOV, which can serve as a per-pixel loss weight. Patches
    reaching past the image edge are padded with 0. Flips and rotations move
    all three together. Brightness and contrast jitter touch ``x`` only, and
    only inside the FOV. Call it with a new seed for each epoch.
    """
    import tensorflow as tf

    _validate(images, labels, fovs)
    size = cfg.size
    half = size // 2
    rng = np.random.default_rng(seed)
    centers = sample_centers(list(fovs), cfg.per_epoch, rng)
    aug = sample_augmentation(cfg, cfg.per_epoch, rng)

    stacked = np.stack([images, labels.astype(np.float32), fovs.astype(np.float32)], axis=-1)
    padded = tf.constant(np.pad(stacked, ((0, 0), (half, half), (half, half), (0, 0))))

    def crop_and_augment(
        center: Any, flip_rows: Any, flip_cols: Any, k: Any, brightness: Any, contrast: Any
    ) -> tuple[Any, Any, Any]:
        # A center at (r, c) sits at (r + half, c + half) in the padded
        # stack, so the patch starts at (r, c) there.
        patch = tf.slice(padded, [center[0], center[1], center[2], 0], [1, size, size, 3])[0]
        patch = tf.cond(flip_rows, lambda: tf.reverse(patch, [0]), lambda: patch)
        patch = tf.cond(flip_cols, lambda: tf.reverse(patch, [1]), lambda: patch)
        patch = tf.image.rot90(patch, k)
        x, y, w = patch[..., IMAGE:LABEL], patch[..., LABEL:FOV], patch[..., FOV:]
        mean = tf.reduce_sum(x * w) / tf.maximum(tf.reduce_sum(w), 1.0)
        # Written so that contrast 1 and brightness 0 return x exactly.
        x = (x * contrast + (1 - contrast) * mean + brightness) * w
        return x, y, w

    params = (
        centers,
        aug.flip_rows,
        aug.flip_cols,
        aug.rotations.astype(np.int32),
        aug.brightness.astype(np.float32),
        aug.contrast.astype(np.float32),
    )
    return (
        tf.data.Dataset.from_tensor_slices(params)
        .map(crop_and_augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
