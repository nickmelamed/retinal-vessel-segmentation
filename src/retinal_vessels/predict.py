"""Sliding-window inference over whole images (SPEC section 7).

Windows overlap by ``window - stride`` pixels, and each pixel's probability
is the mean over every window that covers it, which smooths the seams where
one window's border context ends.
"""

from typing import Any

import numpy as np

from retinal_vessels.config import InferenceConfig


def _starts(length: int, window: int, stride: int) -> list[int]:
    # Pad to the smallest length that fits a whole number of strides, so the
    # last window ends at the padded edge and every pixel is covered.
    steps = max(0, -(-(length - window) // stride))
    return [i * stride for i in range(steps + 1)]


def predict_image(
    model: Any, image: np.ndarray, fov: np.ndarray, cfg: InferenceConfig
) -> np.ndarray:
    """Return per-pixel vessel probabilities for one preprocessed image.

    ``image`` is (H, W) float32 from ``preprocess`` and ``fov`` is (H, W)
    bool. The result is (H, W) float32, 0 outside the FOV. ``model`` maps a
    (batch, window, window, 1) float32 array to probabilities of the same
    shape. The image is zero-padded on the bottom and right edges where the
    windows reach past it, matching how patches are padded in training.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be (H, W), got shape {image.shape}")
    if image.dtype != np.float32:
        raise ValueError(f"image must be float32, got {image.dtype}")
    if fov.shape != image.shape or fov.dtype != np.bool_:
        raise ValueError(f"fov must be bool with shape {image.shape}, got {fov.dtype} {fov.shape}")
    height, width = image.shape
    win, stride = cfg.window, cfg.stride
    rows, cols = _starts(height, win, stride), _starts(width, win, stride)
    padded = np.zeros((rows[-1] + win, cols[-1] + win), dtype=np.float32)
    padded[:height, :width] = image

    corners = [(r, c) for r in rows for c in cols]
    total = np.zeros_like(padded)
    count = np.zeros_like(padded)
    for first in range(0, len(corners), cfg.batch_size):
        batch = corners[first : first + cfg.batch_size]
        windows = np.stack([padded[r : r + win, c : c + win] for r, c in batch])[..., None]
        probs = np.asarray(model(windows, training=False), dtype=np.float32)
        if probs.shape != windows.shape:
            raise ValueError(f"model returned shape {probs.shape} for windows {windows.shape}")
        for (r, c), p in zip(batch, probs[..., 0], strict=True):
            total[r : r + win, c : c + win] += p
            count[r : r + win, c : c + win] += 1
    mean = (total / count)[:height, :width]
    return np.where(fov, mean, 0).astype(np.float32)
