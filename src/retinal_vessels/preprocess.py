"""Turn an RGB fundus image into the single-channel model input (SPEC section 7).

The steps run in a fixed order: green channel, CLAHE, scaling to [0, 1], and
standardization with the FOV's own mean and SD. Each can be switched off in
config for ablations. Pixels outside the FOV are always set to 0, so the
black border never enters the statistics or the model input.
"""

import cv2
import numpy as np

from retinal_vessels.config import PreprocessConfig

GREEN = 1
MAX_INTENSITY = 255


def preprocess(image: np.ndarray, fov: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Return the preprocessed image as (H, W) float32.

    ``image`` is (H, W, 3) uint8 RGB and ``fov`` is (H, W) bool. With the
    green channel off, the image is converted to grayscale instead, since
    CLAHE needs one channel. Bad shapes, dtypes, an empty FOV, or a constant
    FOV that cannot be standardized raise ``ValueError``.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"image must be (H, W, 3) uint8, got {image.shape} {image.dtype}")
    if fov.dtype != np.bool_ or fov.shape != image.shape[:2]:
        raise ValueError(
            f"fov must be bool of shape {image.shape[:2]}, got {fov.shape} {fov.dtype}"
        )
    if not fov.any():
        raise ValueError("FOV mask is empty")

    if cfg.green_channel:
        channel = np.ascontiguousarray(image[..., GREEN])
    else:
        channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if cfg.clahe:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid)
        channel = clahe.apply(channel)

    out = channel.astype(np.float32)
    if cfg.scale_unit:
        out /= MAX_INTENSITY
    if cfg.standardize_in_fov:
        inside = out[fov].astype(np.float64)
        # Rounding leaves a tiny nonzero SD on a constant FOV, so test for
        # constancy directly rather than comparing the SD with 0.
        if inside.min() == inside.max():
            raise ValueError("FOV is constant, so it cannot be standardized")
        out = (out - inside.mean()) / inside.std()
    out[~fov] = 0
    return out.astype(np.float32, copy=False)
