from typing import Any

import numpy as np
import pytest

from retinal_vessels.config import InferenceConfig, ModelConfig
from retinal_vessels.model import build_unet
from retinal_vessels.predict import predict_image

CFG = InferenceConfig(window=8, stride=4, batch_size=3)


def identity(x: np.ndarray, training: bool) -> np.ndarray:
    return x


def ramp(x: np.ndarray, training: bool) -> np.ndarray:
    # Depends on the position inside the window, so the averaging shows.
    win = x.shape[1]
    grid = np.add.outer(np.arange(win), np.arange(win)).astype(np.float32)
    return np.broadcast_to(grid[None, :, :, None], x.shape).copy()


def image_and_fov(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.random.default_rng(0).random((h, w), dtype=np.float32)
    fov = np.ones((h, w), dtype=bool)
    fov[0, 0] = False
    return image, fov


@pytest.mark.parametrize("shape", [(20, 20), (21, 17), (5, 3), (8, 8), (584, 565)])
def test_output_matches_input_shape(shape: tuple[int, int]) -> None:
    image, fov = image_and_fov(*shape)
    out = predict_image(identity, image, fov, CFG)
    assert out.shape == shape
    assert out.dtype == np.float32


def test_identity_model_returns_the_image_inside_the_fov() -> None:
    image, fov = image_and_fov(21, 17)
    out = predict_image(identity, image, fov, CFG)
    np.testing.assert_allclose(out[fov], image[fov], rtol=1e-6)
    assert out[0, 0] == 0


def test_overlapping_windows_are_averaged() -> None:
    h, w = 13, 10
    image, fov = image_and_fov(h, w)
    fov[:] = True
    # Starts at 0, 4, 8 on rows (padded to 16) and 0, 4 on cols (padded to 12).
    total = np.zeros((16, 12))
    count = np.zeros((16, 12))
    tile = np.add.outer(np.arange(8), np.arange(8))
    for r in (0, 4, 8):
        for c in (0, 4):
            total[r : r + 8, c : c + 8] += tile
            count[r : r + 8, c : c + 8] += 1
    expected = (total / count)[:h, :w]
    np.testing.assert_allclose(predict_image(ramp, image, fov, CFG), expected, rtol=1e-6)


def test_runs_a_real_unet() -> None:
    model = build_unet(ModelConfig(depth=2, base_filters=2, dropout=0.0, batch_norm=True))
    image, fov = image_and_fov(30, 25)
    out = predict_image(model, image, fov, CFG)
    assert out.shape == (30, 25)
    assert ((out >= 0) & (out <= 1)).all()


def bad_shape(x: np.ndarray, training: bool) -> Any:
    return x[:, :-1]


@pytest.mark.parametrize(
    ("model", "image", "fov", "match"),
    [
        (identity, np.zeros((2, 4, 4), np.float32), np.ones((4, 4), bool), r"\(H, W\)"),
        (identity, np.zeros((4, 4), np.float64), np.ones((4, 4), bool), "float32"),
        (identity, np.zeros((4, 4), np.float32), np.ones((4, 5), bool), "fov"),
        (identity, np.zeros((4, 4), np.float32), np.ones((4, 4), np.uint8), "fov"),
        (bad_shape, np.zeros((4, 4), np.float32), np.ones((4, 4), bool), "model returned"),
    ],
)
def test_rejects_bad_inputs(model: Any, image: np.ndarray, fov: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        predict_image(model, image, fov, CFG)
