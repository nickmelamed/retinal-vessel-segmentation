import numpy as np
import pytest

from retinal_vessels.config import ModelConfig
from retinal_vessels.model import build_unet

SMALL = ModelConfig(depth=2, base_filters=4, dropout=0.1, batch_norm=True)


@pytest.mark.parametrize("shape", [(2, 16, 16, 1), (1, 32, 48, 1)])
def test_output_matches_input_shape(shape: tuple[int, ...]) -> None:
    model = build_unet(SMALL)
    x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    y = np.asarray(model(x, training=False))
    assert y.shape == shape
    assert ((y >= 0) & (y <= 1)).all()


def conv_filters(model: object) -> list[int]:
    layers = model.layers  # type: ignore[attr-defined]
    return [layer.filters for layer in layers if type(layer).__name__ == "Conv2D"]


def test_filters_double_per_level_and_come_back() -> None:
    filters = conv_filters(build_unet(SMALL))
    # Two convs per block: 4, 8 down, 16 at the bottom, 8, 4 up, then 1 out.
    assert filters == [4, 4, 8, 8, 16, 16, 8, 8, 4, 4, 1]


def test_depth_sets_the_number_of_levels() -> None:
    deeper = ModelConfig(depth=3, base_filters=2, dropout=0.0, batch_norm=True)
    assert max(conv_filters(build_unet(deeper))) == 16


def test_switches_off_batch_norm_and_dropout() -> None:
    plain = ModelConfig(depth=2, base_filters=4, dropout=0.0, batch_norm=False)
    names = {type(layer).__name__ for layer in build_unet(plain).layers}
    assert "BatchNormalization" not in names
    assert "Dropout" not in names
    names = {type(layer).__name__ for layer in build_unet(SMALL).layers}
    assert {"BatchNormalization", "Dropout"} <= names


def test_inference_is_deterministic_with_dropout() -> None:
    model = build_unet(SMALL)
    x = np.ones((1, 16, 16, 1), dtype=np.float32)
    np.testing.assert_array_equal(model(x, training=False), model(x, training=False))
