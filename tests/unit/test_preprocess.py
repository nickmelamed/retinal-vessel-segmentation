from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retinal_vessels.config import PreprocessConfig, load_config
from retinal_vessels.preprocess import preprocess

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SHAPE = (48, 40)


def config(**overrides: Any) -> PreprocessConfig:
    base = load_config(CONFIGS / "baseline.yaml").preprocess
    return base.model_copy(update=overrides)


OFF = {"clahe": False, "scale_unit": False, "standardize_in_fov": False}


def fov() -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[4:44, 5:35] = True
    return mask


def image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(*SHAPE, 3), dtype=np.uint8)


def test_output_is_float32_with_the_image_shape() -> None:
    out = preprocess(image(), fov(), config())
    assert out.shape == SHAPE
    assert out.dtype == np.float32


def test_green_channel_is_taken_as_is_when_other_steps_are_off() -> None:
    img = image()
    out = preprocess(img, fov(), config(**OFF))
    expected = np.where(fov(), img[..., 1], 0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_without_the_green_channel_the_image_is_grayscale() -> None:
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    img[..., 0] = 200  # red only, so the green channel would be all zero
    out = preprocess(img, fov(), config(green_channel=False, **OFF))
    assert out[fov()].min() > 0
    assert len(np.unique(out[fov()])) == 1


def test_scaling_maps_to_the_unit_interval() -> None:
    img = image()
    out = preprocess(img, fov(), config(clahe=False, standardize_in_fov=False))
    np.testing.assert_allclose(out[fov()], img[..., 1][fov()] / 255, rtol=1e-6)
    assert out.min() >= 0 and out.max() <= 1


def test_clahe_changes_a_low_contrast_image() -> None:
    img = np.full((*SHAPE, 3), 100, dtype=np.uint8)
    img[10:20, 10:20, 1] = 110
    plain = preprocess(img, fov(), config(**OFF))
    enhanced = preprocess(img, fov(), config(**{**OFF, "clahe": True}))
    assert not np.array_equal(plain, enhanced)
    assert np.ptp(enhanced[fov()]) > np.ptp(plain[fov()])


def test_standardization_uses_the_fov_statistics() -> None:
    out = preprocess(image(), fov(), config())
    inside = out[fov()].astype(np.float64)
    assert abs(inside.mean()) < 1e-5
    assert abs(inside.std() - 1) < 1e-5


def test_pixels_outside_the_fov_are_zero_for_every_setting() -> None:
    for overrides in ({}, OFF, {"clahe": False}, {"green_channel": False}):
        out = preprocess(image(), fov(), config(**overrides))
        assert not out[~fov()].any()


@pytest.mark.parametrize("clahe", [False, True])
@pytest.mark.parametrize("green_channel", [False, True])
def test_pixels_outside_the_fov_do_not_change_the_result(clahe: bool, green_channel: bool) -> None:
    a, b = image(0), image(1)
    b[fov()] = a[fov()]
    cfg = config(clahe=clahe, green_channel=green_channel)
    np.testing.assert_array_equal(preprocess(a, fov(), cfg), preprocess(b, fov(), cfg))


def test_constant_fov_cannot_be_standardized() -> None:
    img = np.full((*SHAPE, 3), 90, dtype=np.uint8)
    with pytest.raises(ValueError, match="FOV is constant"):
        preprocess(img, fov(), config(clahe=False))


@pytest.mark.parametrize(
    ("img", "mask", "message"),
    [
        (np.zeros(SHAPE, dtype=np.uint8), fov(), "image must be"),
        (np.zeros((*SHAPE, 4), dtype=np.uint8), fov(), "image must be"),
        (np.zeros((*SHAPE, 3), dtype=np.float32), fov(), "image must be"),
        (image(), fov().astype(np.uint8), "fov must be bool"),
        (image(), fov()[:-1], "fov must be bool"),
        (image(), np.zeros(SHAPE, dtype=bool), "FOV mask is empty"),
    ],
)
def test_bad_inputs_raise(img: np.ndarray, mask: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        preprocess(img, mask, config())


def test_input_is_not_modified() -> None:
    img = image()
    before = img.copy()
    preprocess(img, fov(), config())
    np.testing.assert_array_equal(img, before)


@settings(max_examples=50, deadline=None)
@given(img=arrays(np.uint8, (*SHAPE, 3)))
def test_scaled_output_stays_in_the_unit_interval(img: np.ndarray) -> None:
    out = preprocess(img, fov(), config(standardize_in_fov=False))
    assert out.min() >= 0 and out.max() <= 1
    assert np.isfinite(out).all()


@settings(max_examples=50, deadline=None)
@given(img=arrays(np.uint8, (*SHAPE, 3)), scale=st.booleans(), clahe=st.booleans())
def test_standardized_output_has_zero_mean_and_unit_sd(
    img: np.ndarray, scale: bool, clahe: bool
) -> None:
    cfg = config(scale_unit=scale, clahe=clahe)
    try:
        out = preprocess(img, fov(), cfg)
    except ValueError as err:
        assert "constant" in str(err)
        return
    inside = out[fov()].astype(np.float64)
    assert abs(inside.mean()) < 1e-4
    assert abs(inside.std() - 1) < 1e-4
