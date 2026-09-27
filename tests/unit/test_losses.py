import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retinal_vessels.config import LossConfig
from retinal_vessels.losses import make_loss, masked_bce, pack_target, soft_dice_loss

SHAPE = (1, 2, 2, 1)


def target(y: list[float], w: list[float]) -> np.ndarray:
    ys = np.array(y, dtype=np.float32).reshape(SHAPE)
    ws = np.array(w, dtype=np.float32).reshape(SHAPE)
    return np.asarray(pack_target(ys, ys, ws)[1])


def pred(p: list[float]) -> np.ndarray:
    return np.array(p, dtype=np.float32).reshape(SHAPE)


def test_pack_target_stacks_label_then_weight() -> None:
    t = target([1, 0, 1, 0], [1, 1, 0, 0])
    assert t.shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(t[..., 0].ravel(), [1, 0, 1, 0])
    np.testing.assert_array_equal(t[..., 1].ravel(), [1, 1, 0, 0])


def test_soft_dice_known_value() -> None:
    # Weighted pixels: labels 1, 0 and predictions 0.5, 0.5.
    # overlap 0.5, total 1 + 1 = 2, so Dice is (1 + s) / (2 + s).
    loss = float(soft_dice_loss(target([1, 0, 1, 1], [1, 1, 0, 0]), pred([0.5, 0.5, 0, 0]), 1.0))
    assert loss == pytest.approx(1 - 2 / 3)


def test_soft_dice_is_zero_for_a_perfect_prediction() -> None:
    t = target([1, 0, 1, 0], [1, 1, 1, 1])
    assert float(soft_dice_loss(t, pred([1, 0, 1, 0]), 1.0)) == pytest.approx(0.0)


def test_masked_bce_known_value() -> None:
    # One weighted pixel, label 1, prediction 0.25: loss is -log(0.25).
    loss = float(masked_bce(target([1, 0, 0, 0], [1, 0, 0, 0]), pred([0.25, 0.9, 0.9, 0.9])))
    assert loss == pytest.approx(-math.log(0.25), rel=1e-5)


def test_masked_bce_is_zero_without_weighted_pixels() -> None:
    assert float(masked_bce(target([1, 0, 1, 0], [0, 0, 0, 0]), pred([0.3] * 4))) == 0.0


def test_masked_bce_is_finite_at_certain_wrong_predictions() -> None:
    loss = float(masked_bce(target([1, 0, 0, 0], [1, 1, 0, 0]), pred([0.0, 1.0, 0, 0])))
    assert math.isfinite(loss)
    assert loss > 10


@pytest.mark.parametrize(
    ("name", "expected"),
    [("dice", 1 - 2 / 3), ("bce_dice", -math.log(0.5) + 1 - 2 / 3)],
)
def test_make_loss_selects_by_name(name: str, expected: float) -> None:
    loss = make_loss(LossConfig(name=name, smooth=1.0))  # type: ignore[arg-type]
    value = loss(target([1, 0, 1, 1], [1, 1, 0, 0]), pred([0.5, 0.5, 0, 0]))
    assert float(value) == pytest.approx(expected, rel=1e-5)


def test_make_loss_rejects_unknown_name() -> None:
    cfg = LossConfig.model_construct(name="focal", smooth=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="focal"):
        make_loss(cfg)


probs = arrays(np.float32, (1, 4, 4, 1), elements=st.floats(0, 1, width=32))
masks = arrays(np.bool_, (1, 4, 4, 1))


@settings(max_examples=50, deadline=None)
@given(y=masks, w=masks, p=probs, noise=probs, name=st.sampled_from(["dice", "bce_dice"]))
def test_pixels_outside_the_fov_never_change_the_loss(
    y: np.ndarray, w: np.ndarray, p: np.ndarray, noise: np.ndarray, name: str
) -> None:
    loss = make_loss(LossConfig(name=name, smooth=1.0))  # type: ignore[arg-type]
    t = np.asarray(pack_target(p, y.astype(np.float32), w.astype(np.float32))[1])
    changed = np.where(w, p, noise)
    assert float(loss(t, p)) == pytest.approx(float(loss(t, changed)), rel=1e-6, abs=1e-6)
