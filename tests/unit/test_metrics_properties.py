"""Properties every metric must hold on any image, checked with Hypothesis."""

from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retinal_vessels.metrics import (
    auc_roc,
    average_precision,
    binarize,
    binary_metrics,
    brier,
    dice_per_threshold,
    reliability,
    threshold_grid,
    width_sensitivity,
)

SHAPE = (8, 9)
EDGE = 1.0
THRESHOLD = 0.5
BINS = 10

probabilities = arrays(np.float32, SHAPE, elements=st.floats(0, 1, width=32, allow_subnormal=False))
masks = arrays(np.bool_, SHAPE)
fovs = masks.filter(lambda f: bool(f.any()))


def scores(prob: np.ndarray, label: np.ndarray, fov: np.ndarray) -> dict[str, Any]:
    width = width_sensitivity(binarize(prob, THRESHOLD), label, fov, EDGE)
    table = reliability([prob], [label], [fov], BINS)
    return {
        "binary": binary_metrics(binarize(prob, THRESHOLD), label, fov),
        "auc_roc": auc_roc(prob, label, fov),
        "average_precision": average_precision(prob, label, fov),
        "brier": brier(prob, label, fov),
        "width": width,
        "reliability": table,
    }


def in_unit_interval(value: float | None) -> bool:
    return value is None or 0 <= value <= 1


@settings(max_examples=150, deadline=None)
@given(prob=probabilities, label=masks, fov=fovs)
def test_every_score_lies_in_the_unit_interval(
    prob: np.ndarray, label: np.ndarray, fov: np.ndarray
) -> None:
    s = scores(prob, label, fov)
    m = s["binary"]
    values = [
        m.dice,
        m.sensitivity,
        m.specificity,
        m.precision,
        m.accuracy,
        m.predicted_vessel_fraction,
        s["auc_roc"],
        s["average_precision"],
        s["brier"],
        s["width"].thin,
        s["width"].thick,
        s["reliability"].pooled_brier,
        *s["reliability"].mean_probability,
        *s["reliability"].vessel_fraction,
    ]
    assert all(in_unit_interval(v) for v in values), values
    assert sum(s["reliability"].counts) == int(fov.sum())


@settings(max_examples=150, deadline=None)
@given(
    prob=probabilities,
    label=masks,
    fov=fovs,
    other_prob=probabilities,
    other_label=masks,
)
def test_pixels_outside_the_fov_never_change_a_score(
    prob: np.ndarray,
    label: np.ndarray,
    fov: np.ndarray,
    other_prob: np.ndarray,
    other_label: np.ndarray,
) -> None:
    changed_prob = np.where(fov, prob, other_prob)
    changed_label = np.where(fov, label, other_label)
    assert scores(changed_prob, changed_label, fov) == scores(prob, label, fov)


@settings(max_examples=150, deadline=None)
@given(label=masks, fov=fovs)
def test_a_perfect_prediction_scores_perfectly(label: np.ndarray, fov: np.ndarray) -> None:
    s = scores(label.astype(np.float32), label, fov)
    m = s["binary"]
    vessel = label[fov]
    assert m.dice == 1.0
    assert m.accuracy == 1.0
    assert m.sensitivity == (1.0 if vessel.any() else None)
    assert m.specificity == (None if vessel.all() else 1.0)
    assert s["auc_roc"] == (1.0 if vessel.any() and not vessel.all() else None)
    assert s["average_precision"] == (1.0 if vessel.any() else None)
    assert s["brier"] == 0.0
    assert s["width"].thin in (1.0, None)
    assert s["width"].thick in (1.0, None)


@settings(max_examples=100, deadline=None)
@given(prob=probabilities, label=masks, fov=fovs)
def test_the_threshold_sweep_agrees_with_binarize(
    prob: np.ndarray, label: np.ndarray, fov: np.ndarray
) -> None:
    grid = threshold_grid(20)
    swept = dice_per_threshold(prob, label, fov, grid)
    one_at_a_time = [binary_metrics(binarize(prob, float(t)), label, fov).dice for t in grid]
    np.testing.assert_array_equal(swept, one_at_a_time)
