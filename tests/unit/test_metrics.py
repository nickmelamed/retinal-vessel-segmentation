import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retinal_vessels.metrics import (
    ConfusionCounts,
    auc_roc,
    average_precision,
    best_threshold,
    binarize,
    binary_metrics,
    brier,
    confusion_counts,
    dice,
    dice_per_threshold,
    reliability,
    selection_rule,
    threshold_grid,
)


def b(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=bool)


LABEL = b([[1, 1, 0], [0, 0, 0]])
PRED = b([[1, 0, 1], [0, 0, 1]])
FOV = b([[1, 1, 1], [1, 1, 0]])


def test_confusion_counts_by_hand() -> None:
    # Inside the FOV: TP at (0,0), FN at (0,1), FP at (0,2), TN at (1,0) and (1,1).
    # The FP at (1,2) is outside the FOV and must not count.
    assert confusion_counts(PRED, LABEL, FOV) == ConfusionCounts(tp=1, fp=1, fn=1, tn=2)


def test_binary_metrics_by_hand() -> None:
    m = binary_metrics(PRED, LABEL, FOV)
    assert m.dice == pytest.approx(2 / 4)
    assert m.sensitivity == pytest.approx(1 / 2)
    assert m.specificity == pytest.approx(2 / 3)
    assert m.precision == pytest.approx(1 / 2)
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.predicted_vessel_fraction == pytest.approx(2 / 5)


def test_pixels_outside_the_fov_are_ignored() -> None:
    outside = ~FOV
    flipped_pred = np.where(outside, ~PRED, PRED)
    flipped_label = np.where(outside, ~LABEL, LABEL)
    assert binary_metrics(flipped_pred, flipped_label, FOV) == binary_metrics(PRED, LABEL, FOV)


def test_perfect_prediction() -> None:
    m = binary_metrics(LABEL, LABEL, FOV)
    assert (m.dice, m.sensitivity, m.specificity, m.precision, m.accuracy) == (1, 1, 1, 1, 1)


def test_undefined_ratios_are_none_and_empty_dice_is_one() -> None:
    empty = np.zeros_like(LABEL)
    m = binary_metrics(empty, empty, FOV)
    assert m.dice == 1.0
    assert m.sensitivity is None
    assert m.precision is None
    assert m.specificity == 1.0
    full = np.ones_like(LABEL)
    assert binary_metrics(full, full, FOV).specificity is None


def test_dice_of_counts() -> None:
    assert dice(ConfusionCounts(tp=0, fp=3, fn=0, tn=1)) == 0.0
    assert dice(ConfusionCounts(tp=0, fp=0, fn=0, tn=9)) == 1.0


@pytest.mark.parametrize(
    ("pred", "label", "fov", "match"),
    [
        (PRED, LABEL, np.zeros_like(FOV), "empty"),
        (PRED[:, :2], LABEL, FOV, "prediction shape"),
        (PRED.astype(np.uint8), LABEL, FOV, "prediction must be bool"),
        (PRED, LABEL.astype(np.float32), FOV, "label must be bool"),
        (PRED[None], LABEL[None], FOV[None], r"\(H, W\)"),
    ],
)
def test_rejects_bad_inputs(
    pred: np.ndarray, label: np.ndarray, fov: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        confusion_counts(pred, label, fov)


def test_threshold_grid() -> None:
    np.testing.assert_allclose(threshold_grid(4), [0.25, 0.5, 0.75])
    assert len(threshold_grid(100)) == 99
    with pytest.raises(ValueError, match="at least 2"):
        threshold_grid(1)


def brute_force(
    prob: np.ndarray, label: np.ndarray, fov: np.ndarray, grid: np.ndarray
) -> list[float]:
    return [binary_metrics(prob >= t, label, fov).dice for t in grid]


def test_dice_per_threshold_counts_equal_as_vessel() -> None:
    # A probability exactly on the threshold counts as vessel.
    prob = np.array([[0.5, 0.25]], dtype=np.float32)
    label = b([[1, 0]])
    fov = b([[1, 1]])
    grid = np.array([0.25, 0.5, 0.75])
    np.testing.assert_allclose(dice_per_threshold(prob, label, fov, grid), [2 / 3, 1.0, 0.0])


@settings(max_examples=100, deadline=None)
@given(
    prob=arrays(np.float32, (5, 6), elements=st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.7, 1.0])),
    label=arrays(np.bool_, (5, 6)),
    fov=arrays(np.bool_, (5, 6)).filter(lambda f: bool(f.any())),
)
def test_dice_per_threshold_matches_brute_force(
    prob: np.ndarray, label: np.ndarray, fov: np.ndarray
) -> None:
    grid = threshold_grid(20)
    np.testing.assert_allclose(
        dice_per_threshold(prob, label, fov, grid), brute_force(prob, label, fov, grid)
    )


def test_best_threshold_maximizes_mean_per_image_dice() -> None:
    fov = b([[1, 1, 1, 1]])
    label = b([[1, 1, 0, 0]])
    first = np.array([[0.9, 0.6, 0.4, 0.1]], dtype=np.float32)
    second = np.array([[0.8, 0.7, 0.2, 0.1]], dtype=np.float32)
    grid = threshold_grid(10)
    t, d = best_threshold([first, second], [label, label], [fov, fov], grid)
    # Both images are perfect for thresholds in (0.4, 0.6], so the lowest of those wins.
    assert t == pytest.approx(0.5)
    assert d == pytest.approx(1.0)
    mean = (brute_force(first, label, fov, grid)[0] + brute_force(second, label, fov, grid)[0]) / 2
    t_low, _ = best_threshold([first, second], [label, label], [fov, fov], grid[:1])
    assert t_low == pytest.approx(0.1)
    assert best_threshold([first, second], [label, label], [fov, fov], grid[:1])[
        1
    ] == pytest.approx(mean)


def test_best_threshold_rejects_mismatched_inputs() -> None:
    grid = threshold_grid(10)
    with pytest.raises(ValueError, match="matching"):
        best_threshold([], [], [], grid)
    with pytest.raises(ValueError, match="matching"):
        best_threshold([PRED.astype(np.float32)], [LABEL, LABEL], [FOV], grid)
    with pytest.raises(ValueError, match="grid is empty"):
        best_threshold([PRED.astype(np.float32)], [LABEL], [FOV], grid[:0])


def test_selection_rule_names_the_grid() -> None:
    assert selection_rule(100) == "max mean per-image validation Dice over the grid i/100"


def test_threshold_is_applied_in_the_same_precision_everywhere() -> None:
    # float32(t) differs from t for many grid values, so a probability that
    # lands exactly on float32(t) sits on one side of t in float64 and on the
    # other in float32. The sweep and binarize must agree on which side.
    grid = threshold_grid(100)
    label = b([[1, 0]])
    fov = b([[1, 1]])
    for t in grid:
        prob = np.array([[np.float32(t), 0.0]], dtype=np.float32)
        swept = dice_per_threshold(prob, label, fov, np.array([t]))[0]
        assert binary_metrics(binarize(prob, float(t)), label, fov).dice == swept


# Inside the FOV the scores are 0.9 (vessel), 0.8, 0.7 (vessel), and 0.1. The
# last pixel, outside the FOV, is a vessel scored 0 and must not count.
SCORES = np.array([[0.9, 0.8, 0.7, 0.1, 0.0]], dtype=np.float32)
SCORE_LABEL = b([[1, 0, 1, 0, 1]])
SCORE_FOV = b([[1, 1, 1, 1, 0]])


def test_auc_roc_by_hand() -> None:
    # Vessel 0.9 beats both background pixels, vessel 0.7 beats only 0.1.
    assert auc_roc(SCORES, SCORE_LABEL, SCORE_FOV) == pytest.approx(3 / 4)


def test_average_precision_by_hand() -> None:
    # Recall rises by 1/2 at 0.9 (precision 1) and by 1/2 at 0.7 (precision 2/3).
    assert average_precision(SCORES, SCORE_LABEL, SCORE_FOV) == pytest.approx(1 / 2 + 1 / 3)


def test_brier_by_hand() -> None:
    expected = (0.1**2 + 0.8**2 + 0.3**2 + 0.1**2) / 4
    assert brier(SCORES, SCORE_LABEL, SCORE_FOV) == pytest.approx(expected)


def test_tied_scores_count_as_one_threshold() -> None:
    prob = np.array([[0.5, 0.5]], dtype=np.float32)
    label, fov = b([[1, 0]]), b([[1, 1]])
    assert auc_roc(prob, label, fov) == pytest.approx(0.5)
    assert average_precision(prob, label, fov) == pytest.approx(0.5)


def test_ranking_scores_on_one_class_fov() -> None:
    fov = b([[1, 1]])
    prob = np.array([[0.2, 0.6]], dtype=np.float32)
    assert auc_roc(prob, b([[0, 0]]), fov) is None
    assert auc_roc(prob, b([[1, 1]]), fov) is None
    assert average_precision(prob, b([[0, 0]]), fov) is None
    assert average_precision(prob, b([[1, 1]]), fov) == 1.0


def test_perfect_scores() -> None:
    prob = SCORE_LABEL.astype(np.float32)
    assert auc_roc(prob, SCORE_LABEL, SCORE_FOV) == 1.0
    assert average_precision(prob, SCORE_LABEL, SCORE_FOV) == 1.0
    assert brier(prob, SCORE_LABEL, SCORE_FOV) == 0.0


def test_ranking_scores_reject_bool_prediction() -> None:
    for score in (auc_roc, average_precision, brier):
        with pytest.raises(ValueError, match="prediction must be float32"):
            score(SCORE_LABEL, SCORE_LABEL, SCORE_FOV)


def pairwise_auc(p: np.ndarray, t: np.ndarray) -> float:
    vessel, background = p[t][:, None], p[~t][None, :]
    return float(np.mean((vessel > background) + 0.5 * (vessel == background)))


def per_pixel_precision(p: np.ndarray, t: np.ndarray) -> float:
    # The mean, over vessel pixels, of the precision at that pixel's score.
    return float(np.mean([np.sum(t[p >= s]) / np.sum(p >= s) for s in p[t]]))


@settings(max_examples=200, deadline=None)
@given(
    prob=arrays(np.float32, (4, 5), elements=st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.7, 1.0])),
    label=arrays(np.bool_, (4, 5)),
    fov=arrays(np.bool_, (4, 5)).filter(lambda f: bool(f.any())),
)
def test_ranking_scores_match_brute_force(
    prob: np.ndarray, label: np.ndarray, fov: np.ndarray
) -> None:
    p, t = prob[fov].astype(np.float64), label[fov]
    roc, ap = auc_roc(prob, label, fov), average_precision(prob, label, fov)
    if t.any() and not t.all():
        assert roc == pytest.approx(pairwise_auc(p, t))
    else:
        assert roc is None
    if t.any():
        assert ap == pytest.approx(per_pixel_precision(p, t))
    else:
        assert ap is None
    assert brier(prob, label, fov) == pytest.approx(float(np.mean((p - t) ** 2)))


def test_reliability_by_hand() -> None:
    # Bins of width 1/4. A probability on an edge goes up a bin, and 1.0 stays
    # in the last. The 0.95 outside the first FOV must not count.
    first = np.array([[0.05, 0.5, 1.0, 0.95]], dtype=np.float32)
    second = np.array([[0.5]], dtype=np.float32)
    table = reliability(
        [first, second],
        [b([[1, 0, 1, 0]]), b([[1]])],
        [b([[1, 1, 1, 0]]), b([[1]])],
        n_bins=4,
    )
    assert table.edges == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert table.counts == (1, 0, 2, 1)
    assert table.mean_probability == pytest.approx((0.05, None, 0.5, 1.0))
    assert table.vessel_fraction == (1.0, None, 0.5, 1.0)
    assert table.pooled_brier == pytest.approx((0.95**2 + 0.5**2 + 0 + 0.5**2) / 4)


def test_reliability_rejects_bad_inputs() -> None:
    prob = np.array([[0.2, 0.6]], dtype=np.float32)
    label, fov = b([[1, 0]]), b([[1, 1]])
    with pytest.raises(ValueError, match="at least 2"):
        reliability([prob], [label], [fov], n_bins=1)
    with pytest.raises(ValueError, match="matching"):
        reliability([prob], [label, label], [fov], n_bins=10)
    with pytest.raises(ValueError, match="matching"):
        reliability([], [], [], n_bins=10)
    outside = np.array([[1.5, 0.6]], dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reliability([outside], [label], [fov], n_bins=10)
    negative = np.array([[-0.1, 0.6]], dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reliability([negative], [label], [fov], n_bins=10)
