"""Segmentation and calibration metrics, computed inside the FOV only (SPEC section 5).

A pixel counts as vessel when its probability is at least the threshold.
Pixels outside the FOV never enter any count. A ratio whose denominator is
zero (sensitivity on an image with no labeled vessels, say) is None, which
the database stores as NULL. Dice is the exception: when neither the label
nor the prediction has a vessel pixel the two agree completely, so Dice is 1.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ConfusionCounts:
    """Pixel counts inside the FOV."""

    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class BinaryMetrics:
    """Metrics of one thresholded prediction against its label."""

    dice: float
    sensitivity: float | None
    specificity: float | None
    precision: float | None
    accuracy: float
    predicted_vessel_fraction: float


def _check(name: str, array: np.ndarray, shape: tuple[int, ...], dtype: type) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    if array.dtype != dtype:
        raise ValueError(f"{name} must be {np.dtype(dtype)}, got {array.dtype}")


def _check_inputs(
    prediction: np.ndarray, label: np.ndarray, fov: np.ndarray, prediction_dtype: type
) -> None:
    if label.ndim != 2:
        raise ValueError(f"label must be (H, W), got shape {label.shape}")
    _check("label", label, label.shape, np.bool_)
    _check("prediction", prediction, label.shape, prediction_dtype)
    _check("fov", fov, label.shape, np.bool_)
    if not fov.any():
        raise ValueError("FOV mask is empty")


def confusion_counts(prediction: np.ndarray, label: np.ndarray, fov: np.ndarray) -> ConfusionCounts:
    """Count true and false positives and negatives inside ``fov``.

    All three arrays are (H, W) bool.
    """
    _check_inputs(prediction, label, fov, np.bool_)
    p, t = prediction[fov], label[fov]
    return ConfusionCounts(
        tp=int(np.sum(p & t)),
        fp=int(np.sum(p & ~t)),
        fn=int(np.sum(~p & t)),
        tn=int(np.sum(~p & ~t)),
    )


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def dice(c: ConfusionCounts) -> float:
    """Return 2TP / (2TP + FP + FN), or 1 when there are no vessel pixels on either side."""
    den = 2 * c.tp + c.fp + c.fn
    return 1.0 if den == 0 else 2 * c.tp / den


def binary_metrics(prediction: np.ndarray, label: np.ndarray, fov: np.ndarray) -> BinaryMetrics:
    """Return every confusion-matrix metric for one (H, W) bool prediction."""
    c = confusion_counts(prediction, label, fov)
    total = c.tp + c.fp + c.fn + c.tn
    return BinaryMetrics(
        dice=dice(c),
        sensitivity=_ratio(c.tp, c.tp + c.fn),
        specificity=_ratio(c.tn, c.tn + c.fp),
        precision=_ratio(c.tp, c.tp + c.fp),
        accuracy=(c.tp + c.tn) / total,
        predicted_vessel_fraction=(c.tp + c.fp) / total,
    )


def _scores(
    probability: np.ndarray, label: np.ndarray, fov: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _check_inputs(probability, label, fov, np.float32)
    return probability[fov].astype(np.float64), label[fov]


def _counts_per_score(p: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Vessel and background pixel counts at each distinct score, in ascending
    # score order. Tied pixels share one entry, which is how both AUCs treat ties.
    values, inverse = np.unique(p, return_inverse=True)
    vessel = np.bincount(inverse, weights=t, minlength=len(values))
    background = np.bincount(inverse, weights=~t, minlength=len(values))
    return vessel, background


def auc_roc(probability: np.ndarray, label: np.ndarray, fov: np.ndarray) -> float | None:
    """Return the area under the ROC curve inside ``fov``, or None if one class is absent.

    This is the chance that a random vessel pixel scores above a random
    background pixel, with a tie counting one half. ``probability`` is
    (H, W) float32.
    """
    p, t = _scores(probability, label, fov)
    n_vessel = int(t.sum())
    n_background = len(t) - n_vessel
    if n_vessel == 0 or n_background == 0:
        return None
    vessel, background = _counts_per_score(p, t)
    below = np.cumsum(background) - background
    return float(np.sum(vessel * (below + background / 2)) / (n_vessel * n_background))


def average_precision(probability: np.ndarray, label: np.ndarray, fov: np.ndarray) -> float | None:
    """Return the area under the precision-recall curve inside ``fov`` as average precision.

    The sum over distinct thresholds of the gain in recall times the
    precision there, with no interpolation between points. None when the FOV
    holds no vessel pixel. ``probability`` is (H, W) float32.
    """
    p, t = _scores(probability, label, fov)
    n_vessel = int(t.sum())
    if n_vessel == 0:
        return None
    vessel, background = _counts_per_score(p, t)
    tp = np.cumsum(vessel[::-1])
    fp = np.cumsum(background[::-1])
    # Dividing by n_vessel once, after the sum, keeps the result at most 1.
    # Summing the recall gains one by one can round to just above it.
    return float(np.sum(vessel[::-1] * (tp / (tp + fp))) / n_vessel)


def brier(probability: np.ndarray, label: np.ndarray, fov: np.ndarray) -> float:
    """Return the mean squared difference between probability and label inside ``fov``."""
    p, t = _scores(probability, label, fov)
    return float(np.mean((p - t) ** 2))


@dataclass(frozen=True)
class ReliabilityTable:
    """Pooled calibration of many images, for the reliability diagram.

    Bin ``i`` holds probabilities in ``[edges[i], edges[i + 1])``, and the
    last bin also holds 1. The mean probability and vessel fraction of an
    empty bin are None.
    """

    edges: tuple[float, ...]
    counts: tuple[int, ...]
    mean_probability: tuple[float | None, ...]
    vessel_fraction: tuple[float | None, ...]
    pooled_brier: float


def reliability(
    probabilities: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    fovs: Sequence[np.ndarray],
    n_bins: int,
) -> ReliabilityTable:
    """Pool every FOV pixel of every image into ``n_bins`` equal-width probability bins.

    Probabilities outside [0, 1] raise ``ValueError``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    if not probabilities or not len(probabilities) == len(labels) == len(fovs):
        raise ValueError(
            f"need matching, non-empty sequences, got {len(probabilities)} probabilities, "
            f"{len(labels)} labels, {len(fovs)} FOV masks"
        )
    edges = np.arange(n_bins + 1) / n_bins
    counts = np.zeros(n_bins, dtype=np.int64)
    prob_sums = np.zeros(n_bins)
    vessel_sums = np.zeros(n_bins)
    squared_error = 0.0
    for probability, label, fov in zip(probabilities, labels, fovs, strict=True):
        p, t = _scores(probability, label, fov)
        if p.min() < 0 or p.max() > 1:
            raise ValueError(f"probabilities must lie in [0, 1], got [{p.min()}, {p.max()}]")
        # Like the threshold grid, a probability on an edge goes to the upper bin.
        index = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
        counts += np.bincount(index, minlength=n_bins)
        prob_sums += np.bincount(index, weights=p, minlength=n_bins)
        vessel_sums += np.bincount(index, weights=t, minlength=n_bins)
        squared_error += float(np.sum((p - t) ** 2))

    def per_bin(sums: np.ndarray) -> tuple[float | None, ...]:
        return tuple(None if c == 0 else float(s / c) for s, c in zip(sums, counts, strict=True))

    return ReliabilityTable(
        edges=tuple(float(e) for e in edges),
        counts=tuple(int(c) for c in counts),
        mean_probability=per_bin(prob_sums),
        vessel_fraction=per_bin(vessel_sums),
        pooled_brier=squared_error / int(counts.sum()),
    )


def skeleton_radius(label: np.ndarray, fov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the vessel skeleton and each pixel's vessel radius, inside ``fov``.

    Both are (H, W): the skeleton bool, and the radius float64, the Euclidean
    distance from each vessel pixel to the nearest non-vessel pixel. A one
    pixel wide vessel has radius 1 on its skeleton. Only ``label & fov``
    counts as vessel, so pixels outside the FOV never change either result.
    """
    _check_inputs(label, label, fov, np.bool_)
    vessel = label & fov
    # skimage ships no type hints for skeletonize.
    skeleton: np.ndarray = skeletonize(vessel)  # type: ignore[no-untyped-call]
    # OpenCV treats beyond the image border as vessel, so pad with background.
    padded = np.pad(vessel, 1).astype(np.uint8)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return skeleton, distance[1:-1, 1:-1].astype(np.float64)


def thin_edge(labels: Sequence[np.ndarray], fovs: Sequence[np.ndarray], quantile: float) -> float:
    """Return the ``quantile`` of skeleton radius pooled over ``labels``.

    This sets a fold's thin/thick edge from its training and validation
    labels (D-021). The result is an observed radius, not an interpolation
    between two, so it is the edge that bins those pixels.
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile}")
    if not labels or len(labels) != len(fovs):
        raise ValueError(f"need matching, non-empty sequences, got {len(labels)}, {len(fovs)}")
    radii = []
    for label, fov in zip(labels, fovs, strict=True):
        skeleton, radius = skeleton_radius(label, fov)
        radii.append(radius[skeleton])
    pooled = np.concatenate(radii)
    if pooled.size == 0:
        raise ValueError("labels have no vessel skeleton inside the FOV")
    return float(np.quantile(pooled, quantile, method="inverted_cdf"))


@dataclass(frozen=True)
class WidthSensitivity:
    """Sensitivity on thin and thick skeleton pixels, with the pixel counts behind them."""

    thin: float | None
    thick: float | None
    n_thin: int
    n_thick: int


def width_sensitivity(
    prediction: np.ndarray, label: np.ndarray, fov: np.ndarray, edge: float
) -> WidthSensitivity:
    """Return the fraction of thin and of thick skeleton pixels that ``prediction`` marks.

    A skeleton pixel is thin when its radius is at most ``edge``. All three
    arrays are (H, W) bool. A bin with no pixels has sensitivity None.
    """
    _check_inputs(prediction, label, fov, np.bool_)
    skeleton, radius = skeleton_radius(label, fov)
    thin = skeleton & (radius <= edge)
    thick = skeleton & (radius > edge)
    n_thin, n_thick = int(thin.sum()), int(thick.sum())
    return WidthSensitivity(
        thin=_ratio(int(np.sum(prediction & thin)), n_thin),
        thick=_ratio(int(np.sum(prediction & thick)), n_thick),
        n_thin=n_thin,
        n_thick=n_thick,
    )


def threshold_grid(divisions: int) -> np.ndarray:
    """Return the candidate thresholds ``i / divisions`` for i = 1 .. divisions - 1."""
    if divisions < 2:
        raise ValueError(f"divisions must be at least 2, got {divisions}")
    return np.arange(1, divisions) / divisions


def binarize(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Return ``probability >= threshold`` as bool, compared in float64.

    Most grid thresholds are not exact in float32, so comparing in float32
    could put a probability on the other side of the threshold than the
    sweep in :func:`dice_per_threshold` does, which also compares in float64.
    """
    return np.asarray(probability, dtype=np.float64) >= np.float64(threshold)


def dice_per_threshold(
    probability: np.ndarray, label: np.ndarray, fov: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Return the Dice of ``probability >= t`` for every ``t`` in ``grid``, in one pass.

    ``probability`` is (H, W) float32 in [0, 1]. Sorting the vessel and
    background probabilities once gives the true and false positive counts
    at every threshold by binary search, the same counts
    :func:`confusion_counts` gives for each threshold separately.
    """
    _check_inputs(probability, label, fov, np.float32)
    p, t = probability[fov].astype(np.float64), label[fov]
    grid = np.asarray(grid, dtype=np.float64)
    vessel, background = np.sort(p[t]), np.sort(p[~t])
    tp = len(vessel) - np.searchsorted(vessel, grid, side="left")
    fp = len(background) - np.searchsorted(background, grid, side="left")
    fn = len(vessel) - tp
    den = 2 * tp + fp + fn
    safe = np.where(den == 0, 1, den)
    return np.where(den == 0, 1.0, 2 * tp / safe)


def best_threshold(
    probabilities: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    fovs: Sequence[np.ndarray],
    grid: np.ndarray,
) -> tuple[float, float]:
    """Return the grid threshold with the highest mean per-image Dice, and that Dice.

    This is the fold threshold rule of SPEC section 5, applied to the
    validation images. Ties go to the lowest threshold.
    """
    if not probabilities or not len(probabilities) == len(labels) == len(fovs):
        raise ValueError(
            f"need matching, non-empty sequences, got {len(probabilities)} probabilities, "
            f"{len(labels)} labels, {len(fovs)} FOV masks"
        )
    if len(grid) == 0:
        raise ValueError("threshold grid is empty")
    per_image = np.stack(
        [
            dice_per_threshold(p, t, f, grid)
            for p, t, f in zip(probabilities, labels, fovs, strict=True)
        ]
    )
    mean = per_image.mean(axis=0)
    best = int(np.argmax(mean))
    return float(grid[best]), float(mean[best])


SELECTION_RULE = "max mean per-image validation Dice over the grid i/{divisions}"


def selection_rule(divisions: int) -> str:
    """Describe the threshold rule for ``thresholds.selection_rule``."""
    return SELECTION_RULE.format(divisions=divisions)
