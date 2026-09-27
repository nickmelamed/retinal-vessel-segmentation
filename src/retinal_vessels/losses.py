"""FOV-masked training losses: BCE + Dice for the baseline, Dice alone as the ablation.

Keras passes a loss only ``y_true`` and ``y_pred``, so the FOV weight rides
along as a second channel of ``y_true``. :func:`pack_target` builds that
target from the ``(x, y, w)`` batches in ``patches.py``. Pixels with weight 0
(outside the FOV, or padding past the image edge) never affect the loss.
"""

from collections.abc import Callable
from typing import Any

from retinal_vessels.config import LossConfig

LABEL, WEIGHT = 0, 1

Loss = Callable[[Any, Any], Any]


def pack_target(x: Any, y: Any, w: Any) -> tuple[Any, Any]:
    """Map a patch batch to ``(x, target)`` with ``target = concat(y, w)`` on the last axis."""
    import keras

    return x, keras.ops.concatenate([y, w], axis=-1)


def _unpack(y_true: Any) -> tuple[Any, Any]:
    return y_true[..., LABEL : LABEL + 1], y_true[..., WEIGHT : WEIGHT + 1]


def soft_dice_loss(y_true: Any, y_pred: Any, smooth: float) -> Any:
    """Return 1 minus the soft Dice of the batch, pooled over every weighted pixel."""
    import keras

    y, w = _unpack(y_true)
    overlap = keras.ops.sum(w * y * y_pred)
    total = keras.ops.sum(w * y) + keras.ops.sum(w * y_pred)
    return 1.0 - (2.0 * overlap + smooth) / (total + smooth)


def masked_bce(y_true: Any, y_pred: Any) -> Any:
    """Return binary cross-entropy averaged over the weighted pixels of the batch.

    A batch with no weighted pixels has loss 0.
    """
    import keras

    y, w = _unpack(y_true)
    eps = keras.config.epsilon()
    p = keras.ops.clip(y_pred, eps, 1.0 - eps)
    bce = -(y * keras.ops.log(p) + (1.0 - y) * keras.ops.log(1.0 - p))
    return keras.ops.sum(w * bce) / keras.ops.maximum(keras.ops.sum(w), 1.0)


def make_loss(cfg: LossConfig) -> Loss:
    """Return the loss named in ``cfg`` as a Keras-compatible function."""
    if cfg.name == "dice":

        def dice(y_true: Any, y_pred: Any) -> Any:
            return soft_dice_loss(y_true, y_pred, cfg.smooth)

        return dice
    if cfg.name == "bce_dice":

        def bce_dice(y_true: Any, y_pred: Any) -> Any:
            return masked_bce(y_true, y_pred) + soft_dice_loss(y_true, y_pred, cfg.smooth)

        return bce_dice
    raise ValueError(f"unknown loss {cfg.name!r}")
