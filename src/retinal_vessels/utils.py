"""Logging setup and seeding shared by every entry point."""

import logging
import random

import numpy as np

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
HANDLER_NAME = "retinal_vessels"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once.

    Later calls do nothing, so a CLI and the library it imports can both call
    this without duplicating handlers.
    """
    root = logging.getLogger()
    if any(h.name == HANDLER_NAME for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def set_seed(seed: int, deterministic: bool = False) -> bool:
    """Seed Python, NumPy, and TensorFlow, and optionally force deterministic ops.

    Return whether deterministic ops were enabled. The run manifest records
    this, since GPU results repeat bit for bit only when it is true.
    TensorFlow is imported here so code that never trains skips its import cost.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    return deterministic
