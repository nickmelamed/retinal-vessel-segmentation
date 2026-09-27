import logging
import random
from collections.abc import Iterator

import numpy as np
import pytest

from retinal_vessels import __version__
from retinal_vessels.utils import HANDLER_NAME, set_seed, setup_logging


@pytest.fixture
def clean_root_logger() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    yield root
    root.handlers = before
    root.setLevel(level)


def test_version_is_exposed() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_setup_logging_adds_one_handler(clean_root_logger: logging.Logger) -> None:
    setup_logging(logging.DEBUG)
    setup_logging(logging.DEBUG)
    ours = [h for h in clean_root_logger.handlers if h.name == HANDLER_NAME]
    assert len(ours) == 1
    assert clean_root_logger.level == logging.DEBUG


def test_set_seed_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)


@pytest.mark.slow
def test_set_seed_repeats_python_numpy_and_tf_draws() -> None:
    import tensorflow as tf

    def draw() -> tuple[float, float, float]:
        return random.random(), float(np.random.rand()), float(tf.random.uniform(()))

    assert set_seed(7) is False
    first = draw()
    set_seed(7)
    assert draw() == first


@pytest.mark.slow
def test_set_seed_repeats_keras_weight_init() -> None:
    # Keras 3 layers draw initial weights from Keras's own seed generator,
    # which seeding TensorFlow alone does not reset.
    from retinal_vessels.config import ModelConfig
    from retinal_vessels.model import build_unet

    cfg = ModelConfig(depth=1, base_filters=2, dropout=0.0, batch_norm=False)
    set_seed(11)
    first = build_unet(cfg).get_weights()
    set_seed(11)
    for a, b in zip(build_unet(cfg).get_weights(), first, strict=True):
        np.testing.assert_array_equal(a, b)
