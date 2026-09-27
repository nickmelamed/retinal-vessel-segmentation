"""A small U-Net for vessel segmentation (SPEC section 7)."""

from typing import Any

from retinal_vessels.config import ModelConfig

KERNEL = 3
POOL = 2
CHANNELS = 1


def _conv_block(x: Any, filters: int, cfg: ModelConfig) -> Any:
    import keras

    for _ in range(2):
        # The bias is redundant when batch norm follows, since its shift absorbs it.
        x = keras.layers.Conv2D(filters, KERNEL, padding="same", use_bias=not cfg.batch_norm)(x)
        if cfg.batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
    return x


def build_unet(cfg: ModelConfig) -> Any:
    """Return an uncompiled U-Net mapping (batch, H, W, 1) to vessel probabilities.

    H and W can be any multiple of ``2**cfg.depth``, so one model serves
    training patches and inference windows. Filters double at each level from
    ``cfg.base_filters``. Upsampling uses transposed convolutions, whose GPU
    kernels are deterministic, unlike bilinear upsampling gradients.
    """
    import keras

    inputs = keras.Input(shape=(None, None, CHANNELS))
    x = inputs
    skips = []
    for level in range(cfg.depth):
        x = _conv_block(x, cfg.base_filters * 2**level, cfg)
        skips.append(x)
        x = keras.layers.MaxPooling2D(POOL)(x)
    x = _conv_block(x, cfg.base_filters * 2**cfg.depth, cfg)
    if cfg.dropout:
        x = keras.layers.Dropout(cfg.dropout)(x)
    for level in reversed(range(cfg.depth)):
        filters = cfg.base_filters * 2**level
        x = keras.layers.Conv2DTranspose(filters, POOL, strides=POOL)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        if cfg.dropout:
            x = keras.layers.Dropout(cfg.dropout)(x)
        x = _conv_block(x, filters, cfg)
    outputs = keras.layers.Conv2D(CHANNELS, 1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="unet")
