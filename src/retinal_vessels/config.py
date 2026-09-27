"""Load an experiment config from YAML into typed, validated objects.

Every key is required and unknown keys are errors, so a typo or a missing
setting stops the run instead of falling back to a default. Types are strict:
``true`` is not a number and ``1`` is not a boolean.
"""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """A config file that is missing, unreadable, or fails validation."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FoldsConfig(_Section):
    """Cross-validation split sizes (SPEC section 5)."""

    n_folds: int = Field(ge=2)
    n_val: int = Field(ge=1)


class PreprocessConfig(_Section):
    """Preprocessing steps, each of which can be switched off for ablations."""

    green_channel: bool
    clahe: bool
    clahe_clip_limit: float = Field(gt=0)
    clahe_tile_grid: tuple[int, int]
    scale_unit: bool
    standardize_in_fov: bool

    @model_validator(mode="after")
    def _positive_tiles(self) -> "PreprocessConfig":
        if min(self.clahe_tile_grid) < 1:
            raise ValueError(f"clahe_tile_grid must be positive, got {self.clahe_tile_grid}")
        return self


class PatchesConfig(_Section):
    """Training patch sampling and augmentation."""

    size: int = Field(ge=2, multiple_of=2)
    per_epoch: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    flip: bool
    rot90: bool
    # Added to the preprocessed image, which is standardized when that step is on.
    brightness_delta: float = Field(ge=0)
    contrast_range: tuple[float, float]

    @model_validator(mode="after")
    def _ordered_contrast(self) -> "PatchesConfig":
        low, high = self.contrast_range
        if not 0 < low <= high:
            raise ValueError(f"contrast_range must satisfy 0 < low <= high, got {low}, {high}")
        return self


class Config(_Section):
    """One experiment variant, as read from ``configs/<variant>.yaml``."""

    variant: str = Field(min_length=1)
    seed: int = Field(ge=0)
    folds: FoldsConfig
    preprocess: PreprocessConfig
    patches: PatchesConfig

    def as_dict(self) -> dict[str, Any]:
        """Return the config as plain JSON types, for hashing and the run record."""
        return self.model_dump(mode="json")

    def as_json(self) -> str:
        """Return the config as canonical JSON text for ``runs.config``."""
        return json.dumps(self.as_dict(), sort_keys=True)


def load_config(path: Path) -> Config:
    """Read and validate ``path``.

    Any problem raises ``ConfigError`` naming the file and every bad key.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(f"cannot read config {path}: {err}") from err
    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} must be a mapping, got {type(raw).__name__}")
    try:
        # Validating from JSON keeps strict types while accepting YAML lists as tuples.
        return Config.model_validate_json(json.dumps(raw))
    except TypeError as err:
        raise ConfigError(f"config {path} holds a value that is not plain data: {err}") from err
    except ValidationError as err:
        raise ConfigError(f"invalid config {path}:\n{err}") from err
