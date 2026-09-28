"""Load an experiment config from YAML into typed, validated objects.

Every key is required and unknown keys are errors, so a typo or a missing
setting stops the run instead of falling back to a default. Types are strict:
``true`` is not a number and ``1`` is not a boolean.
"""

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """A config file that is missing, unreadable, or fails validation."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FoldsConfig(_Section):
    """Cross-validation split sizes and the seed that draws the split (SPEC section 5)."""

    # Kept apart from the top-level seed so that every variant, and every
    # rerun with a new training seed, uses the same folds (D-019).
    seed: int = Field(ge=0)
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
    # Chance of flipping each patch, drawn separately for rows and columns.
    # 0 switches flips off.
    flip_probability: float = Field(ge=0, le=1)
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


class ModelConfig(_Section):
    """U-Net size (SPEC section 7)."""

    # Number of downsampling steps. Patch and window sizes must divide by 2**depth.
    depth: int = Field(ge=1)
    base_filters: int = Field(ge=1)
    dropout: float = Field(ge=0, lt=1)
    batch_norm: bool


class LossConfig(_Section):
    """Training loss: BCE + Dice for the baseline, Dice alone for the ablation."""

    name: Literal["bce_dice", "dice"]
    # Added to the numerator and denominator of soft Dice so an all-background
    # patch has a defined loss.
    smooth: float = Field(gt=0)


class TrainingConfig(_Section):
    """Optimizer and early stopping (SPEC section 7)."""

    max_epochs: int = Field(ge=1)
    # Epochs without a better validation Dice before training stops.
    patience: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    deterministic_ops: bool


class InferenceConfig(_Section):
    """Sliding-window inference over whole images."""

    window: int = Field(ge=2)
    stride: int = Field(ge=1)
    batch_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _overlapping(self) -> "InferenceConfig":
        if self.stride > self.window:
            raise ValueError(
                f"stride {self.stride} exceeds window {self.window}, which leaves gaps"
            )
        return self


class ThresholdConfig(_Section):
    """Candidate thresholds for the validation Dice sweep (SPEC section 5)."""

    # Candidates are i / divisions for i = 1 .. divisions - 1. An integer
    # count keeps the grid exact, where a float step would drift.
    divisions: int = Field(ge=2)


class EvaluationConfig(_Section):
    """Settings for the metrics computed from saved predictions (SPEC section 5, D-021)."""

    # Equal-width probability bins of the pooled reliability diagram.
    reliability_bins: int = Field(ge=2)
    # Each fold's thin/thick edge is this quantile of skeleton radius over
    # its training and validation labels.
    thin_quantile: float = Field(gt=0, lt=1)


class Config(_Section):
    """One experiment variant, as read from ``configs/<variant>.yaml``."""

    variant: str = Field(min_length=1)
    seed: int = Field(ge=0)
    folds: FoldsConfig
    preprocess: PreprocessConfig
    patches: PatchesConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    inference: InferenceConfig
    threshold: ThresholdConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def _sizes_fit_the_model(self) -> "Config":
        # Each U-Net level halves the input, and the skip connections need
        # every level to divide evenly.
        factor = 2**self.model.depth
        for name, size in (
            ("patches.size", self.patches.size),
            ("inference.window", self.inference.window),
        ):
            if size % factor:
                raise ValueError(f"{name}={size} must be divisible by 2**model.depth={factor}")
        return self

    def as_dict(self) -> dict[str, Any]:
        """Return the config as plain JSON types, for hashing and the run record."""
        return self.model_dump(mode="json")

    def as_json(self) -> str:
        """Return the config as canonical JSON text for ``runs.config``."""
        return json.dumps(self.as_dict(), sort_keys=True)


HEX_COLOR = r"^#[0-9a-f]{6}$"


class ReportedRunsConfig(_Section):
    """Hardware every reported run must share (SPEC section 16, D-013)."""

    gpu_type: str = Field(min_length=1)
    compute_platform: str = Field(min_length=1)


class TablesConfig(_Section):
    """Number formatting and bootstrap intervals for ``results/tables``."""

    decimals: int = Field(ge=1)
    bootstrap_resamples: int = Field(ge=1)
    bootstrap_seed: int = Field(ge=0)
    ci_level: float = Field(gt=0, lt=1)


class FiguresConfig(_Section):
    """Resolution, layout, and colors of the committed figures (SPEC section 11)."""

    dpi: int = Field(ge=1)
    n_best_worst: int = Field(ge=1)
    # Figures are committed, and the pre-commit hook rejects larger files.
    max_bytes: int = Field(ge=1)
    surface: str = Field(pattern=HEX_COLOR)
    ink: str = Field(pattern=HEX_COLOR)
    secondary_ink: str = Field(pattern=HEX_COLOR)
    muted_ink: str = Field(pattern=HEX_COLOR)
    gridline: str = Field(pattern=HEX_COLOR)
    series: str = Field(pattern=HEX_COLOR)
    true_positive: str = Field(pattern=HEX_COLOR)
    false_positive: str = Field(pattern=HEX_COLOR)
    false_negative: str = Field(pattern=HEX_COLOR)

    @model_validator(mode="after")
    def _distinct_error_colors(self) -> "FiguresConfig":
        # The error map is unreadable if two outcomes share a color, or one
        # matches the background.
        colors = [self.surface, self.true_positive, self.false_positive, self.false_negative]
        if len(set(colors)) != len(colors):
            raise ValueError(f"surface and error map colors must all differ, got {colors}")
        return self


class ReportingConfig(_Section):
    """Settings for presenting reported results, read from ``configs/reporting.yaml``.

    Kept out of the variant configs, since each run stores its variant config
    and presentation settings must be able to change without touching that record.
    """

    reported_runs: ReportedRunsConfig
    tables: TablesConfig
    figures: FiguresConfig


def _load[M: BaseModel](path: Path, model: type[M]) -> M:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(f"cannot read config {path}: {err}") from err
    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} must be a mapping, got {type(raw).__name__}")
    try:
        # Validating from JSON keeps strict types while accepting YAML lists as tuples.
        return model.model_validate_json(json.dumps(raw))
    except TypeError as err:
        raise ConfigError(f"config {path} holds a value that is not plain data: {err}") from err
    except ValidationError as err:
        raise ConfigError(f"invalid config {path}:\n{err}") from err


def load_config(path: Path) -> Config:
    """Read and validate the variant config at ``path``.

    Any problem raises ``ConfigError`` naming the file and every bad key.
    """
    return _load(path, Config)


def load_reporting_config(path: Path) -> ReportingConfig:
    """Read and validate the reporting config at ``path``, like :func:`load_config`."""
    return _load(path, ReportingConfig)
