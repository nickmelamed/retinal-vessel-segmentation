from pathlib import Path
from typing import Any

import pytest
import yaml

from retinal_vessels.config import Config, ConfigError, load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
NON_CV_CONFIGS = {"final", "external"}


def write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def baseline() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((CONFIGS / "baseline.yaml").read_text())
    return loaded


@pytest.mark.parametrize("name", ["baseline", "smoke"])
def test_shipped_configs_load(name: str) -> None:
    config = load_config(CONFIGS / f"{name}.yaml")
    assert config.variant == name
    assert config.preprocess.clahe_tile_grid == (8, 8)


def test_every_cv_variant_shares_one_fold_seed() -> None:
    # SPEC section 7 pairs per-image results across variants, which only
    # works if every variant trains and tests on the same folds. The frozen
    # model and external validation configs have no folds (section 5).
    paths = [p for p in sorted(CONFIGS.glob("*.yaml")) if p.stem not in NON_CV_CONFIGS]
    configs = [load_config(path) for path in paths]
    assert len(configs) >= 2
    assert len({c.folds.seed for c in configs}) == 1
    assert len({(c.folds.n_folds, c.folds.n_val) for c in configs}) == 1


def test_round_trips_through_json(tmp_path: Path, baseline: dict[str, Any]) -> None:
    config = load_config(write(tmp_path, baseline))
    assert Config.model_validate_json(config.as_json()) == config
    assert config.as_dict()["patches"]["contrast_range"] == [0.9, 1.1]


def test_rejects_unknown_top_level_key(tmp_path: Path, baseline: dict[str, Any]) -> None:
    baseline["learning_rate"] = 0.001
    with pytest.raises(ConfigError, match="learning_rate"):
        load_config(write(tmp_path, baseline))


def test_rejects_unknown_nested_key(tmp_path: Path, baseline: dict[str, Any]) -> None:
    baseline["preprocess"]["clahe_clip"] = 2.0
    with pytest.raises(ConfigError, match="clahe_clip"):
        load_config(write(tmp_path, baseline))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "seed"),
        ("folds", "seed"),
        ("patches", "size"),
        (None, "model"),
        ("model", "depth"),
        ("loss", "name"),
        ("training", "patience"),
        ("training", "deterministic_ops"),
        ("inference", "stride"),
        ("threshold", "divisions"),
    ],
)
def test_rejects_missing_key(
    tmp_path: Path, baseline: dict[str, Any], section: str | None, key: str
) -> None:
    del (baseline if section is None else baseline[section])[key]
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, baseline))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("preprocess", "clahe", 1),
        ("preprocess", "clahe_clip_limit", True),
        ("patches", "size", 63),
        ("patches", "size", "64"),
        ("preprocess", "clahe_tile_grid", [8, 0]),
        ("patches", "contrast_range", [1.1, 0.9]),
        ("folds", "n_val", 0),
        ("folds", "seed", -1),
        ("patches", "flip_probability", 1.5),
        ("patches", "flip_probability", -0.1),
        ("patches", "flip_probability", True),
        ("model", "depth", 0),
        ("model", "dropout", 1.0),
        ("model", "batch_norm", "yes"),
        ("loss", "name", "focal"),
        ("loss", "smooth", 0),
        ("training", "max_epochs", 0),
        ("training", "learning_rate", 0),
        ("inference", "stride", 256),
        ("threshold", "divisions", 1),
    ],
)
def test_rejects_bad_value(
    tmp_path: Path, baseline: dict[str, Any], section: str, key: str, value: Any
) -> None:
    baseline[section][key] = value
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, baseline))


@pytest.mark.parametrize(("section", "key"), [("patches", "size"), ("inference", "window")])
def test_rejects_size_the_unet_cannot_halve(
    tmp_path: Path, baseline: dict[str, Any], section: str, key: str
) -> None:
    # Depth 4 needs sizes divisible by 16. 72 is even but not a multiple of 16.
    baseline[section][key] = 72
    if key == "window":
        baseline["inference"]["stride"] = 72
    with pytest.raises(ConfigError, match=rf"{section}\.{key}=72"):
        load_config(write(tmp_path, baseline))


def test_accepts_deeper_model_only_with_larger_sizes(
    tmp_path: Path, baseline: dict[str, Any]
) -> None:
    baseline["model"]["depth"] = 7
    with pytest.raises(ConfigError, match="2\\*\\*model.depth=128"):
        load_config(write(tmp_path, baseline))
    baseline["patches"]["size"] = 128
    assert load_config(write(tmp_path, baseline)).model.depth == 7


def test_rejects_negative_seed(tmp_path: Path, baseline: dict[str, Any]) -> None:
    baseline["seed"] = -1
    with pytest.raises(ConfigError, match="seed"):
        load_config(write(tmp_path, baseline))


@pytest.mark.parametrize("text", ["", "- a\n- b\n"])
def test_rejects_non_mapping(tmp_path: Path, text: str) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


def test_rejects_non_json_values(tmp_path: Path, baseline: dict[str, Any]) -> None:
    path = write(tmp_path, baseline)
    path.write_text(path.read_text() + "started: 2026-09-26\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not plain data"):
        load_config(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "absent.yaml")


def test_config_is_immutable() -> None:
    config = load_config(CONFIGS / "baseline.yaml")
    with pytest.raises(ValueError, match="frozen"):
        config.seed = 1  # type: ignore[misc]
