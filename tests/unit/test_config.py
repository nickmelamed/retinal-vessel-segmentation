from pathlib import Path
from typing import Any

import pytest
import yaml

from retinal_vessels.config import Config, ConfigError, load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


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


@pytest.mark.parametrize(("section", "key"), [(None, "seed"), ("patches", "size")])
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
    ],
)
def test_rejects_bad_value(
    tmp_path: Path, baseline: dict[str, Any], section: str, key: str, value: Any
) -> None:
    baseline[section][key] = value
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, baseline))


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
