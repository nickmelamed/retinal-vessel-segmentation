import importlib.util
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "make_social_preview.py"
LINEAGE = {"figure": "hero.png", "run_id": "r1", "tag": "v0.1.0-rc.1", "commit": "abc"}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_social_preview", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


@pytest.fixture
def hero(tmp_path: Path) -> Path:
    path = tmp_path / "hero.png"
    # A smooth gradient compresses like a real figure. Random noise would not,
    # and would trip the size limit.
    ramp = np.linspace(0, 255, 1200, dtype=np.float64)
    rgb = np.broadcast_to(ramp[None, :, None], (390, 1200, 3)).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    path.with_suffix(".json").write_text(json.dumps(LINEAGE))
    return path


def test_preview_is_1280_by_640(hero: Path, tmp_path: Path) -> None:
    out = tmp_path / "social_preview.png"
    assert load_script().main(["--hero", str(hero), "--out", str(out)]) == 0
    with Image.open(out) as img:
        assert img.size == (1280, 640)


def test_rebuilding_gives_the_same_bytes(hero: Path, tmp_path: Path) -> None:
    script = load_script()
    for name in ("a.png", "b.png"):
        assert script.main(["--hero", str(hero), "--out", str(tmp_path / name)]) == 0
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()


def test_needs_the_hero_figure(tmp_path: Path) -> None:
    out = tmp_path / "social_preview.png"
    assert load_script().main(["--hero", str(tmp_path / "absent.png"), "--out", str(out)]) == 1
    assert not out.exists()


def test_sidecar_carries_the_hero_lineage(hero: Path, tmp_path: Path) -> None:
    out = tmp_path / "social_preview.png"
    assert load_script().main(["--hero", str(hero), "--out", str(out)]) == 0
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["figure"] == "social_preview.png"
    assert sidecar["built_from"] == "hero.png"
    assert {k: sidecar[k] for k in ("run_id", "tag", "commit")} == {
        "run_id": "r1",
        "tag": "v0.1.0-rc.1",
        "commit": "abc",
    }


def test_needs_the_hero_sidecar(hero: Path, tmp_path: Path) -> None:
    hero.with_suffix(".json").unlink()
    out = tmp_path / "social_preview.png"
    assert load_script().main(["--hero", str(hero), "--out", str(out)]) == 1
    assert not out.exists()


def test_refuses_a_preview_over_the_size_limit(hero: Path, tmp_path: Path) -> None:
    config = tmp_path / "reporting.yaml"
    text = (REPO / "configs" / "reporting.yaml").read_text()
    config.write_text(text.replace("max_bytes: 1048576", "max_bytes: 1000"))
    out = tmp_path / "social_preview.png"
    args = ["--hero", str(hero), "--out", str(out), "--config", str(config)]
    assert load_script().main(args) == 1
    assert not out.exists()
