import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_built_wheel_ships_and_uses_the_schema(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path / "dist"), str(REPO)],
        check=True,
        capture_output=True,
    )
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as zf:
        packaged = zf.read("retinal_vessels/schema.sql")
        zf.extractall(tmp_path / "site")
    assert packaged == (REPO / "sql" / "schema.sql").read_bytes()

    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site")}
    out = subprocess.run(
        [sys.executable, "-c", "from retinal_vessels.db import SCHEMA_PATH; print(SCHEMA_PATH)"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    ).stdout.strip()
    assert Path(out) == (tmp_path / "site" / "retinal_vessels" / "schema.sql").resolve()
