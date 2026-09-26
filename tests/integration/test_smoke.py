"""End-to-end run on synthetic data.

In phase 0 the pipeline is only wired, not trained: synthetic data is written,
its checksums are recorded and verified through the real CLI, the database is
created, and a run manifest is written. Later phases extend this same test to
train, predict, evaluate, and build tables.
"""

import json
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from retinal_vessels.db import SCHEMA_VERSION, connect, create_schema, schema_version
from retinal_vessels.provenance import build_manifest, new_run_id, read_checksums, write_manifest

REPO = Path(__file__).resolve().parents[2]
CHECK_DATA = REPO / "scripts" / "check_data.py"

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def _check_data(data_root: Path, drive_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "DRIVE_DIR"}
    if drive_dir is not None:
        env["DRIVE_DIR"] = str(drive_dir)
    return subprocess.run(
        [sys.executable, str(CHECK_DATA), "--data-root", str(data_root)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_pipeline_is_wired_end_to_end(synthetic_data_root: Path, tmp_path: Path) -> None:
    manifest_path = synthetic_data_root / "CHECKSUMS.sha256"

    first = _check_data(synthetic_data_root)
    assert first.returncode == 0, first.stderr
    assert "Wrote 100 files" in first.stderr
    written = manifest_path.read_bytes()

    second = _check_data(synthetic_data_root)
    assert second.returncode == 0, second.stderr
    assert "Verified 100 files" in second.stderr
    assert manifest_path.read_bytes() == written

    moved = tmp_path / "elsewhere" / "DRIVE"
    moved.parent.mkdir()
    (synthetic_data_root / "DRIVE").rename(moved)
    assert _check_data(synthetic_data_root).returncode == 1
    assert _check_data(synthetic_data_root, drive_dir=moved).returncode == 0

    (moved / "test" / "images" / "05_test.tif").write_bytes(b"corrupt")
    corrupt = _check_data(synthetic_data_root, drive_dir=moved)
    assert corrupt.returncode == 1
    assert "checksum mismatch: DRIVE/test/images/05_test.tif" in corrupt.stderr
    assert manifest_path.read_bytes() == written

    with closing(connect(tmp_path / "results" / "experiments.db")) as conn:
        create_schema(conn)
        assert schema_version(conn) == SCHEMA_VERSION

    manifest = build_manifest(
        run_id=new_run_id(),
        variant="smoke",
        config={"variant": "smoke"},
        seed=0,
        deterministic_ops=False,
        repo=REPO,
        checksums=read_checksums(manifest_path),
    )
    saved = json.loads(write_manifest(manifest, tmp_path / "results").read_text())
    assert saved["variant"] == "smoke"
    assert len(saved["git_commit"]) == 40
    assert saved["environment"]["tensorflow_version"]
