import importlib.util
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest

from retinal_vessels.db import create_schema
from retinal_vessels.provenance import sha256_file
from tests.fixtures.database import RUN, insert

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_checkpoints.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_checkpoints", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load_script()


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


def checkpoint(models: Path, run_id: str, fold: int, content: bytes) -> str:
    path = models / run_id / f"fold_{fold}.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_file(path)


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path]:
    db, models = tmp_path / "experiments.db", tmp_path / "models"
    conn = sqlite3.connect(db)
    create_schema(conn)
    for run_id in ("r1", "r2"):
        insert(conn, "runs", {**RUN, "run_id": run_id})
    for fold in (1, 2):
        sha = checkpoint(models, "r2", fold, f"weights {fold}".encode())
        insert(
            conn,
            "fold_status",
            {
                "run_id": "r2",
                "fold": fold,
                "status": "complete",
                "checkpoint_sha256": sha,
                "started_at": RUN["started_at"],
                "completed_at": RUN["started_at"],
            },
        )
    insert(
        conn,
        "fold_status",
        {"run_id": "r2", "fold": 3, "status": "running", "started_at": RUN["started_at"]},
    )
    conn.commit()
    conn.close()
    return db, models


def cli(db: Path, models: Path, *extra: str) -> int:
    code: int = verify.main(["--db", str(db), "--models-dir", str(models), *extra])
    return code


def test_matching_checkpoints_pass(setup: tuple[Path, Path]) -> None:
    assert cli(*setup) == 0
    assert cli(*setup, "--run-id", "r2") == 0


def test_a_changed_checkpoint_fails(setup: tuple[Path, Path]) -> None:
    db, models = setup
    (models / "r2" / "fold_2.keras").write_bytes(b"other weights")
    assert cli(db, models) == 1


def test_a_missing_checkpoint_fails(setup: tuple[Path, Path]) -> None:
    db, models = setup
    (models / "r2" / "fold_1.keras").unlink()
    assert cli(db, models) == 1


def test_a_run_without_complete_folds_fails(setup: tuple[Path, Path]) -> None:
    assert cli(*setup, "--run-id", "r1") == 1


def test_missing_or_empty_database_fails(tmp_path: Path) -> None:
    assert cli(tmp_path / "absent.db", tmp_path) == 1
    empty = tmp_path / "empty.db"
    with closing(sqlite3.connect(empty)) as conn:
        create_schema(conn)
    assert cli(empty, tmp_path) == 1
