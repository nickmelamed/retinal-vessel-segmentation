import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from retinal_vessels.db import connect, create_schema
from tests.fixtures.synthetic_drive import write_synthetic_drive


@pytest.fixture
def synthetic_data_root(tmp_path: Path) -> Path:
    """A data root holding a full synthetic ``DRIVE/`` tree, like ``data/``."""
    root = tmp_path / "data"
    write_synthetic_drive(root)
    return root


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """An empty experiments database at the current schema version."""
    c = connect(tmp_path / "experiments.db")
    create_schema(c)
    yield c
    c.close()
