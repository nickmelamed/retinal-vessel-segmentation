from pathlib import Path

import pytest

from tests.fixtures.synthetic_drive import write_synthetic_drive


@pytest.fixture
def synthetic_data_root(tmp_path: Path) -> Path:
    """A data root holding a full synthetic ``DRIVE/`` tree, like ``data/``."""
    root = tmp_path / "data"
    write_synthetic_drive(root)
    return root
