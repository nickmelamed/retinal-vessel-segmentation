"""Open the experiments database, create its schema, write rows, and run queries."""

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from retinal_vessels.data import Fold, Sample, fold_rows

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SCHEMA_NAME = "schema.sql"
QUERIES_NAME = "queries"


def _sql_path(name: str) -> Path:
    # A built wheel carries copies of sql/schema.sql and sql/queries inside the
    # package (see pyproject.toml). An editable install has no copies and reads
    # the repo's files.
    packaged = Path(__file__).resolve().parent / name
    if packaged.exists():
        return packaged
    return Path(__file__).resolve().parents[2] / "sql" / name


SCHEMA_PATH = _sql_path(SCHEMA_NAME)
QUERIES_DIR = _sql_path(QUERIES_NAME)


def connect(path: Path) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced, creating the parent directory.

    SQLite leaves foreign keys off by default, per connection, so every
    connection must go through here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn: sqlite3.Connection) -> int | None:
    """Return the highest applied schema version, or None for an empty database."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if exists is None:
        return None
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return None if row is None or row[0] is None else int(row[0])


def create_schema(conn: sqlite3.Connection, schema_path: Path = SCHEMA_PATH) -> None:
    """Create any missing tables, then check the database is at ``SCHEMA_VERSION``.

    Safe to call on every run. A database at another version raises
    ``RuntimeError`` instead of being altered, since there are no migrations yet.
    """
    if not schema_path.is_file():
        raise FileNotFoundError(f"schema not found at {schema_path}")
    found = schema_version(conn)
    if found is not None and found != SCHEMA_VERSION:
        raise RuntimeError(f"database is at schema version {found}, code expects {SCHEMA_VERSION}")
    with conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
    found = schema_version(conn)
    if found != SCHEMA_VERSION:
        raise RuntimeError(f"{schema_path} sets version {found}, code expects {SCHEMA_VERSION}")
    logger.debug("Schema at version %d", SCHEMA_VERSION)


def run_query(
    conn: sqlite3.Connection, name: str, queries_dir: Path = QUERIES_DIR
) -> list[tuple[object, ...]]:
    """Run ``sql/queries/<name>.sql`` and return its rows.

    ``name`` is the file stem, like ``04_leakage_audit``. A missing file
    raises ``FileNotFoundError``.
    """
    path = queries_dir / f"{name}.sql"
    if not path.is_file():
        raise FileNotFoundError(f"query not found at {path}")
    return [tuple(row) for row in conn.execute(path.read_text(encoding="utf-8")).fetchall()]


@dataclass(frozen=True)
class ImageRecord:
    """One row of the ``images`` table. Field order matches the columns."""

    dataset: str
    image_id: str
    patient_id: str | None
    split: str
    has_abnormality: int
    abnormality_note: str | None
    fov_pixels: int
    fov_source: str
    vessel_fraction_in_fov: float | None
    has_labels: int


IMAGE_COLUMNS = tuple(f.name for f in fields(ImageRecord))


def image_record(sample: Sample, fov_source: str = "official") -> ImageRecord:
    """Summarize a sample as its ``images`` row.

    The vessel fraction counts labeled pixels inside the FOV only, since some
    DRIVE labels mark a few pixels outside it. It is None without labels.
    """
    fov_pixels = int(sample.fov.sum())
    fraction = None
    if sample.label is not None:
        fraction = float((sample.label & sample.fov).sum()) / fov_pixels
    return ImageRecord(
        dataset=sample.dataset,
        image_id=sample.image_id,
        patient_id=sample.patient_id,
        split=sample.split,
        has_abnormality=int(sample.has_abnormality),
        abnormality_note=sample.abnormality_note,
        fov_pixels=fov_pixels,
        fov_source=fov_source,
        vessel_fraction_in_fov=fraction,
        has_labels=int(sample.label is not None),
    )


def write_images(conn: sqlite3.Connection, records: Iterable[ImageRecord]) -> int:
    """Insert image rows and return how many were new.

    A row already present with identical values is left alone, so this is
    safe to repeat. A row present with different values raises
    ``ValueError``, since that means the data on disk changed.
    """
    cols = ", ".join(IMAGE_COLUMNS)
    marks = ", ".join("?" for _ in IMAGE_COLUMNS)
    added = 0
    with conn:
        for record in records:
            existing = conn.execute(
                f"SELECT {cols} FROM images WHERE dataset = ? AND image_id = ?",
                (record.dataset, record.image_id),
            ).fetchone()
            if existing is None:
                conn.execute(f"INSERT INTO images ({cols}) VALUES ({marks})", astuple(record))
                added += 1
            elif tuple(existing) != astuple(record):
                raise ValueError(
                    f"images row for {record.dataset} {record.image_id} differs from the data: "
                    f"stored {tuple(existing)}, computed {astuple(record)}"
                )
    return added


def write_fold_assignments(
    conn: sqlite3.Connection, run_id: str, dataset: str, folds: Iterable[Fold]
) -> None:
    """Record every image's role in every fold of ``run_id``, in one transaction."""
    rows = [(run_id, fold, dataset, image_id, role) for fold, image_id, role in fold_rows(folds)]
    with conn:
        conn.executemany(
            "INSERT INTO fold_assignments (run_id, fold, dataset, image_id, role) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
