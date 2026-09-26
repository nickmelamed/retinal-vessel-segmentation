"""Open the experiments database and create its schema from ``sql/schema.sql``."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced.

    SQLite leaves foreign keys off by default, per connection, so every
    connection must go through here.
    """
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
