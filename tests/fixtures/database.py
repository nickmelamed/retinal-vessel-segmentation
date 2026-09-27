"""Rows and helpers for tests that need a populated experiments database."""

import sqlite3
from typing import Any

RUN: dict[str, Any] = {
    "run_id": "r1",
    "variant": "baseline",
    "config": "{}",
    "config_hash": "c",
    "seed": 0,
    "git_commit": "abc",
    "git_dirty": 0,
    "data_hash": "d",
    "python_version": "3.13",
    "tensorflow_version": "2.21",
    "device": "cpu",
    "gpu_type": "cpu",
    "compute_platform": "local",
    "deterministic_ops": 0,
    "started_at": "2026-09-26T00:00:00Z",
}


def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
