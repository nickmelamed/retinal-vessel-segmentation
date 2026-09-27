"""Open the experiments database, create its schema, write rows, and run queries."""

import json
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from retinal_vessels.data import Fold, Sample, fold_rows
from retinal_vessels.metrics import BinaryMetrics
from retinal_vessels.provenance import RunManifest

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


RUN_COLUMNS = (
    "run_id",
    "variant",
    "config",
    "config_hash",
    "seed",
    "git_commit",
    "git_dirty",
    "git_tag",
    "data_hash",
    "python_version",
    "tensorflow_version",
    "cuda_version",
    "device",
    "gpu_type",
    "compute_platform",
    "deterministic_ops",
    "started_at",
)


def insert_run(conn: sqlite3.Connection, manifest: RunManifest) -> None:
    """Write the ``runs`` row for a run that is starting.

    The config is stored as canonical JSON, the same text as ``Config.as_json``.
    ``is_reported`` stays 0. Marking a run as reported is a separate, later step.
    """
    env = manifest.environment
    values = (
        manifest.run_id,
        manifest.variant,
        json.dumps(manifest.config, sort_keys=True),
        manifest.config_hash,
        manifest.seed,
        manifest.git_commit,
        int(manifest.git_dirty),
        manifest.git_tag,
        manifest.data_hash,
        env["python_version"],
        env["tensorflow_version"],
        env.get("cuda_version"),
        env["device"],
        env["gpu_type"],
        env["compute_platform"],
        int(manifest.deterministic_ops),
        manifest.started_at,
    )
    cols = ", ".join(RUN_COLUMNS)
    marks = ", ".join("?" for _ in RUN_COLUMNS)
    with conn:
        conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", values)


ENV_COLUMNS = (
    "python_version",
    "tensorflow_version",
    "cuda_version",
    "device",
    "gpu_type",
    "compute_platform",
)


def run_manifest(conn: sqlite3.Connection, run_id: str) -> RunManifest:
    """Rebuild a run's manifest from its ``runs`` row.

    A resumed run may be on a fresh machine without the original
    ``manifest.json``, so the database row is the source of truth for it.
    An unknown run id raises ``KeyError``.
    """
    cols = (
        "variant",
        "config",
        "config_hash",
        "seed",
        "git_commit",
        "git_dirty",
        "git_tag",
        "data_hash",
        "deterministic_ops",
        "started_at",
        "finished_at",
        *ENV_COLUMNS,
    )
    row = conn.execute(f"SELECT {', '.join(cols)} FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"no run {run_id!r} in the database")
    v = dict(zip(cols, row, strict=True))
    return RunManifest(
        run_id=run_id,
        variant=v["variant"],
        config=json.loads(v["config"]),
        config_hash=v["config_hash"],
        seed=v["seed"],
        git_commit=v["git_commit"],
        git_dirty=bool(v["git_dirty"]),
        git_tag=v["git_tag"],
        data_hash=v["data_hash"],
        deterministic_ops=bool(v["deterministic_ops"]),
        started_at=v["started_at"],
        environment={c: v[c] for c in ENV_COLUMNS},
        finished_at=v["finished_at"],
    )


def find_resumable_run(
    conn: sqlite3.Connection, variant: str, config_hash: str, git_commit: str, data_hash: str
) -> str | None:
    """Return the unfinished clean-tree run with this variant, config, commit, and data.

    A run matching on all four trained with the same code on the same data,
    so its completed folds can stand. A run started on a dirty tree never
    matches, since its code cannot be recovered from the commit. More than one
    match raises ``RuntimeError``, since it is unclear which to continue.
    """
    rows = conn.execute(
        "SELECT run_id FROM runs WHERE finished_at IS NULL AND git_dirty = 0 AND variant = ? "
        "AND config_hash = ? AND git_commit = ? AND data_hash = ? ORDER BY run_id",
        (variant, config_hash, git_commit, data_hash),
    ).fetchall()
    if len(rows) > 1:
        ids = [r[0] for r in rows]
        raise RuntimeError(
            f"several unfinished runs match: {ids}. Pass --new to start a fresh run."
        )
    return None if not rows else str(rows[0][0])


def stored_fold_rows(conn: sqlite3.Connection, run_id: str) -> list[tuple[int, str, str]]:
    """Return ``(fold, image_id, role)`` for ``run_id``, sorted."""
    rows = conn.execute(
        "SELECT fold, image_id, role FROM fold_assignments WHERE run_id = ? "
        "ORDER BY fold, image_id, role",
        (run_id,),
    ).fetchall()
    return [(int(f), str(i), str(r)) for f, i, r in rows]


def fold_statuses(conn: sqlite3.Connection, run_id: str) -> dict[int, str]:
    """Return ``{fold: status}`` for every fold of ``run_id`` that has started."""
    rows = conn.execute(
        "SELECT fold, status FROM fold_status WHERE run_id = ?", (run_id,)
    ).fetchall()
    return {int(f): str(s) for f, s in rows}


FOLD_TABLES = ("thresholds", "per_image_metrics", "training_history")


def start_fold(conn: sqlite3.Connection, run_id: str, fold: int, started_at: str) -> int:
    """Mark ``fold`` as running and return its attempt number.

    A fold left ``running`` by an interrupted attempt has its partial rows
    in the fold tables deleted and its attempt count raised, so the retrained
    fold starts clean. Starting a complete fold raises ``ValueError``.
    """
    with conn:
        row = conn.execute(
            "SELECT status, attempts FROM fold_status WHERE run_id = ? AND fold = ?",
            (run_id, fold),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO fold_status (run_id, fold, status, started_at) "
                "VALUES (?, ?, 'running', ?)",
                (run_id, fold, started_at),
            )
            return 1
        status, attempts = row
        if status == "complete":
            raise ValueError(f"fold {fold} of run {run_id} is already complete")
        for table in FOLD_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE run_id = ? AND fold = ?", (run_id, fold))
        conn.execute(
            "UPDATE fold_status SET attempts = ?, started_at = ? WHERE run_id = ? AND fold = ?",
            (attempts + 1, started_at, run_id, fold),
        )
        return int(attempts) + 1


@dataclass(frozen=True)
class EpochRecord:
    """One ``training_history`` row. Epochs are numbered from 1."""

    epoch: int
    train_loss: float
    val_dice: float


@dataclass(frozen=True)
class ImageMetrics:
    """Single-pass metrics of one held-out image at its fold's threshold."""

    dataset: str
    image_id: str
    metrics: BinaryMetrics


@dataclass(frozen=True)
class FoldRecord:
    """Everything a finished fold writes before it counts as complete."""

    run_id: str
    fold: int
    checkpoint_sha256: str
    threshold: float
    selection_rule: str
    val_dice: float
    history: Sequence[EpochRecord]
    test_metrics: Sequence[ImageMetrics]


def complete_fold(conn: sqlite3.Connection, record: FoldRecord, completed_at: str) -> None:
    """Write a fold's threshold, test metrics, and history, and mark it complete.

    Everything goes in one transaction, so a crash leaves the fold either
    complete with all its rows or still ``running`` with none of them
    (SPEC section 7). The fold must be ``running``.
    """
    run, fold = record.run_id, record.fold
    if not record.history:
        raise ValueError(f"fold {fold} of run {run} has no training history")
    with conn:
        status = conn.execute(
            "SELECT status FROM fold_status WHERE run_id = ? AND fold = ?", (run, fold)
        ).fetchone()
        if status is None or status[0] != "running":
            found = None if status is None else status[0]
            raise ValueError(f"fold {fold} of run {run} is not running (status {found})")
        conn.execute(
            "INSERT INTO thresholds (run_id, fold, threshold, selection_rule, val_dice) "
            "VALUES (?, ?, ?, ?, ?)",
            (run, fold, record.threshold, record.selection_rule, record.val_dice),
        )
        conn.executemany(
            "INSERT INTO training_history (run_id, fold, epoch, train_loss, val_dice) "
            "VALUES (?, ?, ?, ?, ?)",
            [(run, fold, e.epoch, e.train_loss, e.val_dice) for e in record.history],
        )
        conn.executemany(
            "INSERT INTO per_image_metrics (run_id, dataset, image_id, fold, prediction_mode, "
            "dice, sensitivity, specificity, precision_score, accuracy, "
            "predicted_vessel_fraction) VALUES (?, ?, ?, ?, 'single', ?, ?, ?, ?, ?, ?)",
            [
                (
                    run,
                    m.dataset,
                    m.image_id,
                    fold,
                    m.metrics.dice,
                    m.metrics.sensitivity,
                    m.metrics.specificity,
                    m.metrics.precision,
                    m.metrics.accuracy,
                    m.metrics.predicted_vessel_fraction,
                )
                for m in record.test_metrics
            ],
        )
        conn.execute(
            "UPDATE fold_status SET status = 'complete', checkpoint_sha256 = ?, "
            "completed_at = ? WHERE run_id = ? AND fold = ?",
            (record.checkpoint_sha256, completed_at, run, fold),
        )


def finish_run(conn: sqlite3.Connection, run_id: str, finished_at: str) -> None:
    """Set ``finished_at`` once every started fold of ``run_id`` is complete."""
    open_folds = [f for f, s in fold_statuses(conn, run_id).items() if s != "complete"]
    if open_folds:
        raise ValueError(f"run {run_id} still has folds that are not complete: {open_folds}")
    with conn:
        conn.execute("UPDATE runs SET finished_at = ? WHERE run_id = ?", (finished_at, run_id))
