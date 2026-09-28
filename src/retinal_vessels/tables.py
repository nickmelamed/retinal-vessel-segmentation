"""Build the Markdown tables in ``results/tables`` from reported runs.

Documents copy every number they quote from these files, and
``scripts/agent/check_numbers.py`` checks this. Each file ends with a
line naming the runs, tag, commit, and data hash it came from (SPEC section
17). The output has no timestamps, so regenerating it from the same database
gives the same bytes.
"""

import json
import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from retinal_vessels.config import TablesConfig
from retinal_vessels.db import run_query
from retinal_vessels.evaluate import EVALUATION_NAME

logger = logging.getLogger(__name__)

THRESHOLD_LOG = "05_threshold_log"
LEAKAGE_AUDIT = "04_leakage_audit"
DATASET = "drive"
# Column name in per_image_metrics, and the label shown in the tables.
HEADLINE_METRICS = (
    ("dice", "Dice"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("precision_score", "Precision"),
    ("accuracy", "Accuracy"),
    ("auc_roc", "AUC-ROC"),
    ("auc_pr", "AUC-PR"),
    ("brier", "Brier score"),
)
WIDTH_METRICS = (("thin_sensitivity", "Thin"), ("thick_sensitivity", "Thick"))
IMAGE_COLUMNS = (
    "m.image_id, m.fold, i.has_abnormality, i.abnormality_note, i.vessel_fraction_in_fov, "
    "m.dice, m.sensitivity, m.specificity, m.precision_score, m.accuracy, m.auc_roc, "
    "m.auc_pr, m.brier, m.thin_sensitivity, m.thick_sensitivity, m.predicted_vessel_fraction"
)
RUN_COLUMNS = (
    "run_id, variant, git_tag, git_commit, git_dirty, data_hash, gpu_type, compute_platform, "
    "python_version, tensorflow_version, cuda_version, deterministic_ops, started_at, "
    "finished_at, is_reported"
)
DEVELOPMENT_NOTE = "> Development run, not reported. Do not cite these numbers.\n\n"
SECONDS_PER_MINUTE = 60


class TableError(RuntimeError):
    """Runs that cannot be tabulated."""


@dataclass(frozen=True)
class Summary:
    """Spread of one metric across images, with a percentile bootstrap interval for the mean."""

    n: int
    mean: float
    sd: float
    median: float
    minimum: float
    maximum: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class RunTables:
    """What the tables need from one run: its record, images, folds, and evaluation summary.

    ``leaks`` counts the leakage audit's rows for this run, which must be zero.
    """

    run: dict[str, Any]
    images: list[dict[str, Any]]
    folds: list[dict[str, Any]]
    evaluation: dict[str, Any]
    leaks: int


def bootstrap_mean_ci(
    values: Sequence[float], resamples: int, seed: int, level: float
) -> tuple[float, float]:
    """Return the percentile bootstrap interval for the mean of ``values``.

    Images are resampled with replacement. Each call draws from a fresh
    generator seeded with ``seed``, so every metric of a run is resampled
    with the same image draws and the result does not depend on call order.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(resamples, data.size))
    means = data[draws].mean(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)


def summarize(values: Sequence[float], config: TablesConfig) -> Summary:
    """Summarize ``values`` across images. SD is the sample SD, so it needs two values."""
    data = np.asarray(values, dtype=np.float64)
    if data.size < 2:
        raise ValueError(f"need at least 2 values to summarize, got {data.size}")
    low, high = bootstrap_mean_ci(
        values, config.bootstrap_resamples, config.bootstrap_seed, config.ci_level
    )
    return Summary(
        n=int(data.size),
        mean=float(data.mean()),
        sd=float(data.std(ddof=1)),
        median=float(np.median(data)),
        minimum=float(data.min()),
        maximum=float(data.max()),
        ci_low=low,
        ci_high=high,
    )


def fmt(value: float | None, decimals: int) -> str:
    """Format a number with a fixed count of decimals, and a missing one as NA."""
    return "NA" if value is None else f"{value:.{decimals}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a GitHub Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def load_run(conn: sqlite3.Connection, run_id: str, results_dir: Path) -> RunTables:
    """Read one run's rows and its ``evaluation.json``.

    Raises ``TableError`` if the run is unfinished, unevaluated, or its
    summary does not match the database.
    """
    runs = _dicts(conn.execute(f"SELECT {RUN_COLUMNS} FROM runs WHERE run_id = ?", (run_id,)))
    if not runs:
        raise TableError(f"no run {run_id} in the database")
    run = runs[0]
    if run["finished_at"] is None:
        raise TableError(f"run {run_id} is not finished")
    images = _dicts(
        conn.execute(
            f"SELECT {IMAGE_COLUMNS} FROM per_image_metrics AS m "
            "INNER JOIN images AS i ON m.dataset = i.dataset AND m.image_id = i.image_id "
            "WHERE m.run_id = ? AND m.dataset = ? AND m.prediction_mode = 'single' "
            "ORDER BY m.image_id",
            (run_id, DATASET),
        )
    )
    if not images or any(image["brier"] is None for image in images):
        raise TableError(f"run {run_id} is not evaluated. Run make evaluate first.")
    path = results_dir / run_id / EVALUATION_NAME
    if not path.is_file():
        raise TableError(f"{path} is missing")
    evaluation = json.loads(path.read_text(encoding="utf-8"))

    edges = {f["fold"]: f for f in evaluation["folds"]}
    names = (
        "run_id",
        "variant",
        "is_reported",
        "fold",
        "threshold",
        "selection_rule",
        "val_dice",
        "is_finished",
        "best_epoch",
        "n_epochs",
    )
    folds = []
    for row in run_query(conn, THRESHOLD_LOG):
        fold = dict(zip(names, row, strict=True))
        if fold["run_id"] != run_id:
            continue
        summary = edges.get(fold["fold"])
        if summary is None or summary["threshold"] != fold["threshold"]:
            raise TableError(f"{path} does not match the thresholds of run {run_id}")
        fold.update(
            thin_edge=summary["thin_edge"], n_thin=summary["n_thin"], n_thick=summary["n_thick"]
        )
        fold["test"] = sorted(i["image_id"] for i in images if i["fold"] == fold["fold"])
        folds.append(fold)
    if len(folds) != len(edges):
        raise TableError(f"{path} lists {len(edges)} folds, the database {len(folds)}")
    leaks = sum(1 for row in run_query(conn, LEAKAGE_AUDIT) if row[0] == run_id)
    return RunTables(run, images, folds, evaluation, leaks)


def _heading(data: RunTables) -> str:
    run = data.run
    return f"## {run['variant']} (run {run['run_id']})\n\n"


def _values(images: Sequence[dict[str, Any]], column: str) -> list[float]:
    return [float(i[column]) for i in images if i[column] is not None]


def headline(data: RunTables, config: TablesConfig) -> str:
    """Return the headline table of metrics across the out-of-fold images, with intervals."""
    d = config.decimals
    level = round(config.ci_level * 100)
    rows = []
    for column, label in HEADLINE_METRICS:
        s = summarize(_values(data.images, column), config)
        rows.append(
            [
                label,
                str(s.n),
                fmt(s.mean, d),
                f"{fmt(s.ci_low, d)} to {fmt(s.ci_high, d)}",
                fmt(s.sd, d),
                fmt(s.median, d),
                fmt(s.minimum, d),
                fmt(s.maximum, d),
            ]
        )
    majority = [1 - float(i["vessel_fraction_in_fov"]) for i in data.images]
    return (
        _heading(data)
        + markdown_table(
            ["Metric", "Images", "Mean", f"{level}% CI of mean", "SD", "Median", "Min", "Max"],
            rows,
        )
        + f"\nIntervals are percentile bootstraps over images "
        f"({config.bootstrap_resamples} resamples, seed {config.bootstrap_seed}). "
        f"Predicting no vessel anywhere would score a mean accuracy of "
        f"{fmt(float(np.mean(majority)), d)} on the same images.\n"
    )


def per_image(data: RunTables, config: TablesConfig) -> str:
    """Return the table of every out-of-fold image with its fold and metrics."""
    d = config.decimals
    columns = [c for c, _ in HEADLINE_METRICS if c != "accuracy"] + [c for c, _ in WIDTH_METRICS]
    labels = dict(HEADLINE_METRICS + WIDTH_METRICS)
    rows = [
        [i["image_id"], str(i["fold"]), "yes" if i["has_abnormality"] else "no"]
        + [fmt(i[c], d) for c in columns]
        for i in data.images
    ]
    headers = ["Image", "Fold", "Abnormal"] + [
        f"{labels[c]} sensitivity" if c in dict(WIDTH_METRICS) else labels[c] for c in columns
    ]
    return _heading(data) + markdown_table(headers, rows)


def per_fold(data: RunTables, config: TablesConfig) -> str:
    """Return the table of each fold's test images, threshold, epochs, and thin edge."""
    d = config.decimals
    rows = [
        [
            str(f["fold"]),
            ", ".join(f["test"]),
            fmt(f["threshold"], d),
            fmt(f["val_dice"], d),
            str(f["best_epoch"]),
            str(f["n_epochs"]),
            fmt(f["thin_edge"], d),
            str(f["n_thin"]),
            str(f["n_thick"]),
        ]
        for f in data.folds
    ]
    table = markdown_table(
        [
            "Fold",
            "Test images",
            "Threshold",
            "Validation Dice",
            "Best epoch",
            "Epochs trained",
            "Thin edge (px)",
            "Thin skeleton px",
            "Thick skeleton px",
        ],
        rows,
    )
    rule = data.folds[0]["selection_rule"] if data.folds else ""
    return _heading(data) + table + f"\nThreshold rule: {rule}.\n\n" + _edge_note(data, d)


def _edge_note(data: RunTables, decimals: int) -> str:
    edges = sorted({fmt(f["thin_edge"], decimals) for f in data.folds})
    if len(edges) == 1:
        return (
            f"Every fold set the same thin/thick edge, a skeleton radius of {edges[0]} px. "
            "A skeleton pixel is thin when its radius is at most the edge.\n"
        )
    return (
        f"The folds set different thin/thick edges ({', '.join(edges)} px), so thin means a "
        "different set of vessels in each fold. A skeleton pixel is thin when its radius is "
        "at most its fold's edge.\n"
    )


def thin_thick(data: RunTables, config: TablesConfig) -> str:
    """Return the table of sensitivity on thin and thick vessel skeleton pixels."""
    d = config.decimals
    level = round(config.ci_level * 100)
    pixels = {
        "thin_sensitivity": sum(f["n_thin"] for f in data.folds),
        "thick_sensitivity": sum(f["n_thick"] for f in data.folds),
    }
    rows = []
    for column, label in WIDTH_METRICS:
        s = summarize(_values(data.images, column), config)
        rows.append(
            [
                label,
                str(s.n),
                str(pixels[column]),
                fmt(s.mean, d),
                f"{fmt(s.ci_low, d)} to {fmt(s.ci_high, d)}",
                fmt(s.median, d),
            ]
        )
    return (
        _heading(data)
        + markdown_table(
            [
                "Vessels",
                "Images",
                "Skeleton px",
                "Mean sensitivity",
                f"{level}% CI of mean",
                "Median",
            ],
            rows,
        )
        + "\n"
        + _edge_note(data, d)
    )


def calibration(data: RunTables, config: TablesConfig) -> str:
    """Return the pooled reliability table and Brier score over out-of-fold FOV pixels."""
    d = config.decimals
    table = data.evaluation["reliability"]
    edges = table["edges"]
    rows = [
        [
            f"{fmt(edges[k], d)} to {fmt(edges[k + 1], d)}",
            str(table["counts"][k]),
            fmt(table["mean_probability"][k], d),
            fmt(table["vessel_fraction"][k], d),
        ]
        for k in range(len(table["counts"]))
    ]
    return (
        _heading(data)
        + markdown_table(
            ["Predicted probability", "FOV pixels", "Mean predicted", "Observed vessel fraction"],
            rows,
        )
        + f"\nPooled Brier score over all out-of-fold FOV pixels: "
        f"{fmt(table['pooled_brier'], d)}.\n"
    )


def pathology(data: RunTables, config: TablesConfig) -> str:
    """Return the table of abnormal images against the rest, described with no test."""
    d = config.decimals
    abnormal = [i for i in data.images if i["has_abnormality"]]
    others = [i for i in data.images if not i["has_abnormality"]]
    rows = [
        [
            i["image_id"],
            str(i["fold"]),
            fmt(i["dice"], d),
            fmt(i["sensitivity"], d),
            fmt(i["auc_pr"], d),
            i["abnormality_note"],
        ]
        for i in abnormal
    ]
    rest = []
    for column, label in (("dice", "Dice"), ("sensitivity", "Sensitivity"), ("auc_pr", "AUC-PR")):
        values = _values(others, column)
        rest.append(
            [
                label,
                str(len(values)),
                fmt(float(np.median(values)), d),
                fmt(min(values), d),
                fmt(max(values), d),
            ]
        )
    return (
        _heading(data)
        + markdown_table(
            ["Image", "Fold", "Dice", "Sensitivity", "AUC-PR", "Note from the official site"], rows
        )
        + "\nThe other images, for comparison:\n\n"
        + markdown_table(["Metric", "Images", "Median", "Min", "Max"], rest)
        + f"\nThis is descriptive only. With {len(abnormal)} abnormal images there is no basis "
        "for a statistical test.\n"
    )


def _minutes(start: str, end: str) -> int:
    fmt_ = "%Y-%m-%dT%H:%M:%SZ"
    delta = datetime.strptime(end, fmt_) - datetime.strptime(start, fmt_)
    return round(delta.total_seconds() / SECONDS_PER_MINUTE)


def provenance(data: RunTables, config: TablesConfig) -> str:
    """Return the table of the run's code, data, hardware, and timing."""
    run = data.run
    code = data.evaluation["code"]
    rows = [
        ["Variant", run["variant"]],
        ["Tag", str(run["git_tag"])],
        ["Commit", run["git_commit"]],
        ["Trained from a clean tree", "no" if run["git_dirty"] else "yes"],
        ["Evaluated at commit", code["commit"]],
        ["Evaluated from a clean tree", "no" if code["dirty"] else "yes"],
        ["Data hash", run["data_hash"]],
        ["Leakage audit rows (query 04)", str(data.leaks)],
        ["GPU", run["gpu_type"]],
        ["Platform", run["compute_platform"]],
        ["Python", run["python_version"]],
        ["TensorFlow", run["tensorflow_version"]],
        ["CUDA", str(run["cuda_version"])],
        ["Deterministic ops", "yes" if run["deterministic_ops"] else "no"],
        ["Started", run["started_at"]],
        ["Finished", run["finished_at"]],
        ["Training time (minutes)", str(_minutes(run["started_at"], run["finished_at"]))],
    ]
    return _heading(data) + markdown_table(["Field", "Value"], rows)


def dataset(conn: sqlite3.Connection, config: TablesConfig) -> str:
    """Return the table of labeled DRIVE images and their within-FOV vessel fraction."""
    d = config.decimals
    rows = conn.execute(
        "SELECT fov_pixels, vessel_fraction_in_fov FROM images "
        "WHERE dataset = ? AND has_labels = 1",
        (DATASET,),
    ).fetchall()
    if not rows:
        raise TableError("the images table has no labeled DRIVE images. Run make check-data.")
    fov = np.array([r[0] for r in rows], dtype=np.float64)
    fraction = np.array([r[1] for r in rows], dtype=np.float64)
    pooled = float((fov * fraction).sum() / fov.sum())
    return markdown_table(
        ["Labeled images", "Pooled vessel fraction in FOV", "Per-image min", "Per-image max"],
        [
            [
                str(len(rows)),
                fmt(pooled, d),
                fmt(float(fraction.min()), d),
                fmt(float(fraction.max()), d),
            ]
        ],
    )


RUN_TABLES: dict[str, Callable[[RunTables, TablesConfig], str]] = {
    "headline": headline,
    "per_image": per_image,
    "per_fold": per_fold,
    "thin_thick": thin_thick,
    "calibration": calibration,
    "pathology": pathology,
    "provenance": provenance,
}


def _lineage(runs: Sequence[RunTables], source: Path) -> str:
    parts = [
        f"{r.run['run_id']} ({r.run['variant']}, tag {r.run['git_tag']}, commit "
        f"{r.run['git_commit']}, data hash {r.run['data_hash']})"
        for r in runs
    ]
    return f"\nGenerated by scripts/make_tables.py from {source.name}. Runs: {'; '.join(parts)}.\n"


def write_tables(
    conn: sqlite3.Connection,
    run_ids: Sequence[str],
    results_dir: Path,
    out_dir: Path,
    config: TablesConfig,
    source: Path,
    development: bool = False,
) -> list[Path]:
    """Write one Markdown file per table to ``out_dir`` and return their paths.

    Each run gets its own section in every run table. Two runs of one variant
    raise ``TableError``, since a document could not tell which to quote.
    ``development`` marks every file as not citable.
    """
    if not run_ids:
        raise TableError("no runs to tabulate. Mark a run reported first.")
    runs = [load_run(conn, run_id, results_dir) for run_id in run_ids]
    variants = [r.run["variant"] for r in runs]
    if len(set(variants)) != len(variants):
        raise TableError(f"more than one run per variant: {sorted(variants)}")
    if not development and not all(r.run["is_reported"] for r in runs):
        raise TableError("only reported runs go into the tables")

    header = DEVELOPMENT_NOTE if development else ""
    lineage = _lineage(runs, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {name: "\n".join(build(r, config) for r in runs) for name, build in RUN_TABLES.items()}
    texts["dataset"] = dataset(conn, config)
    paths = []
    for name, text in texts.items():
        path = out_dir / f"{name}.md"
        path.write_text(header + text + lineage, encoding="utf-8")
        paths.append(path)
    logger.info("Wrote %d tables to %s for runs %s", len(paths), out_dir, list(run_ids))
    return paths


def reported_run_ids(conn: sqlite3.Connection) -> list[str]:
    """Return the reported runs, oldest first."""
    rows = conn.execute("SELECT run_id FROM runs WHERE is_reported = 1 ORDER BY run_id")
    return [r[0] for r in rows]
