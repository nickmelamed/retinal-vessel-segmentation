"""Draw the committed figures from a reported run (SPEC section 11).

Every figure comes from the database, the run's ``evaluation.json``, and
its saved out-of-fold probabilities, never from numbers typed in by hand.
Each PNG gets a JSON sidecar naming the run, tag, commit, data hash, and
images it used (SPEC section 17). Figures are drawn on a bare ``Figure``
with the Agg canvas, so no plotting backend setting matters, and PNG
metadata is left out so that redrawing gives the same bytes.

Run with ``python -m retinal_vessels.figures`` (``make figures``). The
image figures need the DRIVE data, checked against the committed checksums
and the run's data hash, and the run's saved predictions.
"""

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from retinal_vessels.config import ConfigError, FiguresConfig, TablesConfig, load_reporting_config
from retinal_vessels.data import Sample
from retinal_vessels.datasets.drive import LayoutError, drive_dir, load_drive
from retinal_vessels.db import connect
from retinal_vessels.metrics import binarize, binary_metrics
from retinal_vessels.provenance import check_or_write_checksums, data_hash
from retinal_vessels.tables import RunTables, TableError, fmt, load_run, summarize
from retinal_vessels.train import OutputDirs
from retinal_vessels.utils import setup_logging

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
POINTS_PER_INCH = 72
# Mark specs of the palette, in pixels at the output resolution.
LINE_PX = 2
HAIRLINE_PX = 1
MARKER_PX = 8
ATTRIBUTION = "Images from DRIVE (Staal et al., 2004)."
# Figure sizes in inches, chosen so image panels show DRIVE's 565 px width
# at about its native size and every PNG stays under the large-file limit.
HERO_SIZE = (12.0, 3.9)
BEST_WORST_WIDTH = 5.2
BEST_WORST_ROW_HEIGHT = 2.6
CURVE_COLUMN_WIDTH = 2.6
CURVE_HEIGHT = 5.0
CHART_SIZE = (5.6, 6.2)
STRIP_SIZE = (5.6, 4.6)
# Horizontal layout of the thin/thick strip chart, in category units.
STRIP_SPREAD = 0.12
MEAN_OFFSET = 0.3
POINT_ALPHA = 0.6
# Where the diagonal's label sits, below the curve of a typical segmenter.
CALIBRATION_LABEL_AT = 0.2


class FigureError(RuntimeError):
    """A run, its data, or its predictions that cannot be drawn."""


@dataclass(frozen=True)
class Drawn:
    """One out-of-fold image ready to draw: the sample, its prediction, and its Dice."""

    sample: Sample
    prediction: np.ndarray
    dice: float
    abnormal: bool


def _pt(pixels: float, dpi: int) -> float:
    return pixels * POINTS_PER_INCH / dpi


def error_map(
    prediction: np.ndarray, label: np.ndarray, fov: np.ndarray, colors: FiguresConfig
) -> np.ndarray:
    """Return an (H, W, 3) float RGB image of the outcome of every pixel.

    True positives, false positives, and false negatives inside the FOV get
    their configured colors. True negatives and everything outside the FOV
    get the surface color, so only vessels and errors carry ink.
    """
    if not prediction.shape == label.shape == fov.shape:
        raise ValueError(
            f"shapes differ: prediction {prediction.shape}, label {label.shape}, fov {fov.shape}"
        )
    out = np.empty((*label.shape, 3), dtype=np.float64)
    out[:] = to_rgb(colors.surface)
    out[prediction & label & fov] = to_rgb(colors.true_positive)
    out[prediction & ~label & fov] = to_rgb(colors.false_positive)
    out[~prediction & label & fov] = to_rgb(colors.false_negative)
    return out


def median_image(images: Sequence[Mapping[str, Any]]) -> str:
    """Return the id of the image whose Dice is closest to the median, lowest id on a tie."""
    if not images:
        raise ValueError("no images to choose from")
    median = float(np.median([i["dice"] for i in images]))
    return str(min(images, key=lambda i: (abs(i["dice"] - median), i["image_id"]))["image_id"])


def best_and_worst(images: Sequence[Mapping[str, Any]], n: int) -> tuple[list[str], list[str]]:
    """Return the ``n`` highest and ``n`` lowest Dice image ids, ties broken by id.

    The best list starts with the highest Dice and the worst with the lowest.
    Raises ``ValueError`` if the two lists would share an image.
    """
    if 2 * n > len(images):
        raise ValueError(f"cannot show {n} best and {n} worst of {len(images)} images")
    ranked = sorted(images, key=lambda i: (i["dice"], i["image_id"]))
    worst = [str(i["image_id"]) for i in ranked[:n]]
    best = [
        str(i["image_id"]) for i in sorted(images, key=lambda i: (-i["dice"], i["image_id"]))[:n]
    ]
    return best, worst


def _style(ax: Axes, colors: FiguresConfig, dpi: int) -> None:
    ax.set_facecolor(colors.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors.gridline)
        ax.spines[side].set_linewidth(_pt(HAIRLINE_PX, dpi))
    ax.tick_params(colors=colors.muted_ink, labelcolor=colors.secondary_ink, width=0)
    ax.grid(True, color=colors.gridline, linewidth=_pt(HAIRLINE_PX, dpi))
    ax.set_axisbelow(True)


def _new_figure(width: float, height: float, colors: FiguresConfig, dpi: int) -> Figure:
    return Figure(figsize=(width, height), dpi=dpi, facecolor=colors.surface, layout="constrained")


def _image_axes(ax: Axes, title: str, colors: FiguresConfig) -> None:
    ax.set_axis_off()
    ax.set_title(title, color=colors.ink, fontsize="medium")


def _error_legend(colors: FiguresConfig) -> list[Patch]:
    return [
        Patch(facecolor=colors.true_positive, label="Vessel found"),
        Patch(facecolor=colors.false_positive, label="False vessel"),
        Patch(facecolor=colors.false_negative, label="Missed vessel"),
    ]


def _label(sample: Sample) -> np.ndarray:
    if sample.label is None:
        raise FigureError(f"image {sample.image_id} has no label")
    return sample.label


def hero(item: Drawn, data: RunTables, colors: FiguresConfig, decimals: int, dpi: int) -> Figure:
    """The image closest to the median Dice: image, ground truth, prediction, and errors."""
    s = item.sample
    label = _label(s)
    fig = _new_figure(*HERO_SIZE, colors, dpi)
    axes = fig.subplots(1, 4)
    axes[0].imshow(s.image)
    axes[1].imshow(label & s.fov, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[2].imshow(item.prediction & s.fov, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[3].imshow(error_map(item.prediction, label, s.fov, colors), interpolation="nearest")
    for ax, title in zip(
        axes, ("Fundus image", "Ground truth", "Prediction", "Errors"), strict=True
    ):
        _image_axes(ax, title, colors)
    fig.suptitle(
        f"DRIVE image {s.image_id}, out-of-fold Dice {fmt(item.dice, decimals)}, "
        "the image closest to the median Dice",
        color=colors.ink,
    )
    fig.legend(
        handles=_error_legend(colors),
        loc="outside lower center",
        ncols=3,
        frameon=False,
        labelcolor=colors.secondary_ink,
    )
    fig.text(
        0.01,
        0.01,
        f"{ATTRIBUTION} Run {data.run['run_id']}.",
        color=colors.muted_ink,
        fontsize="x-small",
    )
    return fig


def best_worst(
    worst: Sequence[Drawn],
    best: Sequence[Drawn],
    data: RunTables,
    colors: FiguresConfig,
    decimals: int,
    dpi: int,
) -> Figure:
    """The lowest and highest Dice images, each with its error map."""
    rows = [("Worst", d) for d in worst] + [("Best", d) for d in best]
    fig = _new_figure(BEST_WORST_WIDTH, BEST_WORST_ROW_HEIGHT * len(rows), colors, dpi)
    axes = fig.subplots(len(rows), 2, squeeze=False)
    for (group, item), (left, right) in zip(rows, axes, strict=True):
        s = item.sample
        left.imshow(s.image)
        right.imshow(error_map(item.prediction, _label(s), s.fov, colors), interpolation="nearest")
        note = ", abnormal on the official list" if item.abnormal else ""
        _image_axes(left, f"{group}: image {s.image_id}{note}", colors)
        _image_axes(right, f"Dice {fmt(item.dice, decimals)}", colors)
    fig.suptitle("Lowest and highest out-of-fold Dice", color=colors.ink)
    fig.legend(
        handles=_error_legend(colors),
        loc="outside lower center",
        ncols=3,
        frameon=False,
        labelcolor=colors.secondary_ink,
    )
    fig.text(
        0.01,
        0.005,
        f"{ATTRIBUTION} Run {data.run['run_id']}.",
        color=colors.muted_ink,
        fontsize="x-small",
    )
    return fig


def training_curves(
    history: Mapping[int, Sequence[tuple[int, float, float | None]]],
    data: RunTables,
    colors: FiguresConfig,
    dpi: int,
) -> Figure:
    """Training loss and validation Dice per epoch, one column per fold, best epoch marked."""
    folds = sorted(history)
    best = {f["fold"]: f["best_epoch"] for f in data.folds}
    fig = _new_figure(CURVE_COLUMN_WIDTH * len(folds), CURVE_HEIGHT, colors, dpi)
    axes = fig.subplots(2, len(folds), squeeze=False, sharey="row", sharex="col")
    line = {"color": colors.series, "linewidth": _pt(LINE_PX, dpi), "solid_capstyle": "round"}
    for col, fold in enumerate(folds):
        epochs = [e for e, _, _ in history[fold]]
        loss_ax, dice_ax = axes[0][col], axes[1][col]
        for ax in (loss_ax, dice_ax):
            _style(ax, colors, dpi)
        loss_ax.plot(epochs, [loss for _, loss, _ in history[fold]], **line)
        dice = [(e, d) for e, _, d in history[fold] if d is not None]
        dice_ax.plot([e for e, _ in dice], [d for _, d in dice], **line)
        chosen = [(e, d) for e, d in dice if e == best[fold]]
        if chosen:
            dice_ax.plot(
                *zip(*chosen, strict=True),
                marker="o",
                linestyle="none",
                markersize=_pt(MARKER_PX, dpi),
                color=colors.series,
                markeredgecolor=colors.surface,
                markeredgewidth=_pt(LINE_PX, dpi),
            )
            dice_ax.annotate(
                f"best, epoch {best[fold]}",
                chosen[0],
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize="x-small",
                color=colors.secondary_ink,
            )
        loss_ax.set_title(f"Fold {fold}", color=colors.ink, fontsize="medium")
        dice_ax.set_xlabel("Epoch", color=colors.secondary_ink)
        dice_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0][0].set_ylabel("Training loss", color=colors.secondary_ink)
    axes[1][0].set_ylabel("Validation Dice", color=colors.secondary_ink)
    fig.suptitle("Training curves per fold", color=colors.ink)
    return fig


def reliability(data: RunTables, colors: FiguresConfig, decimals: int, dpi: int) -> Figure:
    """Observed vessel fraction against mean predicted probability, with pixels per bin below."""
    table = data.evaluation["reliability"]
    edges = np.asarray(table["edges"], dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    filled = [
        (p, v)
        for p, v in zip(table["mean_probability"], table["vessel_fraction"], strict=True)
        if p is not None and v is not None
    ]
    fig = _new_figure(*CHART_SIZE, colors, dpi)
    top, bottom = fig.subplots(2, 1, height_ratios=(3, 1), sharex=True)
    for ax in (top, bottom):
        _style(ax, colors, dpi)
    top.plot([0, 1], [0, 1], color=colors.muted_ink, linewidth=_pt(HAIRLINE_PX, dpi))
    top.annotate(
        "Perfect calibration",
        (CALIBRATION_LABEL_AT, CALIBRATION_LABEL_AT),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize="x-small",
        color=colors.muted_ink,
    )
    top.plot(
        [p for p, _ in filled],
        [v for _, v in filled],
        color=colors.series,
        linewidth=_pt(LINE_PX, dpi),
        marker="o",
        markersize=_pt(MARKER_PX, dpi),
        markeredgecolor=colors.surface,
        markeredgewidth=_pt(LINE_PX, dpi),
    )
    top.set(xlim=(0, 1), ylim=(0, 1))
    top.set_ylabel("Observed vessel fraction", color=colors.secondary_ink)
    top.set_title(
        f"Pooled out-of-fold FOV pixels. Brier score {fmt(table['pooled_brier'], decimals)}",
        color=colors.secondary_ink,
        fontsize="small",
    )
    counts = np.asarray(table["counts"], dtype=np.float64)
    shown = counts > 0
    bottom.vlines(
        centers[shown], 1, counts[shown], color=colors.series, linewidth=_pt(LINE_PX, dpi)
    )
    bottom.plot(
        centers[shown],
        counts[shown],
        marker="o",
        linestyle="none",
        markersize=_pt(MARKER_PX, dpi),
        color=colors.series,
        markeredgecolor=colors.surface,
        markeredgewidth=_pt(LINE_PX, dpi),
    )
    bottom.set_yscale("log")
    bottom.set_xlabel("Mean predicted probability", color=colors.secondary_ink)
    bottom.set_ylabel("Pixels per bin", color=colors.secondary_ink)
    fig.suptitle("Reliability diagram", color=colors.ink)
    return fig


def edge_note(data: RunTables, decimals: int) -> str:
    """Say which skeleton radius counts as thin, and whether the folds agree on it."""
    edges = sorted({fmt(f["thin_edge"], decimals) for f in data.folds})
    if len(edges) == 1:
        return f"Thin means a skeleton radius of at most {edges[0]} px, the same in every fold."
    per_fold = ", ".join(f"fold {f['fold']}: {fmt(f['thin_edge'], decimals)}" for f in data.folds)
    return f"Thin edges differ by fold ({per_fold} px), so thin is not the same set in each fold."


def thin_thick(data: RunTables, tables: TablesConfig, colors: FiguresConfig, dpi: int) -> Figure:
    """Per-image sensitivity on thin and thick vessels, with the mean and its interval."""
    fig = _new_figure(*STRIP_SIZE, colors, dpi)
    ax = fig.subplots()
    _style(ax, colors, dpi)
    ax.grid(False, axis="x")
    bins = (("thin_sensitivity", "Thin vessels"), ("thick_sensitivity", "Thick vessels"))
    for x, (column, _) in enumerate(bins):
        values = [float(i[column]) for i in data.images if i[column] is not None]
        # Spread in image order, so horizontal position carries no meaning.
        offsets = np.linspace(-STRIP_SPREAD, STRIP_SPREAD, len(values))
        ax.plot(
            x + offsets,
            values,
            marker="o",
            linestyle="none",
            markersize=_pt(MARKER_PX, dpi),
            color=colors.muted_ink,
            alpha=POINT_ALPHA,
        )
        s = summarize(values, tables)
        ax.errorbar(
            x + MEAN_OFFSET,
            s.mean,
            yerr=[[s.mean - s.ci_low], [s.ci_high - s.mean]],
            fmt="o",
            color=colors.series,
            markersize=_pt(MARKER_PX, dpi),
            elinewidth=_pt(LINE_PX, dpi),
            capsize=0,
        )
        ax.annotate(
            f"mean {fmt(s.mean, tables.decimals)}",
            (x + MEAN_OFFSET, s.mean),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize="x-small",
            color=colors.secondary_ink,
        )
    ax.set_xticks(range(len(bins)), [label for _, label in bins])
    ax.set_xlim(-0.5, len(bins) - 0.2)
    ax.set_ylabel("Sensitivity on skeleton pixels", color=colors.secondary_ink)
    level = round(tables.ci_level * 100)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=colors.muted_ink, label="One image"),
            Line2D([], [], marker="o", color=colors.series, label=f"Mean, {level}% CI"),
        ],
        frameon=False,
        labelcolor=colors.secondary_ink,
        loc="best",
    )
    ax.set_title(
        edge_note(data, tables.decimals), color=colors.secondary_ink, fontsize="small", wrap=True
    )
    fig.suptitle("Sensitivity on thin and thick vessels", color=colors.ink)
    return fig


def load_history(
    conn: sqlite3.Connection, run_id: str
) -> dict[int, list[tuple[int, float, float | None]]]:
    """Return ``{fold: [(epoch, train_loss, val_dice), ...]}`` in epoch order."""
    rows = conn.execute(
        "SELECT fold, epoch, train_loss, val_dice FROM training_history WHERE run_id = ? "
        "ORDER BY fold, epoch",
        (run_id,),
    ).fetchall()
    history: dict[int, list[tuple[int, float, float | None]]] = {}
    for fold, epoch, loss, dice in rows:
        history.setdefault(int(fold), []).append((int(epoch), float(loss), dice))
    if not history:
        raise FigureError(f"run {run_id} has no training history")
    return history


def drawn_images(
    data: RunTables, ids: Sequence[str], samples: Mapping[str, Sample], dirs: OutputDirs
) -> dict[str, Drawn]:
    """Load the saved probabilities of ``ids`` and threshold them at their fold's threshold.

    The Dice recomputed from each file must equal the stored one, so a
    figure can never show a prediction other than the one that was scored.
    """
    rows = {i["image_id"]: i for i in data.images}
    thresholds = {f["fold"]: f["threshold"] for f in data.folds}
    drawn = {}
    for image_id in ids:
        row = rows[image_id]
        sample = samples[image_id]
        path = dirs.predictions(data.run["run_id"]) / f"{image_id}.npy"
        if not path.is_file():
            raise FigureError(f"prediction missing at {path}")
        prediction = binarize(np.load(path), thresholds[row["fold"]])
        dice = binary_metrics(prediction, _label(sample), sample.fov).dice
        if dice != row["dice"]:
            raise FigureError(f"{path} gives Dice {dice}, but the database stores {row['dice']}")
        drawn[image_id] = Drawn(
            sample, prediction, float(row["dice"]), bool(row["has_abnormality"])
        )
    return drawn


def _save(
    fig: Figure,
    out_dir: Path,
    name: str,
    data: RunTables,
    images: Sequence[str],
    development: bool,
    max_bytes: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), metadata={"Software": None})
    size = path.stat().st_size
    if size > max_bytes:
        path.unlink()
        raise FigureError(
            f"{path.name} is {size} bytes, over the {max_bytes} byte limit for committed files. "
            "Shrink its layout rather than raising the limit."
        )
    run = data.run
    sidecar = {
        "figure": path.name,
        "generated_by": "retinal_vessels.figures",
        "development_run": development,
        "run_id": run["run_id"],
        "variant": run["variant"],
        "tag": run["git_tag"],
        "commit": run["git_commit"],
        "data_hash": run["data_hash"],
        "images": list(images),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d bytes)", path, path.stat().st_size)
    return path


def write_figures(
    conn: sqlite3.Connection,
    run_id: str,
    samples: Sequence[Sample],
    dirs: OutputDirs,
    out_dir: Path,
    tables: TablesConfig,
    colors: FiguresConfig,
    development: bool = False,
) -> list[Path]:
    """Draw every figure of ``run_id`` into ``out_dir`` and return the PNG paths.

    Only a reported run is drawn unless ``development`` is set. Raises
    ``FigureError`` or ``TableError`` on any mismatch.
    """
    data = load_run(conn, run_id, dirs.results)
    if not development and not data.run["is_reported"]:
        raise FigureError(f"run {run_id} is not reported. Run make mark-reported first.")
    by_id = {s.image_id: s for s in samples}
    missing = sorted({i["image_id"] for i in data.images} - set(by_id))
    if missing:
        raise FigureError(f"the loaded data has no images {missing}")

    dpi, d = colors.dpi, tables.decimals
    middle = median_image(data.images)
    best, worst = best_and_worst(data.images, colors.n_best_worst)
    drawn = drawn_images(data, [middle, *worst, *best], by_id, dirs)
    all_ids = [i["image_id"] for i in data.images]
    figures = [
        ("hero", hero(drawn[middle], data, colors, d, dpi), [middle]),
        (
            "best_worst",
            best_worst([drawn[i] for i in worst], [drawn[i] for i in best], data, colors, d, dpi),
            [*worst, *best],
        ),
        ("training_curves", training_curves(load_history(conn, run_id), data, colors, dpi), []),
        ("reliability", reliability(data, colors, d, dpi), all_ids),
        ("thin_thick", thin_thick(data, tables, colors, dpi), all_ids),
    ]
    return [
        _save(fig, out_dir, name, data, ids, development, colors.max_bytes)
        for name, fig, ids in figures
    ]


def _reported_run(conn: sqlite3.Connection, variant: str) -> str:
    rows = conn.execute(
        "SELECT run_id FROM runs WHERE is_reported = 1 AND variant = ?", (variant,)
    ).fetchall()
    if len(rows) != 1:
        raise FigureError(f"expected one reported {variant} run, found {len(rows)}")
    return str(rows[0][0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="baseline", help="draw its reported run")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--checksums", type=Path, default=REPO / "data" / "CHECKSUMS.sha256")
    parser.add_argument("--db", type=Path, default=REPO / "results" / "experiments.db")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--out-dir", type=Path, default=REPO / "figures")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "reporting.yaml")
    parser.add_argument(
        "--allow-unreported",
        metavar="RUN_ID",
        help="draw this unreported run as a development run, never into figures/",
    )
    args = parser.parse_args(argv)
    setup_logging()

    if args.allow_unreported and args.out_dir.resolve() == (REPO / "figures").resolve():
        logger.error("unreported runs never go into %s. Pass another --out-dir.", REPO / "figures")
        return 1
    if not args.db.is_file():
        logger.error("database not found at %s", args.db)
        return 1
    try:
        config = load_reporting_config(args.config)
        drive = drive_dir(args.data_root / "DRIVE")
        _, report = check_or_write_checksums(args.checksums, {"DRIVE": drive})
        samples = load_drive(drive, "training")
    except (ConfigError, LayoutError, FileNotFoundError) as err:
        logger.error("%s", err)
        return 1
    dirs = OutputDirs(results=args.results_dir, models=REPO / "models")
    with closing(connect(args.db)) as conn:
        try:
            run_id = args.allow_unreported or _reported_run(conn, args.variant)
            (stored,) = conn.execute(
                "SELECT data_hash FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone() or (None,)
            if not report.ok or data_hash(report.checked) != stored:
                raise FigureError(f"the data does not match the data run {run_id} trained on")
            write_figures(
                conn,
                run_id,
                samples,
                dirs,
                args.out_dir,
                config.tables,
                config.figures,
                development=bool(args.allow_unreported),
            )
        except (FigureError, TableError) as err:
            logger.error("%s", err)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
