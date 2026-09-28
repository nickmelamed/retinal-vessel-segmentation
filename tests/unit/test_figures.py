import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.colors import to_rgb

from retinal_vessels import figures
from retinal_vessels.config import load_reporting_config
from retinal_vessels.db import set_reported
from retinal_vessels.figures import (
    FigureError,
    best_and_worst,
    edge_note,
    error_map,
    load_history,
    median_image,
    write_figures,
)
from retinal_vessels.tables import load_run
from tests.fixtures.evaluated_run import REPO, RUN_ID, TAG, Finished

CONFIG = load_reporting_config(REPO / "configs" / "reporting.yaml")
COLORS = CONFIG.figures
NAMES = ("hero", "best_worst", "training_curves", "reliability", "thin_thick")


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


@pytest.fixture
def reported(reportable: Finished) -> Finished:
    set_reported(reportable.conn, RUN_ID)
    return reportable


def draw(finished: Finished, out: Path, development: bool = False, **colors: Any) -> list[Path]:
    return write_figures(
        finished.conn,
        RUN_ID,
        finished.samples,
        finished.dirs,
        out,
        CONFIG.tables,
        COLORS.model_copy(update=colors),
        development,
    )


def images(*dice: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"image_id": i, "dice": d} for i, d in dice]


def test_error_map_colors_each_outcome() -> None:
    label = np.array([[1, 1, 0, 0, 1]], dtype=bool)
    prediction = np.array([[1, 0, 1, 0, 1]], dtype=bool)
    fov = np.array([[1, 1, 1, 1, 0]], dtype=bool)
    out = error_map(prediction, label, fov, COLORS)
    assert out.shape == (1, 5, 3)
    expected = [
        COLORS.true_positive,
        COLORS.false_negative,
        COLORS.false_positive,
        COLORS.surface,
        # A true positive outside the FOV is not shown.
        COLORS.surface,
    ]
    for k, color in enumerate(expected):
        assert tuple(out[0, k]) == pytest.approx(to_rgb(color)), k


def test_error_map_rejects_mismatched_shapes() -> None:
    a = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="shapes differ"):
        error_map(a, np.zeros((2, 3), dtype=bool), a, COLORS)


def test_median_image_with_an_odd_count() -> None:
    assert median_image(images(("a", 0.1), ("b", 0.5), ("c", 0.9))) == "b"


def test_median_image_breaks_ties_by_id() -> None:
    # The median of 0.4 and 0.6 is 0.5, equally close to both.
    assert median_image(images(("z", 0.6), ("y", 0.4), ("x", 0.1), ("w", 0.9))) == "y"


def test_median_image_needs_images() -> None:
    with pytest.raises(ValueError, match="no images"):
        median_image([])


def test_best_and_worst_order_and_ties() -> None:
    ranked = images(("a", 0.5), ("b", 0.9), ("c", 0.1), ("d", 0.9), ("e", 0.1), ("f", 0.7))
    best, worst = best_and_worst(ranked, 2)
    assert best == ["b", "d"]
    assert worst == ["c", "e"]


def test_best_and_worst_cannot_overlap() -> None:
    with pytest.raises(ValueError, match="cannot show 2 best"):
        best_and_worst(images(("a", 0.1), ("b", 0.2), ("c", 0.3)), 2)


def test_draws_every_figure_with_its_sidecar(reported: Finished, tmp_path: Path) -> None:
    paths = draw(reported, tmp_path)
    assert [p.name for p in paths] == [f"{n}.png" for n in NAMES]
    run = reported.conn.execute(
        "SELECT git_commit, data_hash FROM runs WHERE run_id = ?", (RUN_ID,)
    ).fetchone()
    for path in paths:
        assert path.stat().st_size < COLORS.max_bytes
        sidecar = json.loads(path.with_suffix(".json").read_text())
        assert sidecar["run_id"] == RUN_ID
        assert (sidecar["tag"], sidecar["commit"], sidecar["data_hash"]) == (TAG, *run)
        assert sidecar["development_run"] is False


def test_sidecars_name_the_chosen_images(reported: Finished, tmp_path: Path) -> None:
    draw(reported, tmp_path)
    data = load_run(reported.conn, RUN_ID, reported.dirs.results)
    hero = json.loads((tmp_path / "hero.json").read_text())
    assert hero["images"] == [median_image(data.images)]
    best, worst = best_and_worst(data.images, COLORS.n_best_worst)
    assert json.loads((tmp_path / "best_worst.json").read_text())["images"] == [*worst, *best]


def test_redrawing_gives_the_same_bytes(reported: Finished, tmp_path: Path) -> None:
    first = [p.read_bytes() for p in draw(reported, tmp_path / "a")]
    second = [p.read_bytes() for p in draw(reported, tmp_path / "b")]
    assert first == second


def test_refuses_an_unreported_run(reportable: Finished, tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="not reported"):
        draw(reportable, tmp_path)


def test_draws_a_development_run_when_asked(reportable: Finished, tmp_path: Path) -> None:
    draw(reportable, tmp_path, development=True)
    assert json.loads((tmp_path / "hero.json").read_text())["development_run"] is True


def median_prediction(finished: Finished) -> Path:
    data = load_run(finished.conn, RUN_ID, finished.dirs.results)
    return finished.dirs.predictions(RUN_ID) / f"{median_image(data.images)}.npy"


def test_refuses_a_prediction_that_does_not_match_its_score(
    reported: Finished, tmp_path: Path
) -> None:
    path = median_prediction(reported)
    np.save(path, 1 - np.load(path))
    with pytest.raises(FigureError, match="but the database stores"):
        draw(reported, tmp_path)


def test_refuses_a_missing_prediction(reported: Finished, tmp_path: Path) -> None:
    median_prediction(reported).unlink()
    with pytest.raises(FigureError, match="prediction missing"):
        draw(reported, tmp_path)


def test_refuses_images_the_data_lacks(reported: Finished, tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="has no images"):
        write_figures(
            reported.conn,
            RUN_ID,
            reported.samples[:5],
            reported.dirs,
            tmp_path,
            CONFIG.tables,
            COLORS,
        )


def test_refuses_a_figure_over_the_size_limit(reported: Finished, tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="over the 1000 byte limit"):
        draw(reported, tmp_path, max_bytes=1000)
    assert not (tmp_path / "hero.png").exists()


def test_load_history_needs_rows(reported: Finished) -> None:
    assert sorted(load_history(reported.conn, RUN_ID)) == [1, 2, 3, 4, 5]
    with pytest.raises(FigureError, match="no training history"):
        load_history(reported.conn, "other")


def test_edge_note_says_when_folds_differ(reported: Finished) -> None:
    data = load_run(reported.conn, RUN_ID, reported.dirs.results)
    assert "the same in every fold" in edge_note(data, 3)
    data.folds[1]["thin_edge"] += 1
    assert "differ by fold" in edge_note(data, 3)


def cli(finished: Finished, out: Path, *extra: str) -> int:
    finished.conn.commit()
    root = finished.db.parent
    args = [
        "--db",
        str(finished.db),
        "--results-dir",
        str(finished.dirs.results),
        "--data-root",
        str(root),
        "--checksums",
        str(root / "CHECKSUMS.sha256"),
        "--out-dir",
        str(out),
        *extra,
    ]
    return figures.main(args)


def test_cli_draws_the_reported_run(
    reported: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DRIVE_DIR", raising=False)
    variant = reported.conn.execute("SELECT variant FROM runs").fetchone()[0]
    assert cli(reported, tmp_path / "figs", "--variant", variant) == 0
    assert sorted(p.stem for p in (tmp_path / "figs").glob("*.png")) == sorted(NAMES)


def test_cli_needs_a_reported_run_of_the_variant(
    reported: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DRIVE_DIR", raising=False)
    assert cli(reported, tmp_path / "figs", "--variant", "no_clahe") == 1


def test_cli_refuses_data_the_run_did_not_train_on(
    reported: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DRIVE_DIR", raising=False)
    reported.conn.execute("UPDATE runs SET data_hash = 'other' WHERE run_id = ?", (RUN_ID,))
    variant = reported.conn.execute("SELECT variant FROM runs").fetchone()[0]
    assert cli(reported, tmp_path / "figs", "--variant", variant) == 1


def test_cli_never_draws_a_development_run_into_figures(reportable: Finished) -> None:
    out = REPO / "figures"
    existed = out.exists()
    assert cli(reportable, out, "--allow-unreported", RUN_ID) == 1
    assert out.exists() == existed


def test_cli_needs_a_database(tmp_path: Path) -> None:
    assert figures.main(["--db", str(tmp_path / "absent.db")]) == 1
