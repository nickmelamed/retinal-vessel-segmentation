import importlib.util
import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from retinal_vessels.config import TablesConfig, load_reporting_config
from retinal_vessels.db import set_reported
from retinal_vessels.tables import (
    TableError,
    bootstrap_mean_ci,
    fmt,
    markdown_table,
    reported_run_ids,
    summarize,
    write_tables,
)
from tests.fixtures.database import RUN, insert
from tests.fixtures.evaluated_run import REPO, RUN_ID, TAG, Finished

SCRIPT = REPO / "scripts" / "make_tables.py"
CONFIG = load_reporting_config(REPO / "configs" / "reporting.yaml").tables
SMALL = TablesConfig(decimals=3, bootstrap_resamples=2000, bootstrap_seed=1, ci_level=0.95)
FILES = {
    "headline.md",
    "per_image.md",
    "per_fold.md",
    "thin_thick.md",
    "calibration.md",
    "pathology.md",
    "provenance.md",
    "dataset.md",
}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_tables", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def tables(finished: Finished, out: Path, development: bool = False) -> dict[str, str]:
    paths = write_tables(
        finished.conn, [RUN_ID], finished.dirs.results, out, CONFIG, finished.db, development
    )
    return {p.name: p.read_text() for p in paths}


def column(conn: sqlite3.Connection, name: str) -> list[float]:
    rows = conn.execute(
        f"SELECT {name} FROM per_image_metrics WHERE run_id = ? ORDER BY image_id", (RUN_ID,)
    ).fetchall()
    return [float(r[0]) for r in rows]


def test_constant_values_have_a_zero_width_interval() -> None:
    assert bootstrap_mean_ci([0.7] * 20, 500, 0, 0.95) == pytest.approx((0.7, 0.7))


def test_bootstrap_is_seeded() -> None:
    values = list(np.linspace(0, 1, 20))
    assert bootstrap_mean_ci(values, 500, 3, 0.95) == bootstrap_mean_ci(values, 500, 3, 0.95)
    assert bootstrap_mean_ci(values, 500, 3, 0.95) != bootstrap_mean_ci(values, 500, 4, 0.95)


def test_bootstrap_needs_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean_ci([], 10, 0, 0.95)


@given(
    st.lists(st.floats(0, 1, allow_nan=False), min_size=2, max_size=30),
    st.floats(0.5, 0.99),
)
def test_interval_is_ordered_and_within_the_data(values: list[float], level: float) -> None:
    low, high = bootstrap_mean_ci(values, 200, 0, level)
    assert min(values) - 1e-12 <= low <= high <= max(values) + 1e-12


def test_wider_level_gives_a_wider_interval() -> None:
    values = list(np.random.default_rng(0).uniform(0, 1, 20))
    narrow = bootstrap_mean_ci(values, 2000, 0, 0.5)
    wide = bootstrap_mean_ci(values, 2000, 0, 0.95)
    assert wide[0] <= narrow[0] and narrow[1] <= wide[1]


def test_summarize_hand_worked() -> None:
    s = summarize([1.0, 2.0, 3.0, 4.0], SMALL)
    assert (s.n, s.mean, s.median, s.minimum, s.maximum) == (4, 2.5, 2.5, 1.0, 4.0)
    # Sample SD: sqrt(((1.5**2 + 0.5**2) * 2) / 3).
    assert s.sd == pytest.approx((5 / 3) ** 0.5)
    assert s.ci_low <= s.mean <= s.ci_high


def test_summarize_needs_two_values() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        summarize([0.5], SMALL)


def test_fmt() -> None:
    assert fmt(0.12345, 3) == "0.123"
    assert fmt(1.0, 3) == "1.000"
    assert fmt(None, 3) == "NA"


def test_markdown_table() -> None:
    assert markdown_table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |\n"


def test_writes_every_table(reported: Finished, tmp_path: Path) -> None:
    assert set(tables(reported, tmp_path)) == FILES


def test_headline_matches_the_database(reported: Finished, tmp_path: Path) -> None:
    text = tables(reported, tmp_path)["headline.md"]
    dice = summarize(column(reported.conn, "dice"), CONFIG)
    row = next(line for line in text.splitlines() if line.startswith("| Dice |"))
    assert row == (
        f"| Dice | 20 | {fmt(dice.mean, 3)} | {fmt(dice.ci_low, 3)} to {fmt(dice.ci_high, 3)} | "
        f"{fmt(dice.sd, 3)} | {fmt(dice.median, 3)} | {fmt(dice.minimum, 3)} | "
        f"{fmt(dice.maximum, 3)} |"
    )


def test_headline_gives_the_accuracy_of_predicting_no_vessel(
    reported: Finished, tmp_path: Path
) -> None:
    text = tables(reported, tmp_path)["headline.md"]
    fractions = reported.conn.execute(
        "SELECT vessel_fraction_in_fov FROM images WHERE image_id IN "
        "(SELECT image_id FROM per_image_metrics WHERE run_id = ?)",
        (RUN_ID,),
    ).fetchall()
    expected = fmt(float(np.mean([1 - f[0] for f in fractions])), 3)
    assert f"mean accuracy of {expected}" in text


def test_every_table_names_its_run(reported: Finished, tmp_path: Path) -> None:
    run = reported.conn.execute(
        "SELECT git_commit, data_hash FROM runs WHERE run_id = ?", (RUN_ID,)
    ).fetchone()
    for name, text in tables(reported, tmp_path).items():
        last = text.rstrip().splitlines()[-1]
        assert RUN_ID in last and TAG in last and run[0] in last and run[1] in last, name


def test_regenerating_gives_the_same_bytes(reported: Finished, tmp_path: Path) -> None:
    assert tables(reported, tmp_path / "a") == tables(reported, tmp_path / "b")


def test_per_image_lists_each_image_once(reported: Finished, tmp_path: Path) -> None:
    rows = [
        line
        for line in tables(reported, tmp_path)["per_image.md"].splitlines()
        if line.startswith("| 2") or line.startswith("| 3") or line.startswith("| 40")
    ]
    assert len(rows) == 20


def test_pathology_quotes_the_official_notes(reported: Finished, tmp_path: Path) -> None:
    text = tables(reported, tmp_path)["pathology.md"]
    notes = reported.conn.execute(
        "SELECT abnormality_note FROM images WHERE has_abnormality = 1"
    ).fetchall()
    assert len(notes) == 3
    for (note,) in notes:
        assert note in text
    assert "descriptive only" in text


def test_calibration_counts_every_fov_pixel(reported: Finished, tmp_path: Path) -> None:
    text = tables(reported, tmp_path)["calibration.md"]
    counts = [
        int(line.split("|")[2])
        for line in text.splitlines()
        if line.startswith("| 0.") and " to " in line
    ]
    fov = sum(s.fov.sum() for s in reported.samples)
    assert sum(counts) == fov


def test_dataset_pools_the_vessel_fraction(reported: Finished, tmp_path: Path) -> None:
    text = tables(reported, tmp_path)["dataset.md"]
    labels = sum(int((s.label & s.fov).sum()) for s in reported.samples if s.label is not None)
    fov = sum(int(s.fov.sum()) for s in reported.samples)
    assert f"| 20 | {fmt(labels / fov, 3)} |" in text


def test_says_when_every_fold_shares_an_edge(reported: Finished, tmp_path: Path) -> None:
    text = tables(reported, tmp_path)["per_fold.md"]
    assert "Every fold set the same thin/thick edge" in text


def test_says_plainly_when_fold_edges_differ(reported: Finished, tmp_path: Path) -> None:
    path = reported.dirs.results / RUN_ID / "evaluation.json"
    summary = json.loads(path.read_text())
    summary["folds"][1]["thin_edge"] = summary["folds"][0]["thin_edge"] + 1
    path.write_text(json.dumps(summary))
    written = tables(reported, tmp_path)
    for name in ("per_fold.md", "thin_thick.md"):
        assert "different set of vessels in each fold" in written[name]


def test_refuses_a_summary_that_disagrees_with_the_database(
    reported: Finished, tmp_path: Path
) -> None:
    path = reported.dirs.results / RUN_ID / "evaluation.json"
    summary = json.loads(path.read_text())
    summary["folds"][2]["threshold"] = 0.25
    path.write_text(json.dumps(summary))
    with pytest.raises(TableError, match="does not match the thresholds"):
        tables(reported, tmp_path)


def test_refuses_an_unevaluated_run(finished: Finished, tmp_path: Path) -> None:
    with pytest.raises(TableError, match="not evaluated"):
        tables(finished, tmp_path, development=True)


def test_refuses_an_unfinished_run(reported: Finished, tmp_path: Path) -> None:
    reported.conn.execute("UPDATE runs SET is_reported = 0, finished_at = NULL")
    with pytest.raises(TableError, match="not finished"):
        tables(reported, tmp_path, development=True)


def test_refuses_an_unreported_run(reportable: Finished, tmp_path: Path) -> None:
    with pytest.raises(TableError, match="only reported runs"):
        tables(reportable, tmp_path)


def test_development_tables_say_so(reportable: Finished, tmp_path: Path) -> None:
    for text in tables(reportable, tmp_path, development=True).values():
        assert text.startswith("> Development run, not reported.")


def provenance_row(text: str, field: str) -> str:
    row = next(line for line in text.splitlines() if line.startswith(f"| {field} |"))
    return row.split("|")[2].strip()


def test_provenance_separates_the_training_and_evaluation_trees(
    reportable: Finished, tmp_path: Path
) -> None:
    # A development run trained from a dirty tree can be evaluated from a
    # clean one, and the table must not report the training tree as clean.
    reportable.conn.execute("UPDATE runs SET git_dirty = 1 WHERE run_id = ?", (RUN_ID,))
    text = tables(reportable, tmp_path, development=True)["provenance.md"]
    assert provenance_row(text, "Trained from a clean tree") == "no"
    assert provenance_row(text, "Evaluated from a clean tree") == "yes"


def test_provenance_of_a_reported_run_is_clean_throughout(
    reported: Finished, tmp_path: Path
) -> None:
    text = tables(reported, tmp_path)["provenance.md"]
    assert provenance_row(text, "Trained from a clean tree") == "yes"
    assert provenance_row(text, "Evaluated from a clean tree") == "yes"


def test_refuses_two_runs_of_one_variant(reported: Finished, tmp_path: Path) -> None:
    variant = reported.conn.execute(
        "SELECT variant FROM runs WHERE run_id = ?", (RUN_ID,)
    ).fetchone()[0]
    insert(reported.conn, "runs", {**RUN, "variant": variant})
    with pytest.raises(TableError, match="more than one run per variant"):
        write_tables(
            reported.conn, [RUN_ID, RUN_ID], reported.dirs.results, tmp_path, CONFIG, reported.db
        )


def test_refuses_no_runs(reported: Finished, tmp_path: Path) -> None:
    with pytest.raises(TableError, match="no runs"):
        write_tables(reported.conn, [], reported.dirs.results, tmp_path, CONFIG, reported.db)


def test_reported_run_ids(reported: Finished) -> None:
    insert(reported.conn, "runs", RUN)
    assert reported_run_ids(reported.conn) == [RUN_ID]


def cli(finished: Finished, out: Path, *extra: str) -> int:
    finished.conn.commit()
    args = ["--db", str(finished.db), "--results-dir", str(finished.dirs.results)]
    return int(load_script().main([*args, "--out-dir", str(out), *extra]))


def test_cli_writes_the_reported_tables(reported: Finished, tmp_path: Path) -> None:
    assert cli(reported, tmp_path / "tables") == 0
    assert {p.name for p in (tmp_path / "tables").iterdir()} == FILES


def test_cli_with_no_reported_run_fails(reportable: Finished, tmp_path: Path) -> None:
    assert cli(reportable, tmp_path / "tables") == 1


def test_cli_tabulates_a_development_run_elsewhere(reportable: Finished, tmp_path: Path) -> None:
    assert cli(reportable, tmp_path / "dev", "--allow-unreported", RUN_ID) == 0
    assert (tmp_path / "dev" / "headline.md").read_text().startswith("> Development run")


def test_cli_never_writes_a_development_run_into_results_tables(
    reportable: Finished,
) -> None:
    out = REPO / "results" / "tables"
    existed = out.exists()
    assert cli(reportable, out, "--allow-unreported", RUN_ID) == 1
    assert out.exists() == existed


def test_cli_needs_a_database(tmp_path: Path) -> None:
    assert load_script().main(["--db", str(tmp_path / "absent.db")]) == 1


def test_cli_needs_a_valid_config(reported: Finished, tmp_path: Path) -> None:
    bad = tmp_path / "reporting.yaml"
    bad.write_text("tables: {}\n")
    assert cli(reported, tmp_path / "tables", "--config", str(bad)) == 1
