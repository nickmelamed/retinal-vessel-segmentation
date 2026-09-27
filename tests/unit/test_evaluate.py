import json
import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from retinal_vessels import evaluate
from retinal_vessels.evaluate import EvaluationError, evaluate_run, latest_finished_run
from retinal_vessels.metrics import auc_roc, reliability, thin_edge
from retinal_vessels.provenance import (
    GitState,
)
from retinal_vessels.train import save_predictions
from tests.fixtures.evaluated_run import (
    CODE,
    RUN_ID,
    T1,
    THRESHOLD,
    Finished,
)


@pytest.fixture(autouse=True)
def restore_root_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # main() installs the package's log handler on the root logger.
    monkeypatch.delenv("DRIVE_DIR", raising=False)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


def test_fills_every_evaluated_column(finished: Finished) -> None:
    assert all(v is None for row in finished.rows() for v in row[1:])
    finished.evaluate()
    rows = finished.rows()
    assert len(rows) == 20
    for _, roc, pr, brier, thin, thick in rows:
        assert roc is not None and pr is not None and brier is not None
        assert thin is not None or thick is not None
    first = finished.sample(rows[0][0])
    assert first.label is not None
    expected = auc_roc(finished.probabilities[first.image_id], first.label, first.fov)
    assert rows[0][1] == expected


def test_writes_the_run_summary(finished: Finished) -> None:
    path = finished.evaluate()
    assert path == finished.dirs.results / RUN_ID / "evaluation.json"
    summary = json.loads(path.read_text())
    assert summary["run_id"] == RUN_ID
    assert summary["code"] == {"commit": "c0ffee", "dirty": False, "tag": None}
    assert summary["evaluation"] == {"reliability_bins": 10, "thin_quantile": 0.5}
    assert [f["fold"] for f in summary["folds"]] == [1, 2, 3, 4, 5]

    fold = finished.folds[0]
    tuning = [finished.sample(i) for i in (*fold.train, *fold.val)]
    labels = [s.label for s in tuning if s.label is not None]
    edge = thin_edge(labels, [s.fov for s in tuning], 0.5)
    assert summary["folds"][0]["thin_edge"] == edge
    assert summary["folds"][0]["threshold"] == THRESHOLD

    ids = sorted(finished.probabilities)
    samples = [finished.sample(i) for i in ids]
    table = reliability(
        [finished.probabilities[i] for i in ids],
        [s.label for s in samples if s.label is not None],
        [s.fov for s in samples],
        10,
    )
    assert summary["reliability"]["counts"] == list(table.counts)
    assert summary["reliability"]["pooled_brier"] == pytest.approx(table.pooled_brier)
    assert sum(table.counts) == sum(int(s.fov.sum()) for s in samples)
    assert not list(path.parent.glob("*.partial"))


def test_rerunning_gives_the_same_results(finished: Finished) -> None:
    path = finished.evaluate()
    rows, text = finished.rows(), path.read_text()
    finished.evaluate()
    assert finished.rows() == rows
    assert path.read_text() == text


def test_rewrites_the_manifest_from_the_database(finished: Finished) -> None:
    manifest = finished.dirs.results / RUN_ID / "manifest.json"
    assert not manifest.exists()
    finished.evaluate()
    assert json.loads(manifest.read_text())["finished_at"] == T1


def assert_nothing_written(finished: Finished, match: str, code: GitState = CODE) -> None:
    with pytest.raises(EvaluationError, match=match):
        finished.evaluate(code)
    assert all(v is None for row in finished.rows() for v in row[1:])
    assert not (finished.dirs.results / RUN_ID / "evaluation.json").exists()
    assert not (finished.dirs.results / RUN_ID / "manifest.json").exists()


def test_refuses_an_unfinished_run(finished: Finished) -> None:
    finished.conn.execute("UPDATE runs SET finished_at = NULL")
    assert_nothing_written(finished, "5 of 5 folds complete")


def test_refuses_a_run_missing_a_fold(finished: Finished) -> None:
    finished.conn.execute("DELETE FROM fold_status WHERE fold = 5")
    assert_nothing_written(finished, "4 of 5 folds complete")


def test_refuses_an_unknown_run(finished: Finished) -> None:
    with pytest.raises(EvaluationError, match="no run 'other'"):
        evaluate_run(finished.conn, "other", finished.samples, finished.report, finished.dirs, CODE)


def first_test_file(finished: Finished) -> Path:
    return finished.dirs.predictions(RUN_ID) / f"{finished.folds[0].test[0]}.npy"


def test_refuses_a_missing_prediction(finished: Finished) -> None:
    first_test_file(finished).unlink()
    assert_nothing_written(finished, "is missing")


def test_refuses_a_changed_prediction(finished: Finished) -> None:
    path = first_test_file(finished)
    np.save(path, np.load(path) * np.float32(0.5))
    assert_nothing_written(finished, "does not match")


def test_refuses_a_fold_without_hashes(finished: Finished) -> None:
    finished.dirs.prediction_hashes(RUN_ID, 2).unlink()
    assert_nothing_written(finished, "fold 2 has no prediction hashes")


def test_refuses_hashes_for_other_images(finished: Finished) -> None:
    fold = finished.folds[0]
    kept = {i: finished.probabilities[i] for i in fold.test[:-1]}
    save_predictions(finished.dirs, RUN_ID, fold.number, kept)
    assert_nothing_written(finished, "but fold 1 tests")


def test_refuses_swapped_predictions_with_matching_hashes(finished: Finished) -> None:
    # Hashes written after a swap pass, so only the recomputed metrics catch it.
    fold = finished.folds[0]
    a, b = fold.test[:2]
    swapped = {i: finished.probabilities[i] for i in fold.test}
    swapped[a], swapped[b] = swapped[b], swapped[a]
    save_predictions(finished.dirs, RUN_ID, fold.number, swapped)
    assert_nothing_written(finished, "does not reproduce the metrics stored")


def test_refuses_a_prediction_of_the_wrong_dtype(finished: Finished) -> None:
    fold = finished.folds[0]
    probs = {i: finished.probabilities[i] for i in fold.test}
    probs[fold.test[0]] = probs[fold.test[0]].astype(np.float64)
    save_predictions(finished.dirs, RUN_ID, fold.number, probs)
    assert_nothing_written(finished, "holds float64")


def test_refuses_data_that_differs_from_the_run(finished: Finished) -> None:
    checked = dict(finished.report.checked)
    first = next(iter(checked))
    checked[first] = "0" * 64
    finished.report = replace(finished.report, checked=checked)
    assert_nothing_written(finished, "does not match the data")


def test_latest_finished_run(finished: Finished) -> None:
    assert latest_finished_run(finished.conn) == RUN_ID
    finished.conn.execute("UPDATE runs SET finished_at = NULL")
    assert latest_finished_run(finished.conn) is None


def cli(finished: Finished, root: Path, *extra: str) -> int:
    return evaluate.main(
        [
            "--data-root",
            str(root),
            "--checksums",
            str(root / "CHECKSUMS.sha256"),
            "--db",
            str(finished.db),
            "--results-dir",
            str(finished.dirs.results),
            *extra,
        ]
    )


def test_main_evaluates_the_latest_finished_run(
    finished: Finished, synthetic_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "git_state", lambda repo: CODE)
    finished.conn.commit()
    assert cli(finished, synthetic_data_root) == 0
    assert (finished.dirs.results / RUN_ID / "evaluation.json").is_file()
    assert all(row[1] is not None for row in finished.rows())


def test_main_reports_failures(
    finished: Finished, synthetic_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert cli(finished, synthetic_data_root, "--run-id", "other") == 1
    assert "no run 'other'" in caplog.text
    finished.conn.execute("UPDATE runs SET finished_at = NULL")
    finished.conn.commit()
    assert cli(finished, synthetic_data_root) == 1
    assert "no finished run" in caplog.text
    assert cli(finished, synthetic_data_root / "missing") == 1


def test_main_needs_the_database(synthetic_data_root: Path, tmp_path: Path) -> None:
    assert evaluate.main(["--db", str(tmp_path / "none.db")]) == 1


def test_thin_edge_sees_only_training_and_validation_labels(
    finished: Finished, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Held-out labels must never set the edge (rule 3). The synthetic images
    # give every fold the same edge whichever labels are used, so check the
    # labels passed in rather than the value that comes out.
    seen: list[set[int]] = []

    def recording_thin_edge(labels: list[np.ndarray], fovs: list[np.ndarray], q: float) -> float:
        seen.append({id(label) for label in labels})
        return thin_edge(labels, fovs, q)

    monkeypatch.setattr(evaluate, "thin_edge", recording_thin_edge)
    finished.evaluate()
    by_id = {id(s.label): s.image_id for s in finished.samples}
    assert len(seen) == len(finished.folds)
    for fold, ids in zip(finished.folds, seen, strict=True):
        assert {by_id[i] for i in ids} == {*fold.train, *fold.val}
        assert len(ids) == len(fold.train) + len(fold.val)


def test_a_clean_run_needs_a_clean_tree(finished: Finished) -> None:
    dirty = GitState(commit="c0ffee", dirty=True, tag=None)
    assert_nothing_written(finished, "c0ffee with uncommitted changes", dirty)


def test_a_clean_run_needs_its_own_commit(finished: Finished) -> None:
    later = GitState(commit="beef", dirty=False, tag="v9")
    assert_nothing_written(finished, "clean tree at c0ffee, but this tree is at beef", later)


def test_a_dirty_run_can_be_evaluated_from_any_tree(finished: Finished) -> None:
    finished.conn.execute("UPDATE runs SET git_dirty = 1")
    elsewhere = GitState(commit="beef", dirty=True, tag=None)
    summary = json.loads(finished.evaluate(elsewhere).read_text())
    assert summary["code"] == {"commit": "beef", "dirty": True, "tag": None}


def test_main_refuses_a_clean_run_from_another_commit(
    finished: Finished, synthetic_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "git_state", lambda repo: replace(CODE, commit="beef"))
    finished.conn.commit()
    assert cli(finished, synthetic_data_root) == 1
    assert all(row[1] is None for row in finished.rows())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.5, 1.5])
def test_refuses_a_probability_that_is_not_in_the_unit_interval(
    finished: Finished, bad: float
) -> None:
    # A NaN would reach SQLite as NULL and pass the schema's CHECK, so the
    # image would silently lose its metrics. The bad value goes where it
    # leaves the recomputed confusion metrics the same, so only the range
    # check can catch it: below the threshold on background, or at or above
    # it on a vessel.
    fold = finished.folds[0]
    probs = {i: finished.probabilities[i].copy() for i in fold.test}
    sample = finished.sample(fold.test[0])
    assert sample.label is not None
    prob = probs[sample.image_id]
    if bad > 1:
        where = sample.fov & sample.label & (prob >= THRESHOLD)
    else:
        where = sample.fov & ~sample.label & (prob < THRESHOLD)
    rows, cols = np.nonzero(where)
    probs[sample.image_id][rows[0], cols[0]] = bad
    save_predictions(finished.dirs, RUN_ID, fold.number, probs)
    assert_nothing_written(finished, r"outside \[0, 1\]")
