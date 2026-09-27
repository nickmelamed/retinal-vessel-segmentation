import importlib.util
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from retinal_vessels.config import ReportedRunsConfig
from retinal_vessels.db import set_reported
from retinal_vessels.provenance import sha256_file
from retinal_vessels.reporting import (
    ReportError,
    checkpoint_problems,
    mark_reported,
    reportability_problems,
)
from tests.fixtures.database import RUN, insert
from tests.fixtures.evaluated_run import REPO, RUN_ID, Finished

SCRIPT = REPO / "scripts" / "mark_reported.py"
REQUIRED = ReportedRunsConfig(gpu_type="Tesla T4", compute_platform="colab")
TAG = "v0.1.0-rc.1"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mark_reported", SCRIPT)
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


def write_checkpoints(finished: Finished) -> None:
    for fold in finished.folds:
        path = finished.dirs.checkpoint(RUN_ID, fold.number)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"weights {fold.number}".encode())
        with finished.conn:
            finished.conn.execute(
                "UPDATE fold_status SET checkpoint_sha256 = ? WHERE run_id = ? AND fold = ?",
                (sha256_file(path), RUN_ID, fold.number),
            )


@pytest.fixture
def reportable(finished: Finished) -> Finished:
    """The fixture's run as if trained on a Colab T4 from a tagged commit, then evaluated."""
    with finished.conn:
        finished.conn.execute(
            "UPDATE runs SET git_tag = ?, gpu_type = ?, compute_platform = ? WHERE run_id = ?",
            (TAG, REQUIRED.gpu_type, REQUIRED.compute_platform, RUN_ID),
        )
    write_checkpoints(finished)
    finished.evaluate()
    return finished


def is_reported(finished: Finished) -> int:
    row = finished.conn.execute(
        "SELECT is_reported FROM runs WHERE run_id = ?", (RUN_ID,)
    ).fetchone()
    return int(row[0])


def update_run(finished: Finished, assignment: str, value: object) -> None:
    with finished.conn:
        finished.conn.execute(f"UPDATE runs SET {assignment} WHERE run_id = ?", (value, RUN_ID))


def refused(finished: Finished, match: str) -> None:
    with pytest.raises(ReportError, match=match):
        mark_reported(finished.conn, RUN_ID, REQUIRED, finished.dirs)
    assert is_reported(finished) == 0


def test_marks_a_run_that_passes_every_check(reportable: Finished) -> None:
    assert reportability_problems(reportable.conn, RUN_ID, REQUIRED, reportable.dirs) == []
    mark_reported(reportable.conn, RUN_ID, REQUIRED, reportable.dirs)
    assert is_reported(reportable) == 1


def test_refuses_an_unknown_run(reportable: Finished) -> None:
    with pytest.raises(ReportError, match="no run other"):
        mark_reported(reportable.conn, "other", REQUIRED, reportable.dirs)


def test_refuses_an_unfinished_run(reportable: Finished) -> None:
    update_run(reportable, "finished_at = ?", None)
    refused(reportable, "not finished: 5 of 5")


def test_refuses_a_run_with_an_incomplete_fold(reportable: Finished) -> None:
    with reportable.conn:
        reportable.conn.execute(
            "UPDATE fold_status SET status = 'running' WHERE run_id = ? AND fold = 3", (RUN_ID,)
        )
    refused(reportable, "4 of 5 folds complete")


def test_refuses_a_dirty_tree(reportable: Finished) -> None:
    update_run(reportable, "git_dirty = ?", 1)
    refused(reportable, "dirty tree")


def test_refuses_an_untagged_commit(reportable: Finished) -> None:
    update_run(reportable, "git_tag = ?", None)
    refused(reportable, "no tag")


@pytest.mark.parametrize("assignment", ["gpu_type = 'NVIDIA L4'", "compute_platform = 'local'"])
def test_refuses_other_hardware(reportable: Finished, assignment: str) -> None:
    with reportable.conn:
        reportable.conn.execute(f"UPDATE runs SET {assignment} WHERE run_id = ?", (RUN_ID,))
    refused(reportable, "reported runs use Tesla T4 on colab")


def test_refuses_a_run_that_was_not_evaluated(finished: Finished) -> None:
    with finished.conn:
        finished.conn.execute(
            "UPDATE runs SET git_tag = ?, gpu_type = ?, compute_platform = ? WHERE run_id = ?",
            (TAG, REQUIRED.gpu_type, REQUIRED.compute_platform, RUN_ID),
        )
    write_checkpoints(finished)
    refused(finished, "0 of 20 out-of-fold images are evaluated")
    refused(finished, "evaluation.json is missing")


def test_refuses_a_partly_evaluated_run(reportable: Finished) -> None:
    with reportable.conn:
        reportable.conn.execute(
            "UPDATE per_image_metrics SET brier = NULL WHERE run_id = ? AND image_id = '21'",
            (RUN_ID,),
        )
    refused(reportable, "19 of 20")


@pytest.mark.parametrize(
    ("code", "match"),
    [
        ({"commit": "beef", "dirty": False, "tag": None}, "written by commit beef"),
        ({"commit": "c0ffee", "dirty": True, "tag": None}, "dirty=True"),
    ],
)
def test_refuses_an_evaluation_from_other_code(
    reportable: Finished, code: dict[str, object], match: str
) -> None:
    path = reportable.dirs.results / RUN_ID / "evaluation.json"
    summary = json.loads(path.read_text())
    summary["code"] = code
    path.write_text(json.dumps(summary))
    refused(reportable, match)


def test_refuses_an_evaluation_of_another_run(reportable: Finished) -> None:
    path = reportable.dirs.results / RUN_ID / "evaluation.json"
    summary = json.loads(path.read_text())
    summary["run_id"] = "other"
    path.write_text(json.dumps(summary))
    refused(reportable, "belongs to run other")


def test_refuses_a_missing_checkpoint(reportable: Finished) -> None:
    reportable.dirs.checkpoint(RUN_ID, 2).unlink()
    refused(reportable, "fold 2: checkpoint missing")


def test_refuses_a_changed_checkpoint(reportable: Finished) -> None:
    reportable.dirs.checkpoint(RUN_ID, 4).write_bytes(b"retrained")
    refused(reportable, "fold 4: checksum mismatch")


def test_refuses_while_the_leakage_audit_finds_rows(reportable: Finished) -> None:
    # A leak in any run of the database blocks reporting, since a leaky
    # database cannot vouch for the runs in it.
    insert(reportable.conn, "runs", RUN)
    for fold in (1, 2):
        insert(
            reportable.conn,
            "fold_assignments",
            {
                "run_id": RUN["run_id"],
                "fold": fold,
                "dataset": "drive",
                "image_id": "21",
                "role": "test",
            },
        )
    reportable.conn.commit()
    refused(reportable, "leakage audit returned 1 rows")


def test_lists_every_problem_at_once(reportable: Finished) -> None:
    update_run(reportable, "git_dirty = ?", 1)
    reportable.dirs.checkpoint(RUN_ID, 1).unlink()
    problems = reportability_problems(reportable.conn, RUN_ID, REQUIRED, reportable.dirs)
    assert len(problems) == 2


def test_checkpoint_problems_are_empty_when_every_file_matches(reportable: Finished) -> None:
    assert checkpoint_problems(reportable.conn, RUN_ID, reportable.dirs) == []


def test_set_reported_needs_an_existing_run(reportable: Finished) -> None:
    with pytest.raises(ValueError, match="no run other"):
        set_reported(reportable.conn, "other")


def test_schema_still_refuses_an_untagged_run(reportable: Finished) -> None:
    # The checks come first, but the schema is the last line if they are bypassed.
    update_run(reportable, "git_tag = ?", None)
    with pytest.raises(Exception, match="reported_needs_tag"):
        set_reported(reportable.conn, RUN_ID)


def cli_args(finished: Finished, tmp_path: Path) -> list[str]:
    config = tmp_path / "reporting.yaml"
    source = (REPO / "configs" / "reporting.yaml").read_text()
    config.write_text(source)
    return [
        "--run-id",
        RUN_ID,
        "--db",
        str(finished.db),
        "--results-dir",
        str(finished.dirs.results),
        "--models-dir",
        str(finished.dirs.models),
        "--config",
        str(config),
    ]


def test_cli_marks_the_run(reportable: Finished, tmp_path: Path) -> None:
    reportable.conn.commit()
    assert load_script().main(cli_args(reportable, tmp_path)) == 0
    assert is_reported(reportable) == 1


def test_cli_fails_and_logs_the_problems(
    reportable: Finished, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    update_run(reportable, "git_dirty = ?", 1)
    assert load_script().main(cli_args(reportable, tmp_path)) == 1
    assert "dirty tree" in caplog.text
    assert is_reported(reportable) == 0


def test_cli_needs_a_database(reportable: Finished, tmp_path: Path) -> None:
    args = cli_args(reportable, tmp_path)
    args[args.index("--db") + 1] = str(tmp_path / "absent.db")
    assert load_script().main(args) == 1


def test_cli_needs_a_valid_config(reportable: Finished, tmp_path: Path) -> None:
    args = cli_args(reportable, tmp_path)
    Path(args[-1]).write_text("tables: {}\n")
    assert load_script().main(args) == 1
