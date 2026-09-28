import importlib.util
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from retinal_vessels.db import set_reported
from retinal_vessels.reporting import (
    ReportError,
    checkpoint_problems,
    mark_reported,
    reportability_problems,
)
from tests.fixtures.database import RUN, insert
from tests.fixtures.evaluated_run import REPO, REQUIRED, RUN_ID, TAG, Finished, write_checkpoints

SCRIPT = REPO / "scripts" / "mark_reported.py"


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
        mark_reported(finished.conn, RUN_ID, REQUIRED, finished.dirs, finished.checksums)
    assert is_reported(finished) == 0


def test_marks_a_run_that_passes_every_check(reportable: Finished) -> None:
    assert (
        reportability_problems(
            reportable.conn, RUN_ID, REQUIRED, reportable.dirs, reportable.checksums
        )
        == []
    )
    mark_reported(reportable.conn, RUN_ID, REQUIRED, reportable.dirs, reportable.checksums)
    assert is_reported(reportable) == 1


def test_refuses_an_unknown_run(reportable: Finished) -> None:
    with pytest.raises(ReportError, match="no run other"):
        mark_reported(reportable.conn, "other", REQUIRED, reportable.dirs, reportable.checksums)


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


def test_refuses_a_run_that_never_tested_a_labeled_image(reportable: Finished) -> None:
    # Dropping image 34 everywhere keeps the tested and evaluated sets equal
    # and the leakage audit empty, so only the labeled-image check sees it.
    with reportable.conn:
        for table in ("fold_assignments", "per_image_metrics"):
            reportable.conn.execute(
                f"DELETE FROM {table} WHERE run_id = ? AND image_id = '34'", (RUN_ID,)
            )
    refused(reportable, r"never tested labeled images \['34'\]")


def test_refuses_a_determinism_flag_the_config_did_not_ask_for(reportable: Finished) -> None:
    update_run(reportable, "deterministic_ops = ?", 0)
    refused(reportable, "deterministic_ops=False, but its config asked for True")


def test_refuses_data_that_does_not_match_the_committed_checksums(reportable: Finished) -> None:
    update_run(reportable, "data_hash = ?", "0" * 64)
    refused(reportable, "data hash does not match the checksums")


def test_refuses_when_the_checksums_cannot_be_read(reportable: Finished, tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="cannot read the committed checksums"):
        mark_reported(reportable.conn, RUN_ID, REQUIRED, reportable.dirs, tmp_path / "absent")


def test_refuses_mismatched_images_even_when_the_counts_agree(reportable: Finished) -> None:
    # Image 21 loses its test role and image 22 its metric row, so 19 images
    # are tested and 19 rows are stored, but they are not the same 19.
    with reportable.conn:
        reportable.conn.execute(
            "DELETE FROM fold_assignments WHERE run_id = ? AND image_id = '21' AND role = 'test'",
            (RUN_ID,),
        )
        reportable.conn.execute(
            "DELETE FROM per_image_metrics WHERE run_id = ? AND image_id = '22'", (RUN_ID,)
        )
    problems = reportability_problems(
        reportable.conn, RUN_ID, REQUIRED, reportable.dirs, reportable.checksums
    )
    assert "18 of 19 out-of-fold images are evaluated" in problems
    assert "metric rows exist for images the run never tested: ['21']" in problems


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{", "cannot be read"),
        ("[]", "is not an evaluation summary"),
        ('{"run_id": "x"}', "is not an evaluation summary"),
    ],
)
def test_lists_an_unreadable_evaluation_summary(
    reportable: Finished, content: str, match: str
) -> None:
    (reportable.dirs.results / RUN_ID / "evaluation.json").write_text(content)
    problems = reportability_problems(
        reportable.conn, RUN_ID, REQUIRED, reportable.dirs, reportable.checksums
    )
    assert any(match in p for p in problems)
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
    problems = reportability_problems(
        reportable.conn, RUN_ID, REQUIRED, reportable.dirs, reportable.checksums
    )
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
        "--checksums",
        str(finished.checksums),
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
