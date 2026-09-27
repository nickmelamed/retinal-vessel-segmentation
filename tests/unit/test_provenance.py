import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from retinal_vessels.provenance import (
    ChecksumReport,
    build_manifest,
    check_or_write_checksums,
    compute_checksums,
    compute_platform,
    config_hash,
    data_hash,
    environment,
    format_checksums,
    git_state,
    new_run_id,
    read_checksums,
    sha256_file,
    utc_timestamp,
    verify_checksums,
    write_manifest,
)

FAKE_ENV = {"python_version": "3.13.0", "device": "cpu", "gpu_type": "cpu"}


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args]
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    (path / "a.txt").write_text("a\n")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "init")
    return path


@pytest.fixture
def drive(synthetic_data_root: Path) -> dict[str, Path]:
    return {"DRIVE": synthetic_data_root / "DRIVE"}


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"vessel" * 1000)
    assert sha256_file(path) == hashlib.sha256(b"vessel" * 1000).hexdigest()


def test_checksums_round_trip_through_the_manifest_format(tmp_path: Path) -> None:
    entries = {"DRIVE/b.gif": "b" * 64, "DRIVE/a.tif": "a" * 64}
    path = tmp_path / "CHECKSUMS.sha256"
    path.write_text(format_checksums(entries))
    assert read_checksums(path) == entries
    assert path.read_text().splitlines()[0].endswith("DRIVE/a.tif")


@pytest.mark.parametrize(
    "line",
    ["not a checksum line", "abc  DRIVE/a.tif", f"{'g' * 64}  DRIVE/a.tif", f"{'a' * 64}  "],
)
def test_read_checksums_rejects_malformed_lines(tmp_path: Path, line: str) -> None:
    path = tmp_path / "CHECKSUMS.sha256"
    path.write_text(line + "\n")
    with pytest.raises(ValueError, match=":1:"):
        read_checksums(path)


def test_read_checksums_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "CHECKSUMS.sha256"
    path.write_text(f"{'a' * 64}  DRIVE/x\n\n{'b' * 64}  DRIVE/x\n")
    with pytest.raises(ValueError, match="listed twice"):
        read_checksums(path)


def test_data_hash_ignores_order_but_not_content() -> None:
    a = {"DRIVE/x": "1" * 64, "DRIVE/y": "2" * 64}
    b = {"DRIVE/y": "2" * 64, "DRIVE/x": "1" * 64}
    assert data_hash(a) == data_hash(b)
    assert data_hash(a) != data_hash({**a, "DRIVE/y": "3" * 64})


def test_compute_checksums_skips_hidden_files(drive: dict[str, Path]) -> None:
    (drive["DRIVE"] / ".DS_Store").write_text("finder")
    (drive["DRIVE"] / "training" / ".cache").mkdir()
    (drive["DRIVE"] / "training" / ".cache" / "x").write_text("x")
    entries = compute_checksums(drive)
    assert len(entries) == 100
    assert all(rel.startswith("DRIVE/") for rel in entries)


def test_verify_passes_on_untouched_data(drive: dict[str, Path]) -> None:
    report = verify_checksums(compute_checksums(drive), drive)
    assert report.ok
    assert report.problems() == []
    assert len(report.checked) == 100


def test_verify_reports_missing_mismatched_and_unlisted(drive: dict[str, Path]) -> None:
    expected = compute_checksums(drive)
    root = drive["DRIVE"]
    (root / "training/images/21_training.tif").unlink()
    (root / "training/mask/22_training_mask.gif").write_bytes(b"corrupt")
    (root / "training/images/99_training.tif").write_bytes(b"extra")
    report = verify_checksums(expected, drive)
    assert not report.ok
    assert report.missing == ["DRIVE/training/images/21_training.tif"]
    assert report.mismatched == ["DRIVE/training/mask/22_training_mask.gif"]
    assert report.unlisted == ["DRIVE/training/images/99_training.tif"]
    assert len(report.problems()) == 3


def test_verify_reports_everything_missing_when_directory_is_absent(tmp_path: Path) -> None:
    expected = {"DRIVE/a": "a" * 64}
    report = verify_checksums(expected, {"DRIVE": tmp_path / "nowhere"})
    assert report.missing == ["DRIVE/a"]


def test_verify_ignores_datasets_that_were_not_given(drive: dict[str, Path]) -> None:
    expected = {**compute_checksums(drive), "STARE/im0001.ppm": "f" * 64}
    assert verify_checksums(expected, drive).ok


def test_verify_rejects_a_dataset_the_manifest_does_not_list(drive: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="STARE"):
        verify_checksums(compute_checksums(drive), {**drive, "STARE": drive["DRIVE"]})
    with pytest.raises(ValueError, match="no dataset"):
        verify_checksums({}, {})


def test_check_or_write_writes_a_missing_manifest(tmp_path: Path, drive: dict[str, Path]) -> None:
    path = tmp_path / "CHECKSUMS.sha256"
    written, report = check_or_write_checksums(path, drive)
    assert written
    assert report.ok
    assert read_checksums(path) == compute_checksums(drive)


def test_check_or_write_never_overwrites(tmp_path: Path, drive: dict[str, Path]) -> None:
    path = tmp_path / "CHECKSUMS.sha256"
    check_or_write_checksums(path, drive)
    before = path.read_bytes()
    (drive["DRIVE"] / "test/mask/01_test_mask.gif").write_bytes(b"changed")
    written, report = check_or_write_checksums(path, drive)
    assert not written
    assert report.mismatched == ["DRIVE/test/mask/01_test_mask.gif"]
    assert path.read_bytes() == before


def test_check_or_write_refuses_to_write_from_nothing(tmp_path: Path) -> None:
    path = tmp_path / "CHECKSUMS.sha256"
    with pytest.raises(FileNotFoundError):
        check_or_write_checksums(path, {"DRIVE": tmp_path / "nowhere"})
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no files"):
        check_or_write_checksums(path, {"DRIVE": tmp_path / "empty"})
    assert not path.exists()


def test_git_state_tracks_commit_and_dirty_flag(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert git_state(repo).commit == head
    assert git_state(repo).dirty is False
    (repo / "a.txt").write_text("changed\n")
    assert git_state(repo).dirty is True


def test_git_state_reports_the_tag_on_head(repo: Path) -> None:
    assert git_state(repo).tag is None
    _git(repo, "tag", "-a", "v0.1.0", "-m", "release")
    assert git_state(repo).tag == "v0.1.0"
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "after the tag")
    assert git_state(repo).tag is None


def test_git_state_counts_untracked_files_as_dirty(repo: Path) -> None:
    (repo / "new.py").write_text("")
    assert git_state(repo).dirty is True


def test_git_state_fails_loudly_outside_a_repo(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="rev-parse"):
        git_state(tmp_path)


def test_manifest_captures_commit_dirty_flag_and_data_checksum(
    repo: Path, drive: dict[str, Path], tmp_path: Path
) -> None:
    checksums = compute_checksums(drive)
    head = _git(repo, "rev-parse", "HEAD").strip()
    kwargs = dict(
        variant="baseline",
        config={"b": 1, "a": 2},
        seed=42,
        deterministic_ops=True,
        repo=repo,
        data=verify_checksums(checksums, drive),
        env=FAKE_ENV,
        started_at=datetime(2026, 9, 26, 12, tzinfo=UTC),
    )
    clean = build_manifest(run_id="r1", **kwargs)  # type: ignore[arg-type]
    assert clean.git_commit == head
    assert clean.git_dirty is False
    assert clean.data_hash == data_hash(checksums)
    assert clean.config_hash == config_hash({"a": 2, "b": 1})
    assert clean.environment == FAKE_ENV

    (repo / "a.txt").write_text("edited\n")
    dirty = build_manifest(run_id="r2", **kwargs)  # type: ignore[arg-type]
    assert dirty.git_dirty is True

    path = write_manifest(clean, tmp_path / "results")
    assert path == tmp_path / "results" / "r1" / "manifest.json"
    saved = json.loads(path.read_text())
    assert saved["git_commit"] == head
    assert saved["git_dirty"] is False
    assert saved["git_tag"] is None
    assert saved["data_hash"] == data_hash(checksums)
    assert saved["started_at"] == "2026-09-26T12:00:00Z"
    assert saved["finished_at"] is None


def test_config_hash_ignores_key_order() -> None:
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_run_ids_sort_by_time_and_are_unique() -> None:
    early = new_run_id(datetime(2026, 1, 1, tzinfo=UTC))
    late = new_run_id(datetime(2026, 1, 2, tzinfo=UTC))
    assert early.startswith("20260101T000000Z-")
    assert early < late
    assert new_run_id() != new_run_id()


def test_compute_platform_detects_colab() -> None:
    assert compute_platform({"COLAB_RELEASE_TAG": "release-colab_20260819"}) == "colab"
    assert compute_platform({}) == "local"


def test_empty_report_is_ok() -> None:
    assert ChecksumReport(checked={}).ok


@pytest.mark.slow
def test_environment_describes_this_machine() -> None:
    env = environment()
    assert env["python_version"].startswith("3.13")
    assert env["tensorflow_version"]
    assert env["device"] in {"cpu", "gpu"}
    assert (env["device"] == "cpu") == (env["gpu_type"] == "cpu")
    assert env["compute_platform"] in {"colab", "local"}


def test_timestamps_match_the_schema_format() -> None:
    from datetime import timedelta, timezone

    pacific = timezone(timedelta(hours=-7))
    assert utc_timestamp(datetime(2026, 9, 26, 8, 30, 15, 999, tzinfo=pacific)) == (
        "2026-09-26T15:30:15Z"
    )
    assert utc_timestamp().endswith("Z")


def test_manifest_refuses_data_that_failed_verification(repo: Path) -> None:
    failed = ChecksumReport(checked={"DRIVE/a": "a" * 64}, mismatched=["DRIVE/a"])
    with pytest.raises(ValueError, match="failed verification"):
        build_manifest(
            run_id="r1",
            variant="baseline",
            config={},
            seed=0,
            deterministic_ops=False,
            repo=repo,
            data=failed,
            env=FAKE_ENV,
        )
