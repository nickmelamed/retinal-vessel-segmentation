"""Record where a run came from: data checksums, git state, and environment.

Checksum manifests use the ``sha256sum`` format, one ``<hex>  <path>`` line
per file, with paths that start with the dataset's directory name (for
example ``DRIVE/training/images/21_training.tif``). Each dataset directory
can live anywhere on disk, so verification takes a mapping from dataset name
to its location.
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20
MANIFEST_NAME = "manifest.json"
COLAB_ENV_VAR = "COLAB_RELEASE_TAG"
# Matches strftime in sql/schema.sql, so timestamps compare correctly as text.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    """Parse a checksum manifest into ``{relative path: hex digest}``.

    Blank lines are skipped. Any other line that is not ``<64 hex>  <path>``
    raises ``ValueError`` naming the line, as does a path listed twice.
    """
    entries: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        valid_digest = len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        if not sep or not rel or not valid_digest:
            raise ValueError(f"{path}:{lineno}: expected '<sha256>  <path>', got {line!r}")
        if rel in entries:
            raise ValueError(f"{path}:{lineno}: {rel} is listed twice")
        entries[rel] = digest
    return entries


def format_checksums(entries: Mapping[str, str]) -> str:
    """Render entries as manifest text, sorted by path."""
    return "".join(f"{entries[rel]}  {rel}\n" for rel in sorted(entries))


def compute_checksums(locations: Mapping[str, Path]) -> dict[str, str]:
    """Hash every file under each dataset directory.

    Hidden files and directories (``.DS_Store``, for example) are skipped.
    Keys are ``<dataset name>/<path relative to its directory>``.
    """
    entries: dict[str, str] = {}
    for name, directory in locations.items():
        for path in sorted(directory.rglob("*")):
            rel_parts = path.relative_to(directory).parts
            if path.is_file() and not any(part.startswith(".") for part in rel_parts):
                entries["/".join((name, *rel_parts))] = sha256_file(path)
    return entries


def data_hash(entries: Mapping[str, str]) -> str:
    """Return one hash summarizing a set of checksum entries, independent of order."""
    return hashlib.sha256(format_checksums(entries).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChecksumReport:
    """Differences between a manifest and the files on disk."""

    checked: dict[str, str]
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.mismatched or self.unlisted)

    def problems(self) -> list[str]:
        """Return one human-readable line per problem."""
        return (
            [f"missing: {rel}" for rel in self.missing]
            + [f"checksum mismatch: {rel}" for rel in self.mismatched]
            + [f"not in manifest: {rel}" for rel in self.unlisted]
        )


def verify_checksums(expected: Mapping[str, str], locations: Mapping[str, Path]) -> ChecksumReport:
    """Compare the manifest entries for each dataset in ``locations`` with disk.

    Entries for datasets not in ``locations`` are ignored, so optional
    external datasets are checked only when their directory is given.
    """
    names = set(locations)
    if not names:
        raise ValueError("no dataset locations given to verify")
    wanted = {rel: d for rel, d in expected.items() if rel.split("/", 1)[0] in names}
    unknown = names - {rel.split("/", 1)[0] for rel in wanted}
    if unknown:
        raise ValueError(f"manifest has no entries for dataset(s): {sorted(unknown)}")
    present = {name: path for name, path in locations.items() if path.is_dir()}
    actual = compute_checksums(present)
    return ChecksumReport(
        checked=wanted,
        missing=sorted(set(wanted) - set(actual)),
        mismatched=sorted(r for r in set(wanted) & set(actual) if wanted[r] != actual[r]),
        unlisted=sorted(set(actual) - set(wanted)),
    )


def check_or_write_checksums(
    manifest_path: Path, locations: Mapping[str, Path]
) -> tuple[bool, ChecksumReport]:
    """Verify data against ``manifest_path``, or write it if it does not exist yet.

    An existing manifest is never modified. Returns whether the manifest was
    written and the verification report.
    """
    if manifest_path.exists():
        return False, verify_checksums(read_checksums(manifest_path), locations)
    missing_dirs = [str(p) for p in locations.values() if not p.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"cannot write checksums, directory not found: {missing_dirs}")
    entries = compute_checksums(locations)
    if not entries:
        raise ValueError(f"cannot write checksums, no files under {list(locations.values())}")
    manifest_path.write_text(format_checksums(entries), encoding="utf-8")
    logger.info("Wrote %d checksums to %s", len(entries), manifest_path)
    return True, ChecksumReport(checked=entries)


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: bool


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
    return result.stdout


def git_state(repo: Path) -> GitState:
    """Return the HEAD commit and whether the tree has uncommitted or untracked changes.

    Ignored files (data, results, checkpoints) do not make the tree dirty.
    """
    commit = _git(repo, "rev-parse", "HEAD").strip()
    dirty = bool(_git(repo, "status", "--porcelain").strip())
    return GitState(commit=commit, dirty=dirty)


def compute_platform(environ: Mapping[str, str]) -> str:
    """Return ``"colab"`` when running on a Colab runtime, otherwise ``"local"``."""
    return "colab" if COLAB_ENV_VAR in environ else "local"


def environment() -> dict[str, Any]:
    """Describe the interpreter, TensorFlow build, and device this process runs on.

    Imports TensorFlow, so it takes a few seconds on first call.
    """
    import tensorflow as tf

    build = tf.sysconfig.get_build_info()
    gpus = tf.config.list_physical_devices("GPU")
    gpu_type = "cpu"
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        gpu_type = str(details.get("device_name", "unknown gpu"))
    return {
        "python_version": platform.python_version(),
        "tensorflow_version": str(tf.__version__),
        "cuda_version": build.get("cuda_version"),
        "device": "gpu" if gpus else "cpu",
        "gpu_type": gpu_type,
        "compute_platform": compute_platform(os.environ),
    }


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a SHA-256 of the config that does not depend on key order."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def utc_timestamp(now: datetime | None = None) -> str:
    """Return ``now``, or the current time, as UTC text like ``2026-09-26T15:30:00Z``."""
    return (now or datetime.now(UTC)).astimezone(UTC).strftime(TIMESTAMP_FORMAT)


def new_run_id(now: datetime | None = None) -> str:
    """Return a sortable, unique run id such as ``20260926T153000Z-1a2b3c``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to say which code, data, and machine produced a run."""

    run_id: str
    variant: str
    config: dict[str, Any]
    config_hash: str
    seed: int
    git_commit: str
    git_dirty: bool
    data_hash: str
    deterministic_ops: bool
    started_at: str
    environment: dict[str, Any]
    finished_at: str | None = None


def build_manifest(
    *,
    run_id: str,
    variant: str,
    config: Mapping[str, Any],
    seed: int,
    deterministic_ops: bool,
    repo: Path,
    data: ChecksumReport,
    env: Mapping[str, Any] | None = None,
    started_at: datetime | None = None,
) -> RunManifest:
    """Collect the manifest for a run that is starting now.

    ``data`` is the verification report for the data the run uses, and a
    report with any problem raises ``ValueError``, so a run never records the
    hash of data that failed its checksums. ``env`` defaults to
    :func:`environment`, and tests pass their own to avoid importing TensorFlow.
    """
    if not data.ok:
        raise ValueError(f"data failed verification: {data.problems()[:3]}")
    state = git_state(repo)
    if state.dirty:
        logger.warning("Working tree is dirty. Run %s cannot be reported.", run_id)
    return RunManifest(
        run_id=run_id,
        variant=variant,
        config=dict(config),
        config_hash=config_hash(config),
        seed=seed,
        git_commit=state.commit,
        git_dirty=state.dirty,
        data_hash=data_hash(data.checked),
        deterministic_ops=deterministic_ops,
        started_at=utc_timestamp(started_at),
        environment=dict(env if env is not None else environment()),
    )


def write_manifest(manifest: RunManifest, results_dir: Path) -> Path:
    """Write ``results_dir/<run_id>/manifest.json`` and return its path."""
    run_dir = results_dir / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_NAME
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote run manifest %s", path)
    return path
