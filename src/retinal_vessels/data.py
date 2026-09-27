"""The sample type shared by every dataset adapter, and cross-validation folds.

Folds implement SPEC section 5. The split is by whole image, never by patch,
so no pixel of a held-out image can reach training.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

Dataset = Literal["drive", "stare", "chase"]
Split = Literal["training", "test", "external"]
Role = Literal["train", "val", "test"]

DATASETS: tuple[Dataset, ...] = ("drive", "stare", "chase")
SPLITS: tuple[Split, ...] = ("training", "test", "external")
RGB_CHANNELS = 3


@dataclass(frozen=True, eq=False)
class Sample:
    """One fundus image with its FOV mask and, where published, its vessel label.

    ``image`` is (H, W, 3) uint8 RGB. ``fov`` and ``label`` are (H, W) bool.
    ``label`` is None for images whose annotations are withheld. Construction
    validates everything and raises ``ValueError`` on the first problem.
    """

    dataset: Dataset
    image_id: str
    split: Split
    image: np.ndarray
    fov: np.ndarray
    label: np.ndarray | None
    patient_id: str | None = None
    abnormality_note: str | None = None

    def __post_init__(self) -> None:
        validate_sample(self)

    @property
    def has_abnormality(self) -> bool:
        """Return whether the official site lists an abnormality for this image."""
        return self.abnormality_note is not None


def validate_sample(sample: Sample) -> None:
    """Raise ``ValueError`` naming the image if any field is malformed."""
    where = f"{sample.dataset} image {sample.image_id!r}"
    if sample.dataset not in DATASETS:
        raise ValueError(f"{where}: unknown dataset, expected one of {DATASETS}")
    if sample.split not in SPLITS:
        raise ValueError(f"{where}: unknown split {sample.split!r}, expected one of {SPLITS}")
    if not sample.image_id:
        raise ValueError(f"{where}: image_id is empty")
    image = sample.image
    if image.ndim != 3 or image.shape[2] != RGB_CHANNELS:
        raise ValueError(f"{where}: image must be (H, W, 3), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{where}: image must be uint8, got {image.dtype}")
    _check_mask(where, "fov", sample.fov, image.shape[:2])
    if not sample.fov.any():
        raise ValueError(f"{where}: FOV mask is empty")
    if sample.label is not None:
        _check_mask(where, "label", sample.label, image.shape[:2])
    if sample.abnormality_note is not None and not sample.abnormality_note.strip():
        raise ValueError(f"{where}: abnormality_note is blank, use None for a normal image")


def _check_mask(where: str, name: str, mask: np.ndarray, shape: tuple[int, ...]) -> None:
    if mask.dtype != np.bool_:
        raise ValueError(f"{where}: {name} must be bool, got {mask.dtype}")
    if mask.shape != shape:
        raise ValueError(f"{where}: {name} shape {mask.shape} does not match image {shape}")


@dataclass(frozen=True)
class Fold:
    """The image ids in each role for one cross-validation fold, numbered from 1."""

    number: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


def make_folds(
    image_ids: Iterable[str],
    spread: Collection[str],
    n_folds: int,
    n_val: int,
    seed: int,
) -> list[Fold]:
    """Split whole images into ``n_folds`` folds, each holding every image once.

    Each image is a test image in exactly one fold. Within a fold, ``n_val`` of
    the other images are drawn for validation and the rest are for training.
    The ids in ``spread`` (DRIVE's abnormal images) go to different test folds,
    so no one fold's scores are dominated by pathology. The result depends only
    on the set of ids and the seed, not on their order. Invalid sizes raise
    ``ValueError``.
    """
    ids = sorted(image_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("image ids contain duplicates")
    if len(set(spread)) != len(spread):
        raise ValueError("spread ids contain duplicates")
    if not ids or n_folds < 2 or len(ids) % n_folds != 0:
        raise ValueError(f"cannot split {len(ids)} images into {n_folds} equal folds")
    per_fold = len(ids) // n_folds
    if not set(spread) <= set(ids):
        raise ValueError(f"spread ids not among the images: {sorted(set(spread) - set(ids))}")
    if len(spread) > n_folds:
        raise ValueError(f"cannot spread {len(spread)} images over {n_folds} folds")
    if n_val < 1 or per_fold + n_val >= len(ids):
        raise ValueError(f"n_val={n_val} leaves no training images with {len(ids)} images")

    rng = np.random.default_rng(seed)
    tests: list[list[str]] = [[] for _ in range(n_folds)]
    spread_ids = [str(i) for i in rng.permutation(sorted(spread))]
    for fold_index, image_id in zip(rng.permutation(n_folds), spread_ids, strict=False):
        tests[fold_index].append(image_id)
    others = [str(i) for i in rng.permutation(sorted(set(ids) - set(spread)))]
    for test in tests:
        need = per_fold - len(test)
        test.extend(others[:need])
        others = others[need:]

    folds = []
    for number, test in enumerate(tests, start=1):
        rest = sorted(set(ids) - set(test))
        val = {str(i) for i in rng.choice(rest, size=n_val, replace=False)}
        folds.append(
            Fold(
                number=number,
                train=tuple(i for i in rest if i not in val),
                val=tuple(sorted(val)),
                test=tuple(sorted(test)),
            )
        )
    return folds


def fold_rows(folds: Iterable[Fold]) -> list[tuple[int, str, Role]]:
    """Flatten folds into ``(fold, image_id, role)`` rows for ``fold_assignments``."""
    rows: list[tuple[int, str, Role]] = []
    for fold in folds:
        groups: tuple[tuple[Role, tuple[str, ...]], ...] = (
            ("train", fold.train),
            ("val", fold.val),
            ("test", fold.test),
        )
        for role, members in groups:
            rows.extend((fold.number, image_id, role) for image_id in members)
    return rows
