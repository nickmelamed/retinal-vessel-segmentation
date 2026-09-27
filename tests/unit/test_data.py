from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from retinal_vessels.data import Fold, Sample, fold_rows, make_folds

DRIVE_TRAIN = [f"{i:02d}" for i in range(21, 41)]
ABNORMAL = {"25", "26", "32"}
SHAPE = (6, 5)


def sample(**overrides: Any) -> Sample:
    fov = np.zeros(SHAPE, dtype=bool)
    fov[1:5, 1:4] = True
    fields: dict[str, Any] = {
        "dataset": "drive",
        "image_id": "21",
        "split": "training",
        "image": np.zeros((*SHAPE, 3), dtype=np.uint8),
        "fov": fov,
        "label": fov.copy(),
    }
    return Sample(**{**fields, **overrides})


def drive_folds(seed: int = 0) -> list[Fold]:
    return make_folds(DRIVE_TRAIN, ABNORMAL, n_folds=5, n_val=2, seed=seed)


def test_valid_sample_builds() -> None:
    s = sample(abnormality_note="background diabetic retinopathy")
    assert s.has_abnormality
    assert not sample().has_abnormality
    assert sample(label=None, split="test").label is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dataset": "hrf"}, "unknown dataset"),
        ({"split": "val"}, "unknown split"),
        ({"image_id": ""}, "image_id is empty"),
        ({"image": np.zeros(SHAPE, dtype=np.uint8)}, r"must be \(H, W, 3\)"),
        ({"image": np.zeros((*SHAPE, 4), dtype=np.uint8)}, r"must be \(H, W, 3\)"),
        ({"image": np.zeros((*SHAPE, 3), dtype=np.float32)}, "must be uint8"),
        ({"fov": np.ones(SHAPE, dtype=np.uint8)}, "fov must be bool"),
        ({"fov": np.ones((5, 6), dtype=bool)}, "fov shape"),
        ({"fov": np.zeros(SHAPE, dtype=bool)}, "FOV mask is empty"),
        ({"label": np.ones(SHAPE, dtype=np.uint8)}, "label must be bool"),
        ({"label": np.ones((6, 4), dtype=bool)}, "label shape"),
        ({"abnormality_note": "  "}, "abnormality_note is blank"),
    ],
)
def test_invalid_sample_raises(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fov": np.zeros(SHAPE, dtype=bool)},
        {"fov": np.ones(SHAPE, dtype=np.uint8)},
        {"label": np.ones((6, 4), dtype=bool)},
    ],
)
def test_error_names_the_image(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="^drive image '33': "):
        sample(image_id="33", **overrides)


def test_replace_revalidates() -> None:
    with pytest.raises(ValueError, match="FOV mask is empty"):
        replace(sample(), fov=np.zeros(SHAPE, dtype=bool))


def test_drive_folds_have_the_section_5_sizes() -> None:
    folds = drive_folds()
    assert [f.number for f in folds] == [1, 2, 3, 4, 5]
    for fold in folds:
        assert (len(fold.train), len(fold.val), len(fold.test)) == (14, 2, 4)


def test_roles_within_a_fold_are_disjoint_and_complete() -> None:
    for fold in drive_folds():
        train, val, test = set(fold.train), set(fold.val), set(fold.test)
        assert not train & val and not train & test and not val & test
        assert train | val | test == set(DRIVE_TRAIN)


def test_every_image_is_test_exactly_once() -> None:
    counts = Counter(i for fold in drive_folds() for i in fold.test)
    assert set(counts) == set(DRIVE_TRAIN)
    assert set(counts.values()) == {1}


def test_abnormal_images_land_in_different_test_folds() -> None:
    for seed in range(50):
        holders = [f.number for f in drive_folds(seed) for i in f.test if i in ABNORMAL]
        assert len(holders) == 3
        assert len(set(holders)) == 3


def test_same_seed_gives_same_folds_regardless_of_input_order() -> None:
    reversed_ids = make_folds(DRIVE_TRAIN[::-1], ABNORMAL, n_folds=5, n_val=2, seed=7)
    assert drive_folds(7) == reversed_ids


def test_different_seeds_give_different_folds() -> None:
    assert drive_folds(0) != drive_folds(1)


def test_seed_changes_both_test_and_val_draws() -> None:
    tests = {tuple(f.test for f in drive_folds(s)) for s in range(10)}
    vals = {tuple(f.val for f in drive_folds(s)) for s in range(10)}
    assert len(tests) > 1
    assert len(vals) > 1


def test_members_are_sorted_strings() -> None:
    for fold in drive_folds():
        for members in (fold.train, fold.val, fold.test):
            assert list(members) == sorted(members)
            assert all(type(i) is str for i in members)


def test_fold_rows_flatten_every_assignment() -> None:
    folds = drive_folds()
    rows = fold_rows(folds)
    assert len(rows) == 5 * 20
    assert Counter(role for _, _, role in rows) == {"train": 70, "val": 10, "test": 20}
    first = folds[0]
    assert (1, first.test[0], "test") in rows
    assert (1, first.val[0], "val") in rows
    assert (1, first.train[0], "train") in rows


@pytest.mark.parametrize(
    ("ids", "spread", "n_folds", "n_val", "message"),
    [
        (["a", "a", "b", "c"], set(), 2, 1, "^image ids contain duplicates$"),
        ([], set(), 5, 2, "cannot split 0 images"),
        (DRIVE_TRAIN[:19], set(), 5, 2, "cannot split 19 images into 5"),
        (DRIVE_TRAIN, set(), 1, 2, "into 1 equal folds"),
        (DRIVE_TRAIN, {"99"}, 5, 2, "not among the images"),
        (DRIVE_TRAIN, set(DRIVE_TRAIN[:6]), 5, 2, "cannot spread 6 images over 5"),
        (DRIVE_TRAIN, set(), 5, 0, "n_val=0"),
        (DRIVE_TRAIN, set(), 5, 16, "n_val=16 leaves no training"),
    ],
)
def test_invalid_split_raises(
    ids: list[str], spread: set[str], n_folds: int, n_val: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_folds(ids, spread, n_folds=n_folds, n_val=n_val, seed=0)


def test_largest_valid_val_leaves_one_training_image() -> None:
    folds = make_folds(DRIVE_TRAIN, set(), n_folds=5, n_val=15, seed=0)
    assert all(len(f.train) == 1 for f in folds)


def test_two_folds_is_the_smallest_split() -> None:
    folds = make_folds(["a", "b", "c", "d"], {"a", "b"}, n_folds=2, n_val=1, seed=0)
    assert [len(f.test) for f in folds] == [2, 2]
    assert all(len({"a", "b"} & set(f.test)) == 1 for f in folds)


def test_spread_may_fill_every_fold() -> None:
    spread = set(DRIVE_TRAIN[:5])
    folds = make_folds(DRIVE_TRAIN, spread, n_folds=5, n_val=2, seed=3)
    assert all(len(set(f.test) & spread) == 1 for f in folds)


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n_folds=st.integers(min_value=3, max_value=6),
    per_fold=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
def test_fold_invariants_hold_for_any_valid_split(
    seed: int, n_folds: int, per_fold: int, data: st.DataObject
) -> None:
    ids = [f"img{i}" for i in range(n_folds * per_fold)]
    n_val = data.draw(st.integers(min_value=1, max_value=len(ids) - per_fold - 1))
    spread = data.draw(st.sets(st.sampled_from(ids), max_size=n_folds))
    folds = make_folds(ids, spread, n_folds=n_folds, n_val=n_val, seed=seed)

    assert len(folds) == n_folds
    assert Counter(i for f in folds for i in f.test) == dict.fromkeys(ids, 1)
    for fold in folds:
        assert len(fold.test) == per_fold
        assert len(fold.val) == n_val
        assert sorted(fold.train + fold.val + fold.test) == sorted(ids)
    holders = [f.number for f in folds for i in f.test if i in spread]
    assert len(holders) == len(set(holders)) == len(spread)
