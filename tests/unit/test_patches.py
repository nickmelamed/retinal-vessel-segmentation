from pathlib import Path
from typing import Any

import numpy as np
import pytest

from retinal_vessels.config import PatchesConfig, load_config
from retinal_vessels.patches import make_patch_dataset, sample_augmentation, sample_centers

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
N, H, W = 3, 40, 36
SIZE = 16
HALF = SIZE // 2


def config(**overrides: Any) -> PatchesConfig:
    base = load_config(CONFIGS / "smoke.yaml").patches
    return base.model_copy(update={"size": SIZE, "per_epoch": 50, "batch_size": 8, **overrides})


NO_AUG = {"flip": False, "rot90": False, "brightness_delta": 0.0, "contrast_range": (1.0, 1.0)}


def fovs() -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    disc = (yy - H / 2) ** 2 + (xx - W / 2) ** 2 <= 15**2
    out = np.stack([disc] * N)
    out[1, :, : W // 2] = False  # a half disc, so the images differ
    return out


def arrays(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = fovs()
    images = (rng.standard_normal((N, H, W)) * f).astype(np.float32)
    labels = (rng.random((N, H, W)) < 0.2) & f
    return images, labels, f


def batches(dataset: Any) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(x.numpy(), y.numpy(), w.numpy()) for x, y, w in dataset]


def test_centers_lie_inside_the_fov() -> None:
    f = fovs()
    centers = sample_centers(list(f), 2000, np.random.default_rng(0))
    assert centers.shape == (2000, 3)
    assert centers.dtype == np.int64
    assert f[centers[:, 0], centers[:, 1], centers[:, 2]].all()
    assert set(centers[:, 0]) == {0, 1, 2}


def test_centers_are_seeded() -> None:
    f = list(fovs())
    a = sample_centers(f, 100, np.random.default_rng(3))
    b = sample_centers(f, 100, np.random.default_rng(3))
    c = sample_centers(f, 100, np.random.default_rng(4))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    ("masks", "n", "message"),
    [
        ([], 5, "no FOV masks"),
        ([np.ones((4, 4), dtype=bool)], 0, "at least one center"),
        ([np.ones((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool)], 5, "FOV mask 1 is empty"),
    ],
)
def test_bad_center_requests_raise(masks: list[np.ndarray], n: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_centers(masks, n, np.random.default_rng(0))


def test_disabled_augmentation_is_the_identity() -> None:
    aug = sample_augmentation(config(**NO_AUG), 20, np.random.default_rng(0))
    assert not aug.flip_rows.any() and not aug.flip_cols.any()
    assert not aug.rotations.any()
    np.testing.assert_array_equal(aug.brightness, 0)
    np.testing.assert_array_equal(aug.contrast, 1)


def test_augmentation_stays_in_its_configured_ranges() -> None:
    cfg = config(brightness_delta=0.2, contrast_range=(0.8, 1.25))
    aug = sample_augmentation(cfg, 5000, np.random.default_rng(0))
    assert aug.flip_rows.any() and not aug.flip_rows.all()
    assert aug.flip_cols.any() and not aug.flip_cols.all()
    assert set(aug.rotations.tolist()) == {0, 1, 2, 3}
    assert np.abs(aug.brightness).max() <= 0.2
    assert aug.contrast.min() >= 0.8 and aug.contrast.max() <= 1.25


def test_patches_have_the_configured_shapes() -> None:
    out = batches(make_patch_dataset(*arrays(), config(), seed=0))
    assert [len(b[0]) for b in out] == [8] * 6 + [2]
    for x, y, w in out:
        for part in (x, y, w):
            assert part.shape[1:] == (SIZE, SIZE, 1)
            assert part.dtype == np.float32


def test_patch_centers_fall_inside_the_fov_and_match_the_source() -> None:
    images, labels, f = arrays()
    cfg = config(**NO_AUG)
    centers = sample_centers(list(f), cfg.per_epoch, np.random.default_rng(0))
    x, y, w = (
        np.concatenate(parts)
        for parts in zip(*batches(make_patch_dataset(images, labels, f, cfg, 0)), strict=True)
    )
    np.testing.assert_array_equal(w[:, HALF, HALF, 0], 1)
    i, r, c = centers.T
    np.testing.assert_array_equal(x[:, HALF, HALF, 0], images[i, r, c])
    np.testing.assert_array_equal(y[:, HALF, HALF, 0], labels[i, r, c])


def test_patches_past_the_edge_are_zero_padded() -> None:
    # A 2x2 FOV in the top-left corner, so every patch reaches past two edges.
    # Every patch contains all four FOV pixels, and reflected or wrapped
    # padding would add copies of them.
    images = np.ones((1, H, W), dtype=np.float32)
    f = np.zeros((1, H, W), dtype=bool)
    f[0, :2, :2] = True
    for x, y, w in batches(make_patch_dataset(images, f.copy(), f, config(**NO_AUG), 0)):
        np.testing.assert_array_equal(w.sum(axis=(1, 2, 3)), 4)
        np.testing.assert_array_equal(x, w)
        np.testing.assert_array_equal(y, w)


def test_same_seed_gives_identical_patches() -> None:
    a = batches(make_patch_dataset(*arrays(), config(), seed=5))
    b = batches(make_patch_dataset(*arrays(), config(), seed=5))
    for batch_a, batch_b in zip(a, b, strict=True):
        for part_a, part_b in zip(batch_a, batch_b, strict=True):
            np.testing.assert_array_equal(part_a, part_b)


def test_different_seeds_give_different_patches() -> None:
    a = batches(make_patch_dataset(*arrays(), config(), seed=5))[0][0]
    b = batches(make_patch_dataset(*arrays(), config(), seed=6))[0][0]
    assert not np.array_equal(a, b)


def test_geometric_augmentation_keeps_image_and_label_aligned() -> None:
    _, labels, f = arrays()
    images = labels.astype(np.float32)  # the image is its own label
    cfg = config(flip=True, rot90=True, brightness_delta=0.0, contrast_range=(1.0, 1.0))
    for x, y, w in batches(make_patch_dataset(images, labels, f, cfg, seed=1)):
        np.testing.assert_array_equal(x, y * w)


def test_geometric_augmentation_changes_some_patches() -> None:
    images, labels, f = arrays()
    plain = batches(make_patch_dataset(images, labels, f, config(**NO_AUG), seed=2))
    moved = batches(make_patch_dataset(images, labels, f, config(**{**NO_AUG, "rot90": True}), 2))
    assert any(not np.array_equal(p[2], m[2]) for p, m in zip(plain, moved, strict=True))


def test_jitter_touches_only_the_image_inside_the_fov() -> None:
    images, labels, f = arrays()
    cfg = config(flip=False, rot90=False, brightness_delta=0.5, contrast_range=(0.5, 2.0))
    plain = batches(make_patch_dataset(images, labels, f, config(**NO_AUG), seed=3))
    jittered = batches(make_patch_dataset(images, labels, f, cfg, seed=3))
    for (px, py, pw), (jx, jy, jw) in zip(plain, jittered, strict=True):
        np.testing.assert_array_equal(py, jy)
        np.testing.assert_array_equal(pw, jw)
        assert not jx[jw == 0].any()
        assert not np.allclose(px, jx)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda i, lab, f: (i[0], lab[0], f[0]), r"images must be \(N, H, W\)"),
        (lambda i, lab, f: (i.astype(np.float64), lab, f), "images must be float32"),
        (lambda i, lab, f: (i, lab[:, :-1], f), "labels shape"),
        (lambda i, lab, f: (i, lab, f.astype(np.uint8)), "fovs must be bool"),
        (lambda i, lab, f: (i[:0], lab[:0], f[:0]), "no images"),
    ],
)
def test_bad_inputs_raise(change: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_patch_dataset(*change(*arrays()), config(), seed=0)
