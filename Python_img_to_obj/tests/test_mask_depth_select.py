import numpy as np
from pipeline.geometry import camera as camlib
from pipeline.geometry.mask_depth_select import select_masked_depth, save_samples, load_samples


def _setup():
    H = W = 10
    mask = np.zeros((H, W), bool)
    mask[3:7, 3:7] = True            # center 4x4
    depth = np.full((H, W), 2.0)
    cam = camlib.default_camera(W, H)
    return mask, depth, cam


def test_only_masked_selected():
    mask, depth, cam = _setup()
    s = select_masked_depth(mask, depth, cam)
    assert len(s) == 16
    # every selected pixel is inside the mask
    for u, v in s.pixels:
        assert mask[int(v), int(u)]


def test_invalid_depth_ignored():
    mask, depth, cam = _setup()
    depth[4, 4] = np.nan
    depth[5, 5] = -1.0
    s = select_masked_depth(mask, depth, cam)
    assert len(s) == 14


def test_count_equals_valid_mask_pixels():
    mask, depth, cam = _setup()
    s = select_masked_depth(mask, depth, cam)
    assert s.points_world.shape[0] == int(mask.sum())


def test_npz_roundtrip(tmp_path):
    mask, depth, cam = _setup()
    s = select_masked_depth(mask, depth, cam)
    p = tmp_path / "sel.npz"
    save_samples(p, s)
    s2 = load_samples(p)
    assert np.allclose(s2.points_world, s.points_world)
