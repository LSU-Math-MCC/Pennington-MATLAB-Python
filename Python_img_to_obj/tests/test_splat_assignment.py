import numpy as np
from pipeline.geometry import camera as camlib
from pipeline.geometry.mask_depth_select import select_masked_depth
from pipeline.geometry.splat_assign import assign_splats_to_instance, resolve_ambiguous
from pipeline.types import SplatCloud


def _make(centers):
    n = centers.shape[0]
    return SplatCloud(centers=centers, scales=np.full((n, 3), 0.01),
                      rotations=np.tile([1.0, 0, 0, 0], (n, 1)),
                      opacities=np.ones(n), colors=np.full((n, 3), 0.5))


def _scene():
    H = W = 40
    mask = np.zeros((H, W), bool)
    mask[10:30, 10:30] = True
    depth = np.where(mask, 3.0, np.nan)
    cam = camlib.default_camera(W, H)
    samples = select_masked_depth(mask, depth, cam)
    return mask, depth, cam, samples


def test_inside_and_depth_consistent_selected():
    mask, depth, cam, samples = _scene()
    # splat that backprojects to center of mask at depth 3
    c_in = camlib.backproject_to_world(cam, np.array([[20.0, 20.0]]), np.array([3.0]))
    # splat outside mask
    c_out = camlib.backproject_to_world(cam, np.array([[2.0, 2.0]]), np.array([3.0]))
    # inside mask but wrong depth
    c_wrong = camlib.backproject_to_world(cam, np.array([[20.0, 20.0]]), np.array([6.0]))
    splats = _make(np.concatenate([c_in, c_out, c_wrong]))
    idx, scores = assign_splats_to_instance(splats, cam, mask, depth, samples,
                                            depth_tau=0.05, tau_3d=0.1,
                                            person_threshold=0.35)
    assert 0 in idx
    assert 1 not in idx
    assert 2 not in idx


def test_scores_in_range():
    mask, depth, cam, samples = _scene()
    rng = np.random.default_rng(0)
    pts = camlib.backproject_to_world(
        cam, rng.uniform(5, 35, (50, 2)), rng.uniform(2.5, 3.5, 50))
    splats = _make(pts)
    idx, scores = assign_splats_to_instance(splats, cam, mask, depth, samples)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_resolve_ambiguous():
    # two instances, splat 0 clearly inst0, splat1 clearly inst1, splat2 tie
    s0 = np.array([0.9, 0.1, 0.5])
    s1 = np.array([0.1, 0.9, 0.5])
    assign, amb = resolve_ambiguous([s0, s1], margin=0.15)
    assert assign[0] == 0 and assign[1] == 1
    assert amb[2]
    assert not amb[0]
