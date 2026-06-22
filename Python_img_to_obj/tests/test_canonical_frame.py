import numpy as np
from pipeline.geometry.canonicalize import estimate_canonical_frame
from pipeline.geometry.transforms import apply_T


def _joints():
    return {
        "left_hip": (-1.0, 0, 0, 0.9), "right_hip": (1.0, 0, 0, 0.9),
        "left_shoulder": (-1.0, 2.0, 0, 0.9), "right_shoulder": (1.0, 2.0, 0, 0.9),
    }


def test_pelvis_maps_to_origin():
    ct = estimate_canonical_frame(_joints())
    pelvis = np.array([0.0, 0.0, 0.0])
    assert np.allclose(apply_T(ct.world_to_canonical, pelvis)[0], [0, 0, 0], atol=1e-6)


def test_spine_maps_to_plus_y():
    ct = estimate_canonical_frame(_joints())
    shoulder_mid = np.array([0.0, 2.0, 0.0])
    p = apply_T(ct.world_to_canonical, shoulder_mid)[0]
    # along +Y, x and z near zero
    assert p[1] > 0
    assert abs(p[0]) < 1e-6 and abs(p[2]) < 1e-6


def test_hip_axis_maps_to_plus_x():
    ct = estimate_canonical_frame(_joints())
    rh = np.array([1.0, 0.0, 0.0])
    p = apply_T(ct.world_to_canonical, rh)[0]
    assert p[0] > 0
    assert abs(p[1]) < 1e-6 and abs(p[2]) < 1e-6


def test_right_handed():
    ct = estimate_canonical_frame(_joints())
    R = ct.world_to_canonical[:3, :3] / ct.scale
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_scale_normalization_stable():
    ct = estimate_canonical_frame(_joints())
    # torso length 2 -> scale 0.5; shoulder_mid maps to y=1
    p = apply_T(ct.world_to_canonical, np.array([0.0, 2.0, 0.0]))[0]
    assert np.isclose(p[1], 1.0, atol=1e-6)
