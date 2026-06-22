import sys
from pathlib import Path

import numpy as np
import trimesh

TOOLS = Path(__file__).resolve().parents[1] / "tools"
for _sub in ("benchmark", "anthro"):
    _p = str(TOOLS / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_metrics as BM  # noqa: E402
import lhm_anthropometry as A  # noqa: E402
import shapy_measure as SM  # noqa: E402


def _ellipse_cylinder(a=0.20, b=0.15, height=1.8, sections=160, xoff=0.0):
    mesh = trimesh.creation.cylinder(radius=1.0, height=height, sections=sections)
    v = np.asarray(mesh.vertices).copy()
    # trimesh cylinders are z-up; the anthropometry code is y-up.
    x = v[:, 0] * a + xoff
    y = v[:, 2]
    z = v[:, 1] * b
    mesh.vertices = np.column_stack([x, y, z])
    return mesh


def _ramanujan_ellipse_perimeter(a, b):
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


def test_shapy_plane_perimeter_matches_ellipse_section():
    mesh = _ellipse_cylinder(a=0.22, b=0.14)
    got = SM.plane_perimeter(np.asarray(mesh.vertices), mesh.faces, 0.0, torso_only=True)
    expected = _ramanujan_ellipse_perimeter(0.22, 0.14)
    assert np.isclose(got, expected, rtol=0.04)


def test_shapy_plane_perimeter_drops_separate_arm_loops():
    torso = _ellipse_cylinder(a=0.20, b=0.15)
    left_arm = _ellipse_cylinder(a=0.035, b=0.035, xoff=-0.42)
    right_arm = _ellipse_cylinder(a=0.035, b=0.035, xoff=0.42)
    mesh = trimesh.util.concatenate([torso, left_arm, right_arm])
    v = np.asarray(mesh.vertices)

    torso_only = SM.plane_perimeter(v, mesh.faces, 0.0, torso_only=True)
    all_loops = SM.plane_perimeter(v, mesh.faces, 0.0, torso_only=False)

    assert np.isclose(torso_only, _ramanujan_ellipse_perimeter(0.20, 0.15), rtol=0.05)
    assert all_loops > torso_only + 0.35


def test_bench_metrics_units_and_hbw_decision_rule():
    gt = np.zeros((4, 3), float)
    pred = np.array([[0.001, 0, 0], [0, 0.002, 0], [0, 0, 0.003], [0, 0, 0.004]])
    assert BM.p2p20k(pred, gt, sample_idx=np.arange(4)) == np.linalg.norm(pred, axis=1).mean() * 1000
    assert BM.v2v(pred, gt) == np.linalg.norm(pred, axis=1).mean() * 1000

    shapy = {"p2p20k_mm": 21, "height_mm": 51, "chest_mm": 65, "waist_mm": 69, "hips_mm": 57}
    good = {"p2p20k_mm": 20, "height_mm": 50, "chest_mm": 64, "waist_mm": 68, "hips_mm": 57}
    bad_shape = dict(good, p2p20k_mm=22)
    bad_regress = dict(good, hips_mm=80)

    assert BM.beats_hbw(good, shapy)
    assert not BM.beats_hbw(bad_shape, shapy)
    assert not BM.beats_hbw(bad_regress, shapy)


def test_lhm_fuse_betas_rejects_large_outlier_and_tracks_spread():
    per_view = [
        {"beta": np.array([0.0, 1.0]), "is_full_body": True},
        {"beta": np.array([0.1, 1.1]), "is_full_body": True},
        {"beta": np.array([50.0, -40.0]), "is_full_body": True},
    ]
    fused, spread, n = A.fuse_betas(per_view)

    assert n == 3
    assert np.allclose(fused, [0.05, 1.05])
    assert np.allclose(spread, [0.05, 0.05])


def test_lhm_fuse_betas_weights_partial_body_lower():
    per_view = [
        {"beta": np.array([0.0]), "is_full_body": True},
        {"beta": np.array([10.0]), "is_full_body": False},
    ]
    fused, _, _ = A.fuse_betas(per_view)

    assert np.allclose(fused, [10.0 * 0.4 / 1.4])
