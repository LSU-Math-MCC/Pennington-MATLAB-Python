import numpy as np
from pipeline.geometry import camera as camlib
from pipeline.geometry.canonicalize import (estimate_canonical_frame, frame_from_points,
                                            lift_joints_3d)
from pipeline.geometry.visibility import analyze_visibility
from pipeline.types import Pose2DResult, Face2DResult


def _pose(names_xy, conf=0.9):
    return Pose2DResult(keypoints={n: (x, y, conf) for n, (x, y) in names_xy.items()},
                        skeleton_edges=[])


def test_head_only_uses_face_or_shoulder_frame():
    # only head/shoulders visible
    joints = {"left_shoulder": (-0.3, 1.0, 0, 0.9), "right_shoulder": (0.3, 1.0, 0, 0.9),
              "nose": (0.0, 1.3, 0, 0.9)}
    ct = estimate_canonical_frame(joints)
    assert ct.anchor_used in ("shoulders", "torso")
    assert ct.confidence > 0


def test_body_only_without_face():
    joints = {"left_hip": (-0.2, 0, 0, 0.9), "right_hip": (0.2, 0, 0, 0.9),
              "left_shoulder": (-0.3, 1.0, 0, 0.9), "right_shoulder": (0.3, 1.0, 0, 0.9)}
    ct = estimate_canonical_frame(joints)
    assert ct.anchor_used == "torso"
    # pelvis at origin
    pelvis = 0.5 * (np.array(joints["left_hip"][:3]) + np.array(joints["right_hip"][:3]))
    from pipeline.geometry.transforms import apply_T
    assert np.allclose(apply_T(ct.world_to_canonical, pelvis)[0], [0, 0, 0], atol=1e-6)


def test_missing_regions_tagged_not_failed():
    pose = _pose({"nose": (5, 5), "left_eye": (4, 4), "right_eye": (6, 4)})
    vis = analyze_visibility(np.ones((20, 20), bool), (0, 0, 19, 19), pose, None, (20, 20))
    assert "left_leg" in vis.occlusion_flags
    assert vis.occlusion_flags["left_leg"] is True       # not visible, flagged not crashed
    assert vis.quality_score >= 0


def test_silhouette_fallback_frame():
    pts = np.random.default_rng(0).normal(size=(200, 3)) * np.array([0.2, 1.0, 0.1])
    ct = frame_from_points(pts)
    assert ct.anchor_used == "silhouette"
    R = ct.world_to_canonical[:3, :3] / ct.scale
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_low_conf_cannot_overwrite_high_conf_fusion():
    # fusion confidence gating: low-confidence splats dropped under threshold
    from pipeline.geometry.fusion import fuse_clouds
    from pipeline.types import SplatCloud
    good = SplatCloud(np.zeros((1, 3)), np.full((1, 3), .01), np.tile([1., 0, 0, 0], (1, 1)),
                      np.ones(1), np.full((1, 3), .8), extras={"confidence": np.array([0.9]),
                                                               "region": np.array([0])})
    bad = SplatCloud(np.zeros((1, 3)) + 0.001, np.full((1, 3), .01),
                     np.tile([1., 0, 0, 0], (1, 1)), np.ones(1), np.full((1, 3), .1),
                     extras={"confidence": np.array([0.05]), "region": np.array([0])})
    fused, rep = fuse_clouds([good, bad], conf_thresh=0.2)
    assert rep["output"] == 1     # the low-confidence splat is dropped
