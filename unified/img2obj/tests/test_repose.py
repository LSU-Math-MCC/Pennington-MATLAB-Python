import numpy as np
from pipeline.geometry.repose import (axis_angle_to_mat, forward_kinematics, repose,
                                      smpl_apose_thetas, SMPL_L_SHOULDER, SMPL_R_SHOULDER)


def test_axis_angle_identity():
    R = axis_angle_to_mat(np.zeros(3))
    assert np.allclose(R, np.eye(3))


def test_axis_angle_90z():
    R = axis_angle_to_mat(np.array([0, 0, np.pi / 2]))
    v = R @ np.array([1.0, 0, 0])
    assert np.allclose(v, [0, 1, 0], atol=1e-9)


def _two_bone():
    # root at origin, child along +x; one vertex at the child tip
    joints = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    parents = np.array([-1, 0])
    verts = np.array([[2.0, 0, 0]])           # tip beyond child
    weights = np.array([[0.0, 1.0]])          # fully skinned to child joint
    return verts, joints, parents, weights


def test_repose_identity_when_poses_equal():
    verts, joints, parents, w = _two_bone()
    src = np.zeros((2, 3))
    out = repose(verts, joints, parents, w, src, src)
    assert np.allclose(out, verts, atol=1e-9)


def test_repose_rotates_child_bone():
    verts, joints, parents, w = _two_bone()
    src = np.zeros((2, 3))
    dst = np.zeros((2, 3))
    dst[1] = [0, 0, np.pi / 2]                # rotate child joint 90deg about z
    out = repose(verts, joints, parents, w, src, dst)
    # vertex at (2,0,0): child joint at (1,0,0); rotating about it 90deg z ->
    # (1,0,0) + Rz90 @ (1,0,0) = (1,0,0)+(0,1,0) = (1,1,0)
    assert np.allclose(out[0], [1, 1, 0], atol=1e-8)


def test_apose_thetas_shape_and_arms():
    t = smpl_apose_thetas(50.0)
    assert t.shape == (24, 3)
    assert not np.allclose(t[SMPL_L_SHOULDER], 0)
    assert not np.allclose(t[SMPL_R_SHOULDER], 0)
    # arms drop in opposite z directions (symmetric)
    assert np.isclose(t[SMPL_L_SHOULDER, 2], -t[SMPL_R_SHOULDER, 2])
