"""Repose a posed human body into the canonical A-pose.

This is the piece the project was missing: turning a subject in ANY pose into a
normalized A-pose figure (not just a re-centered coordinate frame).

It is written model-agnostically against a minimal skeleton interface so it can be
unit-tested without the (license-gated) SMPL model:

    verts            : (V, 3) rest/posed vertices
    joints           : (J, 3) joint locations
    parents          : (J,)   kinematic parent index (-1 for root)
    skin_weights     : (V, J) linear-blend-skin weights (rows sum to 1)
    pose_src/pose_dst: (J, 3) per-joint axis-angle rotations (local)

We compute global joint transforms for source and target poses via forward
kinematics, then move each vertex by the per-joint LBS-weighted relative transform
T_dst @ inv(T_src). Setting pose_dst to the A-pose angles yields the A-pose mesh.
"""
from __future__ import annotations

import numpy as np


def axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    """(...,3) axis-angle -> (...,3,3) rotation (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    axis = np.where(small, 0.0, aa / np.where(small, 1.0, theta))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(theta)[..., 0]
    s = np.sin(theta)[..., 0]
    C = 1 - c
    R = np.empty(aa.shape[:-1] + (3, 3))
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


def forward_kinematics(joints: np.ndarray, parents, pose_aa: np.ndarray):
    """Return (J,4,4) global transforms for a pose given rest joints + parents.

    Each joint's local transform rotates about its rest position by pose_aa[j].
    """
    J = joints.shape[0]
    R = axis_angle_to_mat(pose_aa)                  # (J,3,3)
    G = np.zeros((J, 4, 4))
    for j in range(J):
        Tl = np.eye(4)
        Tl[:3, :3] = R[j]
        if parents[j] < 0:
            Tl[:3, 3] = joints[j]
            G[j] = Tl
        else:
            # offset from parent in rest pose
            Tl[:3, 3] = joints[j] - joints[parents[j]]
            G[j] = G[parents[j]] @ Tl
    # remove the rest-pose offset so transforms act about rest joints
    G_rel = np.zeros_like(G)
    for j in range(J):
        rest = np.eye(4)
        rest[:3, 3] = joints[j]
        G_rel[j] = G[j] @ np.linalg.inv(rest)
    return G_rel


def repose(verts, joints, parents, skin_weights, pose_src, pose_dst):
    """Move verts from pose_src to pose_dst via LBS. Returns (V,3)."""
    verts = np.asarray(verts, dtype=np.float64)
    G_src = forward_kinematics(joints, parents, pose_src)
    G_dst = forward_kinematics(joints, parents, pose_dst)
    # relative transform per joint: dst then undo src
    Trel = np.einsum("jab,jbc->jac", G_dst, np.linalg.inv(G_src))  # (J,4,4)
    Vh = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)  # (V,4)
    W = np.asarray(skin_weights, dtype=np.float64)                      # (V,J)
    # blended transform per vertex
    Tv = np.einsum("vj,jab->vab", W, Trel)                             # (V,4,4)
    out = np.einsum("vab,vb->va", Tv, Vh)[:, :3]
    return out


# ---- SMPL A-pose target (24-joint axis-angle, 72 dims) -----------------------
# Zero pose is SMPL's T-pose (arms straight out). A-pose lowers the arms ~50 deg
# by rotating the shoulder joints about the forward (z) axis.
SMPL_NUM_JOINTS = 24
SMPL_L_SHOULDER = 16
SMPL_R_SHOULDER = 17


def smpl_apose_thetas(arm_drop_deg: float = 50.0) -> np.ndarray:
    """Return (24,3) axis-angle pose for an SMPL A-pose (global_orient handled
    separately, kept zero here)."""
    thetas = np.zeros((SMPL_NUM_JOINTS, 3))
    a = np.deg2rad(arm_drop_deg)
    # rotate about z so arms swing down from the horizontal T-pose
    thetas[SMPL_L_SHOULDER] = [0, 0, -a]
    thetas[SMPL_R_SHOULDER] = [0, 0, a]
    return thetas
