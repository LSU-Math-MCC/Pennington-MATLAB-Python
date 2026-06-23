"""Estimate a canonical human-centered (A-pose) coordinate frame and apply it.

Canonical frame:
  origin: pelvis midpoint
  +Y:     up along spine (shoulder_mid - pelvis)
  +X:     hip axis (right_hip - left_hip)  -> subject left-to-right
  +Z:     forward = cross(X, Y)
  scale:  1 / torso_length
"""
from __future__ import annotations

import numpy as np

from . import camera as camlib
from .transforms import make_T, apply_T, right_handed_frame, quat_to_mat, mat_to_quat
from ..types import (Camera, Pose2DResult, SplatCloud, CanonicalTransform,
                     SelectedDistanceSamples)


def lift_joints_3d(
    pose: Pose2DResult,
    depth: np.ndarray,
    mask: np.ndarray,
    cam: Camera,
    radius: int = 3,
    kp_conf_thresh: float = 0.2,
) -> dict:
    """Lift 2D keypoints to 3D world points via masked median depth in a window.

    Returns name -> (x, y, z, confidence).
    """
    H, W = depth.shape
    mask = np.asarray(mask, bool)
    out = {}
    for name, (x, y, c) in pose.keypoints.items():
        if c < kp_conf_thresh:
            continue
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        win_d = depth[y0:y1, x0:x1]
        win_m = mask[y0:y1, x0:x1]
        vals = win_d[win_m & np.isfinite(win_d) & (win_d > 0)]
        if vals.size == 0:
            vals = win_d[np.isfinite(win_d) & (win_d > 0)]
        if vals.size == 0:
            continue
        d = float(np.median(vals))
        var = float(np.var(vals)) if vals.size > 1 else 0.0
        depth_conf = float(np.exp(-var / (0.05 * d + 1e-6) ** 2)) if d > 0 else 0.0
        pw = camlib.backproject_to_world(cam, np.array([[x, y]]), np.array([d]))[0]
        out[name] = (float(pw[0]), float(pw[1]), float(pw[2]), float(c) * (0.5 + 0.5 * depth_conf))
    return out


def _kp(joints, name):
    return np.array(joints[name][:3], dtype=np.float64) if name in joints else None


def estimate_canonical_frame(joints_3d: dict) -> CanonicalTransform:
    """Build world_to_canonical from lifted 3D joints with graceful fallbacks.

    Anchor tiers: full torso (hips+shoulders) > shoulders+head > hips only.
    """
    lh, rh = _kp(joints_3d, "left_hip"), _kp(joints_3d, "right_hip")
    ls, rs = _kp(joints_3d, "left_shoulder"), _kp(joints_3d, "right_shoulder")
    nose = _kp(joints_3d, "nose")

    anchor = "silhouette"
    conf = 0.2
    pelvis = None
    x_axis = None
    y_axis = None
    torso_len = 1.0

    if lh is not None and rh is not None and ls is not None and rs is not None:
        pelvis = 0.5 * (lh + rh)
        shoulder_mid = 0.5 * (ls + rs)
        x_axis = rh - lh
        y_axis = shoulder_mid - pelvis
        torso_len = np.linalg.norm(shoulder_mid - pelvis)
        anchor = "torso"
        conf = 0.9
    elif ls is not None and rs is not None:
        shoulder_mid = 0.5 * (ls + rs)
        pelvis = shoulder_mid.copy()
        x_axis = rs - ls
        up = (nose - shoulder_mid) if nose is not None else np.array([0, -1.0, 0])
        # image y grows downward; "up" toward head is -y typically -> let frame fix sign
        y_axis = -up if np.dot(up, np.array([0, -1, 0])) < 0 else up
        y_axis = (shoulder_mid - nose) if nose is not None else np.array([0, 1.0, 0])
        torso_len = np.linalg.norm(rs - ls) + 1e-6
        anchor = "shoulders"
        conf = 0.55
    elif lh is not None and rh is not None:
        pelvis = 0.5 * (lh + rh)
        x_axis = rh - lh
        y_axis = np.array([0, 1.0, 0])
        torso_len = np.linalg.norm(rh - lh) + 1e-6
        anchor = "hips"
        conf = 0.4

    if pelvis is None:
        # last resort: identity-ish frame centered on mean joint
        pts = np.array([v[:3] for v in joints_3d.values()]) if joints_3d else np.zeros((1, 3))
        pelvis = pts.mean(axis=0)
        R = np.eye(3)
        T = make_T(R.T, -R.T @ pelvis, scale=1.0)
        return CanonicalTransform(world_to_canonical=T, scale=1.0, confidence=0.1,
                                  anchor_used="fallback")

    Rcols = right_handed_frame(x_axis, y_axis)  # columns = canonical axes in world
    R = Rcols.T                                  # world -> canonical rotation (rows = axes)
    scale = 1.0 / max(torso_len, 1e-6)
    # world_to_canonical: p_can = scale * R @ (p_world - pelvis)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * R @ pelvis
    jt = {}
    for name, v in joints_3d.items():
        pc = apply_T(T, np.array(v[:3]))[0] if False else (scale * R @ (np.array(v[:3]) - pelvis))
        Tj = np.eye(4)
        Tj[:3, 3] = pc
        jt[name] = Tj
    return CanonicalTransform(world_to_canonical=T, joint_transforms=jt,
                              scale=float(scale), confidence=float(conf), anchor_used=anchor)


def frame_from_points(points_world: np.ndarray) -> CanonicalTransform:
    """Silhouette/PCA fallback: build a frame from principal axes of a point cloud."""
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.shape[0] < 3:
        T = np.eye(4)
        return CanonicalTransform(world_to_canonical=T, scale=1.0, confidence=0.1,
                                  anchor_used="silhouette")
    mu = pts.mean(axis=0)
    cov = np.cov((pts - mu).T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(-w)
    v = v[:, order]
    # major axis -> Y (height), next -> X, normal -> Z
    y_axis = v[:, 0]
    x_axis = v[:, 1]
    Rcols = right_handed_frame(x_axis, y_axis)
    R = Rcols.T
    extent = np.sqrt(max(w[order[0]], 1e-9))
    scale = 1.0 / max(extent, 1e-6)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * R @ mu
    return CanonicalTransform(world_to_canonical=T, scale=float(scale),
                              confidence=0.25, anchor_used="silhouette")


def canonicalize_splats(splats: SplatCloud, ct: CanonicalTransform) -> SplatCloud:
    """Apply world_to_canonical to splat centers, rotations, and scales."""
    T = ct.world_to_canonical
    R = T[:3, :3]
    # decompose scale from R (uniform)
    s = float(np.cbrt(max(np.linalg.det(R), 1e-12)))
    Rrot = R / s if s > 0 else R
    centers = apply_T(T, splats.centers)
    # rotate quaternions
    quats = splats.rotations
    new_q = np.zeros_like(quats)
    for i in range(quats.shape[0]):
        Rs = quat_to_mat(quats[i])
        new_q[i] = mat_to_quat(Rrot @ Rs)
    new_scales = splats.scales * s
    return SplatCloud(
        centers=centers, scales=new_scales, rotations=new_q,
        opacities=splats.opacities.copy(), colors=splats.colors.copy(),
        extras=dict(splats.extras),
    )


CANONICAL_APOSE = {
    "pelvis": (0, 0, 0), "neck": (0, 1, 0), "head": (0, 1.35, 0),
    "left_shoulder": (-0.35, 1.0, 0), "right_shoulder": (0.35, 1.0, 0),
    "left_elbow": (-0.65, 0.65, 0), "right_elbow": (0.65, 0.65, 0),
    "left_wrist": (-0.9, 0.35, 0), "right_wrist": (0.9, 0.35, 0),
    "left_hip": (-0.18, 0, 0), "right_hip": (0.18, 0, 0),
    "left_knee": (-0.18, -0.7, 0), "right_knee": (0.18, -0.7, 0),
    "left_ankle": (-0.18, -1.35, 0), "right_ankle": (0.18, -1.35, 0),
}
