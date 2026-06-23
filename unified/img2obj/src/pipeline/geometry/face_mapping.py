"""Face/head fixed-frame logic: face region gating, 3D anchor lifting, face frame.

The face provides a stable fixed frame for cross-view fusion when visible. We gate
depth with (person_mask & face_region), lift sparse/dense landmarks to 3D, and build
a face-local coordinate frame. Face thresholds are stricter than body thresholds.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import camera as camlib
from .projection import project_centers, sample_map
from .transforms import right_handed_frame
from ..types import (Camera, Face2DResult, SplatCloud, SelectedDistanceSamples,
                     Face3DResult)


def face_region_from_landmarks(landmarks: dict, shape, margin_px: int = 6) -> np.ndarray:
    """Build a boolean face region as the dilated convex hull of 2D landmarks."""
    H, W = shape[:2]
    region = np.zeros((H, W), bool)
    pts = np.array([[v[0], v[1]] for v in landmarks.values()
                    if v[2] > 0 and np.isfinite(v[0])], dtype=np.float64)
    if pts.shape[0] < 3:
        return region
    filled = False
    try:
        import cv2
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        poly = pts[hull.vertices].astype(np.int32)
        canvas = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(canvas, poly, 1)
        region = canvas.astype(bool)
        filled = True
    except Exception:
        filled = False
    if not filled:
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        region[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True

    if margin_px > 0:
        from scipy.ndimage import binary_dilation
        region = binary_dilation(region, iterations=int(margin_px))
    return region


def build_face_region(face: Face2DResult, person_mask: np.ndarray, margin_px: int = 6):
    """face_mask if available else landmark hull, intersected with person_mask."""
    shape = person_mask.shape
    if face.face_mask is not None:
        region = np.asarray(face.face_mask, bool)
    else:
        region = face_region_from_landmarks(face.landmarks, shape, margin_px)
    return region & np.asarray(person_mask, bool)


def select_face_depth(face_region, depth, cam: Camera, confidence=None,
                      depth_min=1e-3, depth_max=1e6) -> SelectedDistanceSamples:
    from .mask_depth_select import select_masked_depth
    return select_masked_depth(face_region, depth, cam, confidence,
                               depth_min=depth_min, depth_max=depth_max)


def lift_face_anchors(landmarks: dict, face_region, depth, cam: Camera, radius: int = 2) -> dict:
    """Lift named face landmarks to 3D using median masked depth windows."""
    H, W = depth.shape
    fr = np.asarray(face_region, bool)
    out = {}
    for name, (x, y, c) in landmarks.items():
        if c <= 0 or not np.isfinite(x):
            continue
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        wd = depth[y0:y1, x0:x1]
        wm = fr[y0:y1, x0:x1]
        vals = wd[wm & np.isfinite(wd) & (wd > 0)]
        if vals.size == 0:
            vals = wd[np.isfinite(wd) & (wd > 0)]
        if vals.size == 0:
            continue
        d = float(np.median(vals))
        std = float(np.std(vals)) if vals.size > 1 else 0.0
        if d > 0 and std > 0.5 * d:            # reject only wildly inconsistent windows
            continue
        pw = camlib.backproject_to_world(cam, np.array([[x, y]]), np.array([d]))[0]
        out[name] = (float(pw[0]), float(pw[1]), float(pw[2]), float(c))
    return out


def _avg(anchors, names):
    pts = [np.array(anchors[n][:3]) for n in names if n in anchors]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def derive_face_centers(anchors: dict) -> dict:
    """Compute eye/mouth/face centers from available landmarks."""
    d = {}
    le = _avg(anchors, ["left_eye_outer", "left_eye_inner", "left_eye", "left_eye_center"])
    re = _avg(anchors, ["right_eye_outer", "right_eye_inner", "right_eye", "right_eye_center"])
    if le is not None:
        d["left_eye_center"] = le
    if re is not None:
        d["right_eye_center"] = re
    mouth = _avg(anchors, ["mouth_left", "mouth_right", "upper_lip", "lower_lip"])
    if mouth is not None:
        d["mouth_center"] = mouth
    nose = _avg(anchors, ["nose_tip", "nose", "nose_bridge"])
    if nose is not None:
        d["nose_tip"] = nose
    chin = _avg(anchors, ["chin"])
    if chin is not None:
        d["chin"] = chin
    if le is not None and re is not None:
        d["eye_mid"] = 0.5 * (le + re)
    comps = [v for k, v in d.items() if k in ("eye_mid", "nose_tip", "mouth_center", "chin")]
    if comps:
        d["face_center"] = np.mean(comps, axis=0)
    return d


def face_canonical_frame(anchors: dict):
    """Build face-local coordinate frame. Returns (4x4 world_to_face, scale, confidence)."""
    centers = derive_face_centers(anchors)
    le = centers.get("left_eye_center")
    re = centers.get("right_eye_center")
    eye_mid = centers.get("eye_mid")
    mouth = centers.get("mouth_center")
    chin = centers.get("chin")
    nose = centers.get("nose_tip")
    origin = nose if nose is not None else centers.get("face_center")
    if le is None or re is None or origin is None or (mouth is None and chin is None):
        return None, 1.0, 0.0

    x_face = re - le                                  # left->right eye
    lower = mouth if mouth is not None else chin
    y_face = eye_mid - lower                           # up
    Rcols = right_handed_frame(x_face, y_face)
    R = Rcols.T
    interocular = np.linalg.norm(re - le)
    scale = 1.0 / max(interocular, 1e-6)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * R @ origin
    conf = float(min(1.0, 0.5 + 0.5 * (interocular > 1e-3)))
    return T, float(scale), conf


def assign_face_splats(splats: SplatCloud, cam: Camera, face_region, depth,
                       face_samples: SelectedDistanceSamples, person_indices,
                       tau_face_depth=0.03, tau_face_3d=0.05,
                       median_face_depth=None):
    """A splat is a face splat if it is a person splat AND projects into face_region
    AND its depth agrees with face-selected depth (or nearest face sample close)."""
    centers = splats.centers
    uv, z_cam, valid = project_centers(cam, centers)
    in_face, _ = sample_map(np.asarray(face_region, bool), uv)
    d_map, _ = sample_map(depth.astype(np.float64), uv, default=np.nan)
    if median_face_depth is None and len(face_samples) > 0:
        median_face_depth = float(np.median(face_samples.depths))
    tau_d = max((tau_face_depth * median_face_depth) if median_face_depth else tau_face_depth, 1e-4)

    proj = np.zeros(centers.shape[0])
    good = valid & in_face & np.isfinite(d_map) & (d_map > 0)
    if np.any(good):
        proj[good] = np.exp(-(np.abs(z_cam[good] - d_map[good]) / tau_d) ** 2)
    near = np.zeros(centers.shape[0])
    if len(face_samples) > 0:
        tree = cKDTree(face_samples.points_world)
        dist, _ = tree.query(centers, k=1)
        near = np.exp(-(dist / max(tau_face_3d, 1e-6)) ** 2)
    score = np.clip(0.6 * proj + 0.4 * near, 0, 1)
    person_set = set(int(i) for i in np.asarray(person_indices).tolist())
    mask_person = np.array([i in person_set for i in range(centers.shape[0])])
    score = score * mask_person
    indices = np.nonzero(score >= 0.35)[0]
    return indices, score
