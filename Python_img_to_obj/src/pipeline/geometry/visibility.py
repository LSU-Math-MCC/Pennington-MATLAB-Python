"""Determine which body regions are visible / occluded / cropped for an instance.

Uses 2D pose keypoint confidences plus the instance mask/bbox position relative to
the image border. The output drives the anchor hierarchy and missing-region policy.
"""
from __future__ import annotations

import numpy as np

from ..types import Pose2DResult, Face2DResult, VisibilityResult


REGION_KEYPOINTS = {
    "face": ["nose", "left_eye", "right_eye"],
    "head": ["nose", "left_ear", "right_ear", "left_eye", "right_eye"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
    "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_leg": ["left_hip", "left_knee", "left_ankle"],
    "right_leg": ["right_hip", "right_knee", "right_ankle"],
}


def analyze_visibility(
    mask: np.ndarray,
    bbox,
    pose: Pose2DResult | None,
    face: Face2DResult | None,
    image_shape,
    kp_conf_thresh: float = 0.25,
) -> VisibilityResult:
    H, W = image_shape[:2]
    visible = {}
    occ = {}
    crop = {}

    kps = pose.keypoints if pose is not None else {}
    for region, names in REGION_KEYPOINTS.items():
        confs = [kps[n][2] for n in names if n in kps]
        if confs:
            score = float(np.mean([c for c in confs if c >= kp_conf_thresh] or [0.0]))
            present = sum(1 for c in confs if c >= kp_conf_thresh)
            visible[region] = score
            occ[region] = present < max(1, len(names) // 2)
        else:
            visible[region] = 0.0
            occ[region] = True

    if face is not None and face.confidence > 0:
        visible["face"] = max(visible.get("face", 0.0), float(face.confidence))
        occ["face"] = False

    # crop flags: bbox touching image border
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        crop["left"] = x0 <= 1
        crop["right"] = x1 >= W - 2
        crop["top"] = y0 <= 1
        crop["bottom"] = y1 >= H - 2

    mask_area = float(np.asarray(mask, bool).sum())
    quality = float(np.clip(mask_area / (H * W) * 4.0, 0.0, 1.0))
    quality = 0.5 * quality + 0.5 * float(np.mean(list(visible.values()) or [0.0]))
    return VisibilityResult(visible_regions=visible, occlusion_flags=occ,
                            crop_flags=crop, quality_score=quality)
