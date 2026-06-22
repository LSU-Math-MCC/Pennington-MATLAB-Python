"""Typed dataclasses for the 3DGS person -> canonical A-Frame pipeline.

These types are deliberately backend-agnostic. All geometry/fusion code consumes
and produces these dataclasses so that real model backends can be swapped without
touching the core pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Camera:
    K: np.ndarray              # 3x3 intrinsics
    R: np.ndarray              # 3x3 world-to-camera rotation
    t: np.ndarray              # 3 camera translation (world point -> cam: R @ X + t)
    width: int
    height: int


@dataclass
class SplatCloud:
    centers: np.ndarray        # Nx3
    scales: np.ndarray         # Nx3
    rotations: np.ndarray      # Nx4 quaternion (w, x, y, z)
    opacities: np.ndarray      # N
    colors: np.ndarray         # Nx3 (0..1 RGB)
    extras: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.centers.shape[0])


@dataclass
class SegmentationResult:
    person_mask: np.ndarray    # HxW bool
    confidence: Optional[np.ndarray] = None  # HxW float or None
    bbox: Optional[tuple] = None             # x0, y0, x1, y1


@dataclass
class Pose2DResult:
    keypoints: dict            # name -> (x, y, confidence)
    skeleton_edges: list = field(default_factory=list)


@dataclass
class Face2DResult:
    face_mask: Optional[np.ndarray]   # HxW bool, face-only when available
    landmarks: dict                   # name -> (x, y, confidence)
    bbox: Optional[tuple]             # x0, y0, x1, y1
    confidence: float = 0.0


@dataclass
class Face3DResult:
    landmarks_3d: dict                     # name -> (x, y, z, confidence)
    face_points_world: np.ndarray          # Fx3 selected face depth samples
    face_splat_indices: np.ndarray         # selected face splats
    canonical_face_transform: np.ndarray   # 4x4
    confidence: float = 0.0


@dataclass
class VisibilityResult:
    visible_regions: dict             # region -> confidence
    occlusion_flags: dict             # region -> bool or reason
    crop_flags: dict                  # region -> bool
    quality_score: float = 0.0


@dataclass
class DepthResult:
    depth: np.ndarray                 # HxW float, camera-space z proxy
    confidence: Optional[np.ndarray] = None  # HxW float or None


@dataclass
class SelectedDistanceSamples:
    pixels: np.ndarray         # Mx2, u/v
    depths: np.ndarray         # M
    points_cam: np.ndarray     # Mx3
    points_world: np.ndarray   # Mx3
    confidence: np.ndarray     # M

    def __len__(self) -> int:
        return int(self.pixels.shape[0])


@dataclass
class CanonicalTransform:
    world_to_canonical: np.ndarray    # 4x4
    joint_transforms: dict = field(default_factory=dict)  # bone/joint -> 4x4
    scale: float = 1.0
    confidence: float = 0.0
    anchor_used: str = "unknown"      # which anchor tier produced this frame


@dataclass
class SubjectInstance:
    instance_id: str
    image_path: str
    mask: np.ndarray                  # HxW bool for this person instance
    bbox: tuple                       # x0, y0, x1, y1
    pose_2d: Optional[Pose2DResult] = None
    face_2d: Optional[Face2DResult] = None
    visibility: Optional[VisibilityResult] = None
    selected_samples: Optional[SelectedDistanceSamples] = None
    splat_indices: Optional[np.ndarray] = None
    splat_scores: Optional[np.ndarray] = None
    canonical_transform: Optional[CanonicalTransform] = None
    association_confidence: float = 0.0


@dataclass
class SubjectTrack:
    subject_id: str
    instances: list = field(default_factory=list)          # list[SubjectInstance]
    canonical_splats: list = field(default_factory=list)   # list[SplatCloud]
    fused_splats: Optional[SplatCloud] = None
    confidence: float = 0.0
