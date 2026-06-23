"""Abstract backend interfaces. Implementations take a loaded RGB image (HxWx3 uint8)
plus an output directory, returning typed dataclasses from pipeline.types.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import (SplatCloud, SegmentationResult, Pose2DResult, Face2DResult,
                     DepthResult, Camera)


class GSReconstructionBackend:
    name = "base-gs"
    version = "0"

    def reconstruct(self, image: np.ndarray, out_dir: Path):
        raise NotImplementedError


class SegmentationBackend:
    name = "base-seg"
    version = "0"

    def segment_people(self, image: np.ndarray, out_dir: Path) -> list[SegmentationResult]:
        raise NotImplementedError


class PoseBackend:
    name = "base-pose"
    version = "0"

    def estimate_pose(self, image: np.ndarray, out_dir: Path, bbox=None) -> Pose2DResult:
        raise NotImplementedError


class FaceBackend:
    name = "base-face"
    version = "0"

    def estimate_face(self, image: np.ndarray, out_dir: Path, bbox=None) -> Face2DResult:
        raise NotImplementedError


class DepthBackend:
    name = "base-depth"
    version = "0"

    def estimate_or_render_depth(self, image: np.ndarray, splats: SplatCloud,
                                 camera: Camera | None, out_dir: Path) -> DepthResult:
        raise NotImplementedError


class SubjectAssociationBackend:
    name = "base-assoc"
    version = "0"

    def embed(self, image: np.ndarray, instance) -> np.ndarray | None:
        return None
