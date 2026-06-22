"""Dummy backends: synthetic, GPU-free, geometrically self-consistent.

These let the entire pipeline (selection -> assignment -> canonicalization -> fusion
-> export) and all geometry tests run without any ML model. The synthetic splats are
backprojected from the masked depth so mask/depth/splat geometry is consistent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import (SplatCloud, SegmentationResult, Pose2DResult, Face2DResult,
                     DepthResult, Camera)
from ..geometry import camera as camlib
from . import base


def _person_mask(H, W, cx_frac=0.5, w_frac=0.5, h_frac=0.85):
    mask = np.zeros((H, W), bool)
    cx = int(W * cx_frac)
    bw = int(W * w_frac)
    bh = int(H * h_frac)
    x0 = max(0, cx - bw // 2)
    x1 = min(W, cx + bw // 2)
    y0 = int(H * 0.08)
    y1 = min(H, y0 + bh)
    yy, xx = np.mgrid[0:H, 0:W]
    # body as a vertical ellipse-ish region
    ex = (x0 + x1) / 2
    ey = (y0 + y1) / 2
    rx = (x1 - x0) / 2
    ry = (y1 - y0) / 2
    mask = ((xx - ex) / (rx + 1e-6)) ** 2 + ((yy - ey) / (ry + 1e-6)) ** 2 <= 1.0
    return mask, (x0, y0, x1, y1)


def _pose_in_bbox(bbox):
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    def P(fx, fy, c=0.9):
        return (x0 + fx * w, y0 + fy * h, c)
    kp = {
        "nose": P(0.5, 0.08), "left_eye": P(0.45, 0.06), "right_eye": P(0.55, 0.06),
        "left_ear": P(0.4, 0.08), "right_ear": P(0.6, 0.08),
        "left_shoulder": P(0.32, 0.25), "right_shoulder": P(0.68, 0.25),
        "left_elbow": P(0.24, 0.45), "right_elbow": P(0.76, 0.45),
        "left_wrist": P(0.2, 0.62), "right_wrist": P(0.8, 0.62),
        "left_hip": P(0.4, 0.58), "right_hip": P(0.6, 0.58),
        "left_knee": P(0.4, 0.8), "right_knee": P(0.6, 0.8),
        "left_ankle": P(0.4, 0.98), "right_ankle": P(0.6, 0.98),
    }
    return kp


SKELETON = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
            ("nose", "left_eye"), ("nose", "right_eye")]


class DummyGS(base.GSReconstructionBackend):
    name = "dummy-gs"
    version = "1"

    def reconstruct(self, image: np.ndarray, out_dir: Path):
        H, W = image.shape[:2]
        cam = camlib.default_camera(W, H)
        mask, bbox = _person_mask(H, W)
        depth = np.where(mask, 2.0, np.nan)
        # sample masked pixels, backproject at depth 2.0 (+small z relief)
        vs, us = np.nonzero(mask)
        if us.size > 4000:
            idx = np.linspace(0, us.size - 1, 4000).astype(int)
            us, vs = us[idx], vs[idx]
        ex, ey = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        rx = (bbox[2] - bbox[0]) / 2 + 1e-6
        relief = 0.15 * np.sqrt(np.clip(1 - ((us - ex) / rx) ** 2, 0, 1))
        d = 2.0 - relief
        uv = np.stack([us, vs], 1).astype(float)
        centers = camlib.backproject_to_world(cam, uv, d)
        n = centers.shape[0]
        colors = image[vs, us].astype(np.float64) / 255.0
        scales = np.full((n, 3), 0.01)
        rots = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
        opac = np.full(n, 0.9)
        splats = SplatCloud(centers, scales, rots, opac, colors, extras={})
        return splats, cam


class DummySeg(base.SegmentationBackend):
    name = "dummy-seg"
    version = "1"

    def segment_people(self, image: np.ndarray, out_dir: Path):
        H, W = image.shape[:2]
        mask, bbox = _person_mask(H, W)
        return [SegmentationResult(person_mask=mask, confidence=mask.astype(float), bbox=bbox)]


class DummyPose(base.PoseBackend):
    name = "dummy-pose"
    version = "1"

    def estimate_pose(self, image: np.ndarray, out_dir: Path, bbox=None):
        H, W = image.shape[:2]
        if bbox is None:
            _, bbox = _person_mask(H, W)
        return Pose2DResult(keypoints=_pose_in_bbox(bbox), skeleton_edges=SKELETON)


class DummyFace(base.FaceBackend):
    name = "dummy-face"
    version = "1"

    def estimate_face(self, image: np.ndarray, out_dir: Path, bbox=None):
        H, W = image.shape[:2]
        if bbox is None:
            _, bbox = _person_mask(H, W)
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        def P(fx, fy):
            return (x0 + fx * w, y0 + fy * h, 0.9)
        lm = {
            "left_eye_outer": P(0.42, 0.06), "left_eye_inner": P(0.47, 0.06),
            "right_eye_inner": P(0.53, 0.06), "right_eye_outer": P(0.58, 0.06),
            "left_eye": P(0.445, 0.06), "right_eye": P(0.555, 0.06),
            "nose_tip": P(0.5, 0.10), "nose_bridge": P(0.5, 0.07),
            "mouth_left": P(0.46, 0.14), "mouth_right": P(0.54, 0.14),
            "upper_lip": P(0.5, 0.13), "lower_lip": P(0.5, 0.15),
            "chin": P(0.5, 0.18),
            "left_face_contour": P(0.4, 0.12), "right_face_contour": P(0.6, 0.12),
        }
        fx0, fy0 = x0 + 0.38 * w, y0 + 0.03 * h
        fx1, fy1 = x0 + 0.62 * w, y0 + 0.20 * h
        return Face2DResult(face_mask=None, landmarks=lm,
                            bbox=(fx0, fy0, fx1, fy1), confidence=0.85)


class DummyDepth(base.DepthBackend):
    name = "dummy-depth"
    version = "1"

    def estimate_or_render_depth(self, image, splats, camera, out_dir):
        H, W = image.shape[:2]
        mask, bbox = _person_mask(H, W)
        ex = (bbox[0] + bbox[2]) / 2
        rx = (bbox[2] - bbox[0]) / 2 + 1e-6
        xx = np.arange(W)[None, :].repeat(H, 0)
        relief = 0.15 * np.sqrt(np.clip(1 - ((xx - ex) / rx) ** 2, 0, 1))
        depth = np.where(mask, 2.0 - relief, np.nan)
        conf = mask.astype(float)
        return DepthResult(depth=depth, confidence=conf)


class DummyAssoc(base.SubjectAssociationBackend):
    name = "dummy-assoc"
    version = "1"

    def embed(self, image, instance):
        return None


def make_backend_set():
    from . import BackendSet
    return BackendSet(
        gs=DummyGS(), segment=DummySeg(), pose=DummyPose(), face=DummyFace(),
        depth=DummyDepth(), association=DummyAssoc(),
        versions={"gs": "dummy-1", "seg": "dummy-1", "pose": "dummy-1",
                  "face": "dummy-1", "depth": "dummy-1", "assoc": "dummy-1"},
    )
