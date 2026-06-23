"""Core algorithm: the 2D mask gates which depth samples are trusted.

This is the central geometric bridge of the whole pipeline. We do NOT color-label
splats from the mask. We use the mask to select depth/distance samples, backproject
them to 3D, and later associate those samples with splats.
"""
from __future__ import annotations

import numpy as np

from . import camera as camlib
from ..types import Camera, DepthResult, SelectedDistanceSamples


def select_masked_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    cam: Camera,
    confidence: np.ndarray | None = None,
    depth_min: float = 1e-3,
    depth_max: float = 1e6,
    conf_thresh: float = 0.0,
    subsample: int | None = None,
) -> SelectedDistanceSamples:
    """Select trusted depth samples inside `mask` and backproject them to 3D.

    Returns a SelectedDistanceSamples with pixels (u,v), depths, camera- and
    world-space points, and per-sample confidence.
    """
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float64)
    H, W = depth.shape
    valid_depth = np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
    sel = mask & valid_depth
    if confidence is not None:
        sel = sel & (np.asarray(confidence, dtype=np.float64) >= conf_thresh)

    vs, us = np.nonzero(sel)
    if us.size == 0:
        return SelectedDistanceSamples(
            pixels=np.zeros((0, 2)), depths=np.zeros(0),
            points_cam=np.zeros((0, 3)), points_world=np.zeros((0, 3)),
            confidence=np.zeros(0),
        )

    if subsample and us.size > subsample:
        idx = np.linspace(0, us.size - 1, subsample).astype(int)
        us, vs = us[idx], vs[idx]

    uv = np.stack([us, vs], axis=1).astype(np.float64)
    d = depth[vs, us]
    pts_cam = camlib.backproject(cam, uv, d)
    pts_world = camlib.cam_to_world(cam, pts_cam)
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64)[vs, us]
    else:
        conf = np.ones(us.size)
    return SelectedDistanceSamples(
        pixels=uv, depths=d, points_cam=pts_cam, points_world=pts_world, confidence=conf
    )


def save_samples(path, samples: SelectedDistanceSamples) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        pixels=samples.pixels, depths=samples.depths,
        points_cam=samples.points_cam, points_world=samples.points_world,
        confidence=samples.confidence,
    )


def load_samples(path) -> SelectedDistanceSamples:
    d = np.load(path)
    return SelectedDistanceSamples(
        pixels=d["pixels"], depths=d["depths"], points_cam=d["points_cam"],
        points_world=d["points_world"], confidence=d["confidence"],
    )
