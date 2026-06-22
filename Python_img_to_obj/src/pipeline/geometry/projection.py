"""Convenience projection helpers shared by assignment / visibility code."""
from __future__ import annotations

import numpy as np

from . import camera as camlib
from ..types import Camera


def project_centers(cam: Camera, centers: np.ndarray):
    """Project splat centers. Returns uv (Nx2), z_cam (N), valid (N bool)."""
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    x_cam = camlib.world_to_cam(cam, centers)
    uv, valid = camlib.project(cam, centers)
    return uv, x_cam[:, 2], valid


def sample_map(arr: np.ndarray, uv: np.ndarray, default=np.nan):
    """Nearest-pixel sample of a HxW map at float uv coords. Out-of-bounds -> default."""
    arr = np.asarray(arr)
    H, W = arr.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = np.full(uv.shape[0], default, dtype=np.float64 if arr.dtype != bool else bool)
    if arr.dtype == bool:
        out = np.zeros(uv.shape[0], dtype=bool)
    out_vals = arr[np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)]
    out[inb] = out_vals[inb]
    return out, inb
