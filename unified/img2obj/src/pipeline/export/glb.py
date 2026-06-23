"""GLB export wrapper around trimesh."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_points_glb(path, points: np.ndarray, colors: np.ndarray | None = None):
    """Export a colored point cloud as a GLB (as a trimesh PointCloud)."""
    import trimesh
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float64)
    if colors is not None:
        c = np.clip(np.asarray(colors), 0, 1)
        rgba = (np.c_[c, np.ones(len(c))] * 255).astype(np.uint8)
    else:
        rgba = None
    pc = trimesh.PointCloud(pts, colors=rgba)
    scene = trimesh.Scene(pc)
    scene.export(path)
    return path
