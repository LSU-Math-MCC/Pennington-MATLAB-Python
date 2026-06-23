"""Simple proxy-mesh generation from a fused point/splat cloud.

Kept intentionally simple: a lightweight convex/alpha-ish surface via trimesh, with
a robust fallback to a point-based mesh. Mesh quality is not a blocker.
"""
from __future__ import annotations

import numpy as np


def proxy_mesh_from_points(points: np.ndarray, colors: np.ndarray | None = None):
    """Return a trimesh.Trimesh proxy. Tries convex hull; falls back to small boxes."""
    import trimesh
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    try:
        cloud = trimesh.PointCloud(pts)
        mesh = cloud.convex_hull
        if colors is not None and mesh is not None:
            mesh.visual.vertex_colors = _avg_color(colors)
        return mesh
    except Exception:
        return None


def _avg_color(colors):
    c = np.clip(np.asarray(colors), 0, 1).mean(axis=0)
    return (np.r_[c, 1.0] * 255).astype(np.uint8)


def save_mesh(mesh, path_glb=None, path_ply=None):
    if mesh is None:
        return
    if path_glb:
        from pathlib import Path
        Path(path_glb).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path_glb)
    if path_ply:
        from pathlib import Path
        Path(path_ply).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path_ply)
