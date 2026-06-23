"""PLY export for splat clouds and point clouds (ASCII, dependency-light)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import SplatCloud


def save_point_ply(path, points: np.ndarray, colors: np.ndarray | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if colors is None:
        cols = np.full((n, 3), 200, dtype=np.uint8)
    else:
        cols = np.clip(np.asarray(colors) * (255 if np.asarray(colors).max() <= 1.0 else 1),
                       0, 255).astype(np.uint8)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                    f"{cols[i,0]} {cols[i,1]} {cols[i,2]}\n")
    return path


def save_splat_ply(path, splats: SplatCloud):
    """Write splats as a points+color PLY (centers) plus sidecar npz for full data.

    The viewer uses points; full splat attributes live in the .npz alongside.
    """
    path = Path(path)
    save_point_ply(path, splats.centers, splats.colors)
    np.savez(
        path.with_suffix(".npz"),
        centers=splats.centers, scales=splats.scales, rotations=splats.rotations,
        opacities=splats.opacities, colors=splats.colors,
        **{k: np.asarray(v) for k, v in splats.extras.items()
           if hasattr(v, "__len__")},
    )
    return path


def load_splat_npz(path) -> SplatCloud:
    d = np.load(path, allow_pickle=True)
    extras = {k: d[k] for k in d.files
              if k not in ("centers", "scales", "rotations", "opacities", "colors")}
    return SplatCloud(centers=d["centers"], scales=d["scales"], rotations=d["rotations"],
                      opacities=d["opacities"], colors=d["colors"], extras=extras)
