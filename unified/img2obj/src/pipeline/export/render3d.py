"""Offscreen 3D renders of a canonical splat/point cloud:

  * a 3D scatter plot (perspective-ish), and
  * orthographic A-pose views (front / side / top + a 3D inset)

Canonical frame: +Y up (spine), +X subject left->right, +Z forward.
Uses a non-interactive matplotlib backend so it runs headless on Windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _prep(points, colors, max_points=25000):
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return pts, None
    if colors is None:
        colors = np.full((pts.shape[0], 3), 0.6)
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size and colors.max() > 1.01:
        colors = colors / 255.0
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts, colors = pts[idx], colors[idx]
    return pts, np.clip(colors, 0, 1)


def _equal_limits(ax, pts):
    if pts.shape[0] == 0:
        return
    mn = pts.min(0); mx = pts.max(0)
    c = 0.5 * (mn + mx)
    r = 0.5 * float((mx - mn).max()) + 1e-6
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    if hasattr(ax, "set_zlim"):
        ax.set_zlim(c[2] - r, c[2] + r)


def render_3d_plot(out_path, points, colors=None, joints=None, title="canonical 3D"):
    pts, cols = _prep(points, colors)
    fig = plt.figure(figsize=(6, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    if pts.shape[0]:
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=cols, s=2, alpha=0.7,
                   edgecolors="none")
    if joints:
        for name, p in joints.items():
            p = np.asarray(p).reshape(-1)[:3]
            ax.scatter([p[0]], [p[2]], [p[1]], c="red", s=24)
    _equal_limits(ax, pts[:, [0, 2, 1]] if pts.shape[0] else pts)
    ax.set_xlabel("X (left-right)"); ax.set_ylabel("Z (forward)"); ax.set_zlabel("Y (up)")
    ax.set_title(title)
    ax.view_init(elev=12, azim=-70)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def render_ortho_views(out_path, points, colors=None, joints=None, title="A-pose ortho"):
    """2x2 panel: front (X-Y), side (Z-Y), top (X-Z), and a 3D inset."""
    pts, cols = _prep(points, colors)
    fig = plt.figure(figsize=(10, 9), dpi=110)

    def scat2d(ax, ax_h, ax_v, hlabel, vlabel, name, invert_v=False):
        if pts.shape[0]:
            ax.scatter(pts[:, ax_h], pts[:, ax_v], c=cols, s=2, alpha=0.7, edgecolors="none")
        if joints:
            for _, p in joints.items():
                p = np.asarray(p).reshape(-1)[:3]
                ax.scatter([p[ax_h]], [p[ax_v]], c="red", s=18)
        ax.set_aspect("equal", "datalim")
        ax.set_xlabel(hlabel); ax.set_ylabel(vlabel); ax.set_title(name)
        ax.grid(alpha=0.2)

    ax1 = fig.add_subplot(2, 2, 1); scat2d(ax1, 0, 1, "X", "Y", "Front (look -Z)")
    ax2 = fig.add_subplot(2, 2, 2); scat2d(ax2, 2, 1, "Z", "Y", "Side (look +X)")
    ax3 = fig.add_subplot(2, 2, 3); scat2d(ax3, 0, 2, "X", "Z", "Top (look -Y)")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    if pts.shape[0]:
        ax4.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=cols, s=2, alpha=0.6, edgecolors="none")
    _equal_limits(ax4, pts[:, [0, 2, 1]] if pts.shape[0] else pts)
    ax4.set_xlabel("X"); ax4.set_ylabel("Z"); ax4.set_zlabel("Y"); ax4.set_title("3D")
    ax4.view_init(elev=12, azim=-70)
    fig.suptitle(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def render_canonical(debug_dir, splats, joints=None, label="canonical"):
    """Write both canonical_3d.png and canonical_ortho.png for a SplatCloud."""
    debug_dir = Path(debug_dir)
    pts = splats.centers
    cols = splats.colors
    render_3d_plot(debug_dir / "canonical_3d.png", pts, cols, joints,
                   title=f"{label} 3D ({len(pts)} pts)")
    render_ortho_views(debug_dir / "canonical_ortho.png", pts, cols, joints,
                       title=f"{label} orthographic A-pose views")
