"""Voxel-hash fusion of canonical splat clouds with per-region resolution.

Face splats fuse at a smaller voxel size than body splats. Merged splats use
confidence-weighted means for center/color, quaternion averaging for rotation.
"""
from __future__ import annotations

import numpy as np

from .transforms import quat_average
from ..types import SplatCloud


def _voxel_merge(centers, scales, rots, opac, colors, conf, voxel):
    keys = np.floor(centers / voxel).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_s = keys[order]
    uniq, start = np.unique(keys_s, axis=0, return_index=True)
    groups = np.split(np.arange(centers.shape[0])[order], start[1:])
    out_c, out_s, out_r, out_o, out_col, out_cf = [], [], [], [], [], []
    for grp in groups:
        w = conf[grp]
        wsum = w.sum() + 1e-9
        out_c.append((centers[grp] * w[:, None]).sum(0) / wsum)
        out_s.append(np.median(scales[grp], axis=0))
        out_r.append(quat_average(rots[grp], w))
        out_o.append(float(np.max(opac[grp])))
        out_col.append((colors[grp] * w[:, None]).sum(0) / wsum)
        out_cf.append(float(w.max()))
    return (np.array(out_c), np.array(out_s), np.array(out_r),
            np.array(out_o), np.array(out_col), np.array(out_cf))


def fuse_clouds(clouds: list[SplatCloud], body_voxel=0.02, face_voxel=0.006,
                conf_thresh=0.0, region_key="region"):
    """Fuse a list of canonical SplatClouds. Splats tagged region=="face" (in extras
    sidecar array) fuse at face_voxel; everything else at body_voxel.

    Returns (fused SplatCloud, report dict).
    """
    if not clouds:
        empty = SplatCloud(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)),
                           np.zeros(0), np.zeros((0, 3)))
        return empty, {"input": 0, "output": 0}

    centers = np.concatenate([c.centers for c in clouds], 0)
    scales = np.concatenate([c.scales for c in clouds], 0)
    rots = np.concatenate([c.rotations for c in clouds], 0)
    opac = np.concatenate([c.opacities for c in clouds], 0)
    colors = np.concatenate([c.colors for c in clouds], 0)

    def get_arr(key, default):
        parts = []
        for c in clouds:
            n = c.centers.shape[0]
            v = c.extras.get(key)
            if v is None or len(v) != n:
                v = np.full(n, default)
            parts.append(np.asarray(v))
        return np.concatenate(parts, 0)

    conf = get_arr("confidence", 1.0).astype(np.float64)
    region = get_arr(region_key, 0).astype(np.int64)  # 1 => face

    keep = conf >= conf_thresh
    centers, scales, rots, opac, colors, conf, region = (
        centers[keep], scales[keep], rots[keep], opac[keep], colors[keep],
        conf[keep], region[keep])
    n_in = centers.shape[0]

    out_parts = []
    out_region = []
    for rval, voxel in ((1, face_voxel), (0, body_voxel)):
        m = region == rval
        if not np.any(m):
            continue
        merged = _voxel_merge(centers[m], scales[m], rots[m], opac[m], colors[m], conf[m], voxel)
        out_parts.append(merged)
        out_region.append(np.full(merged[0].shape[0], rval))

    if not out_parts:
        fused = SplatCloud(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)),
                           np.zeros(0), np.zeros((0, 3)))
        return fused, {"input": n_in, "output": 0}

    c = np.concatenate([p[0] for p in out_parts], 0)
    s = np.concatenate([p[1] for p in out_parts], 0)
    r = np.concatenate([p[2] for p in out_parts], 0)
    o = np.concatenate([p[3] for p in out_parts], 0)
    col = np.concatenate([p[4] for p in out_parts], 0)
    cf = np.concatenate([p[5] for p in out_parts], 0)
    reg = np.concatenate(out_region, 0)

    fused = SplatCloud(centers=c, scales=s, rotations=r, opacities=o, colors=col,
                       extras={"confidence": cf, region_key: reg})
    report = {"input": int(n_in), "output": int(c.shape[0]),
              "face_out": int((reg == 1).sum()), "body_out": int((reg == 0).sum()),
              "body_voxel": body_voxel, "face_voxel": face_voxel}
    return fused, report


def remove_outliers(cloud: SplatCloud, radius=0.15, min_neighbors=2) -> SplatCloud:
    """Radius outlier removal on centers."""
    from scipy.spatial import cKDTree
    if len(cloud) <= min_neighbors:
        return cloud
    tree = cKDTree(cloud.centers)
    counts = tree.query_ball_point(cloud.centers, r=radius, return_length=True)
    keep = counts > min_neighbors
    if keep.all():
        return cloud
    ext = {k: (np.asarray(v)[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v)
           for k, v in cloud.extras.items()}
    return SplatCloud(cloud.centers[keep], cloud.scales[keep], cloud.rotations[keep],
                      cloud.opacities[keep], cloud.colors[keep], ext)
