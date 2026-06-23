"""Assign splats to a subject instance using mask + depth + nearest selected sample.

Method A: project splat center, require it lands inside the person mask at a depth
          consistent with the mask-gated depth map.
Method B: nearest distance from splat center to the selected masked 3D point cloud.
Combined: person_score = 0.7 * projection_depth_score + 0.3 * nearest_sample_score.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import camera as camlib
from .projection import project_centers, sample_map
from ..types import Camera, SplatCloud, SelectedDistanceSamples


def _proj_depth_score(cam, centers, mask, depth, tau_depth):
    uv, z_cam, valid = project_centers(cam, centers)
    in_mask, _ = sample_map(mask.astype(bool), uv)
    d_map, _ = sample_map(depth.astype(np.float64), uv, default=np.nan)
    good = valid & in_mask & np.isfinite(d_map) & (d_map > 0)
    score = np.zeros(centers.shape[0])
    if np.any(good):
        diff = np.abs(z_cam[good] - d_map[good])
        score[good] = np.exp(-(diff / max(tau_depth, 1e-6)) ** 2)
    return score, uv, z_cam, valid


def _nearest_sample_score(centers, samples: SelectedDistanceSamples, tau_3d):
    score = np.zeros(centers.shape[0])
    if len(samples) == 0:
        return score
    tree = cKDTree(samples.points_world)
    dist, _ = tree.query(centers, k=1)
    score = np.exp(-(dist / max(tau_3d, 1e-6)) ** 2)
    return score


def assign_splats_to_instance(
    splats: SplatCloud,
    cam: Camera,
    mask: np.ndarray,
    depth: np.ndarray,
    samples: SelectedDistanceSamples,
    depth_tau: float = 0.05,
    tau_3d: float = 0.1,
    person_threshold: float = 0.35,
    median_person_depth: float | None = None,
):
    """Return (indices, scores_full) where indices are splats above threshold.

    scores_full is the per-splat person_score in [0,1] for ALL splats.
    depth_tau is treated as a fraction of median person depth (adaptive) when that
    is available, else as an absolute threshold.
    """
    centers = splats.centers
    if median_person_depth is None and len(samples) > 0:
        median_person_depth = float(np.median(samples.depths))
    tau_depth_abs = depth_tau * median_person_depth if median_person_depth else depth_tau
    tau_depth_abs = max(tau_depth_abs, 1e-4)

    proj_score, uv, z_cam, valid = _proj_depth_score(cam, centers, mask, depth, tau_depth_abs)
    near_score = _nearest_sample_score(centers, samples, tau_3d)
    person_score = 0.7 * proj_score + 0.3 * near_score
    person_score = np.clip(person_score, 0.0, 1.0)
    indices = np.nonzero(person_score >= person_threshold)[0]
    return indices, person_score


def resolve_ambiguous(scores_per_instance: list[np.ndarray], margin: float = 0.15):
    """Given a list of per-instance score arrays (each length N), decide assignment.

    Returns (assignment: int array length N, ambiguous: bool array length N).
    assignment[i] = winning instance index, or -1 if no instance claims it.
    A splat is ambiguous if the top-2 instance scores are within `margin` and both
    are non-trivial.
    """
    if not scores_per_instance:
        return np.zeros(0, int), np.zeros(0, bool)
    S = np.stack(scores_per_instance, axis=1)  # N x K
    N, K = S.shape
    order = np.argsort(-S, axis=1)
    top1 = S[np.arange(N), order[:, 0]]
    assignment = np.where(top1 > 0, order[:, 0], -1)
    ambiguous = np.zeros(N, bool)
    if K >= 2:
        top2 = S[np.arange(N), order[:, 1]]
        ambiguous = (top1 > 0) & ((top1 - top2) < margin) & (top2 > 0)
    return assignment, ambiguous
