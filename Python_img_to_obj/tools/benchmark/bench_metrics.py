"""SHAPY-compatible shape + measurement metrics (per docs/PROJECT.md §9).
Identical measurement extraction is used for ALL methods (no per-method measurement code).
Ready for HBW the moment the dataset is present (it is currently absent -> verdict blocks).
"""

# --- tool-path bootstrap (capability-tree reorg): make sibling tool modules importable ---
import os as _os, sys as _sys
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.exists(_os.path.join(_repo, "pyproject.toml")):
    _repo = _os.path.dirname(_repo)
for _sub in ("smplx", "texture", "benchmark", "geometry", "anthro", "render"):
    _p = _os.path.join(_repo, "tools", _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def p2p20k(pred_v, gt_v, sample_idx=None):
    """Sampled surface point-to-point error (mm). SHAPY's main HBW metric (avoids SMPL-X non-
    uniform vertex-density bias). Assumes shared topology + benchmark alignment; samples 20k."""
    if sample_idx is None:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(pred_v), size=min(20000, len(pred_v)), replace=False)
    d = np.linalg.norm(pred_v[sample_idx] - gt_v[sample_idx], axis=1)
    return float(d.mean() * 1000.0)


def v2v(pred_v, gt_v):
    return float(np.linalg.norm(pred_v - gt_v, axis=1).mean() * 1000.0)


def pa_v2v(pred_v, gt_v):
    """Procrustes-aligned V2V (mm) — diagnostic only."""
    def center(x):
        return x - x.mean(0)
    P, G = center(pred_v), center(gt_v)
    U, _, Vt = np.linalg.svd(P.T @ G)
    R = (U @ Vt).T
    s = np.trace(np.linalg.svd(P.T @ G, compute_uv=False)) / (P ** 2).sum()
    return float(np.linalg.norm((s * P @ R.T) - G, axis=1).mean() * 1000.0)


def measurements_mm(verts, faces, lm):
    """height/chest/waist/hips in MM via the SHARED reproduced-SHAPY plane-section method."""
    import sys
    tools_dir = str(REPO / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import shapy_measure as SM
    return {"height_mm": abs(verts[lm["HEAD_TOP"], 1] - verts[lm["LEFT_HEEL"], 1]) * 1000.0,
            "chest_mm": SM.plane_perimeter(verts, faces, verts[lm["LEFT_NIPPLE"], 1]) * 1000.0,
            "waist_mm": SM.plane_perimeter(verts, faces, verts[lm["BELLY_BUTTON"], 1]) * 1000.0,
            "hips_mm": SM.plane_perimeter(verts, faces, verts[lm["PUBIC_BONE"], 1]) * 1000.0}


def measurement_mae(pred_m, gt_m):
    return {k: abs(pred_m[k] - gt_m[k]) for k in ("height_mm", "chest_mm", "waist_mm", "hips_mm")}


def beats_hbw(metrics, shapy):
    """Decision rule: P2P20K lower AND >=3 of 4 measurement errors lower; no measurement regressed
    >10% unless P2P20K improves >20% (explained)."""
    p_ok = metrics["p2p20k_mm"] < shapy["p2p20k_mm"]
    keys = ["height_mm", "chest_mm", "waist_mm", "hips_mm"]
    lower = sum(metrics[k] < shapy[k] for k in keys)
    big_p = metrics["p2p20k_mm"] < 0.8 * shapy["p2p20k_mm"]
    regress = any(metrics[k] > 1.1 * shapy[k] for k in keys)
    return bool(p_ok and lower >= 3 and (not regress or big_p))
