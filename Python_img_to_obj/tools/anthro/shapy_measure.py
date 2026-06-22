"""Reproduce SHAPY's virtual anthropometric measurements WITHOUT the gated utility_files
AND without SHAPY's CUDA extension (which won't build under the system's CUDA 12 vs torch
cu117 mismatch). We reproduce the *method* exactly:

  * landmarks  = OPEN published SMPL-X anthropometry vertex indices (DavidBoja/SMPL-Anthropometry)
  * height     = |y(head_top) - y(heel)|                       (SHAPY compute_height)
  * mass       = signed-volume of the closed mesh * 985 kg/m^3 (SHAPY compute_mass, DENSITY=985)
  * chest/waist/hip circumference = total length of the horizontal-plane cross-section at the
    landmark's y-height = sum of per-triangle plane-intersection segment lengths
    (identical to SHAPY compute_peripheries, just CPU triangle-plane intersection vs CUDA).

Run (WSL `lhm` or `shapy` env):  python tools/anthro/shapy_measure.py [s1 s2 ...]
Writes runs/fit_sX/shapy_measurements.json and aggregate runs/SHAPY_measurements.json.
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

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parents[3])
TOOLS = str(Path(REPO) / "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
DENSITY = 985.0  # kg/m^3 (SHAPY)
LM = dict(HEAD_TOP=8976, LEFT_HEEL=8847, LEFT_NIPPLE=3572, BELLY_BUTTON=5939, PUBIC_BONE=5949)


def mesh_volume(v, f):
    """Signed volume of a closed triangle mesh (sum of tetra (origin,tri))."""
    t = v[f]                       # (F,3,3)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    return float(np.abs(np.einsum("ij,ij->i", np.cross(a, b), c).sum()) / 6.0)


def plane_perimeter(v, f, y0, torso_only=True):
    """Length of the horizontal cross-section at y=y0. Each straddling triangle gives one
    segment; sum their lengths. If torso_only, keep ONLY the central connected loop (the torso)
    and drop the separate arm-section loops (else arms-at-side inflate chest/waist girth)."""
    tri = v[f]
    above = tri[:, :, 1] > y0
    na = above.sum(1)
    sel = (na == 1) | (na == 2)
    tri = tri[sel]; ab = above[sel]
    segs = []                                   # (p0, p1, mid_x, mid_z, length)
    for T, A in zip(tri, ab):
        s = A.astype(int)
        odd = [i for i in (0, 1, 2) if s[i] != s[(i + 1) % 3] and s[i] != s[(i + 2) % 3]]
        lone = odd[0] if odd else 0
        pts = []
        for o in [i for i in (0, 1, 2) if i != lone]:
            p0, p1 = T[lone], T[o]
            t = (y0 - p0[1]) / (p1[1] - p0[1] + 1e-12)
            pts.append(p0 + t * (p1 - p0))
        mid = (pts[0] + pts[1]) / 2
        segs.append((mid[0], mid[2], float(np.linalg.norm(pts[0] - pts[1]))))
    if not segs:
        return 0.0
    segs = np.array(segs)
    if not torso_only:
        return float(segs[:, 2].sum())
    # keep the central contiguous cluster in X (torso); arms separated by an empty X-gap
    xs = np.sort(segs[:, 0]);
    cx = np.median(segs[:, 0])
    order = np.argsort(segs[:, 0]); sx = segs[order, 0]
    gaps = np.diff(sx)
    gap_thr = 0.04                              # 4 cm empty band separates arm from torso
    lo, hi = sx[0], sx[-1]
    for i in np.where(gaps > gap_thr)[0]:
        if sx[i] < cx and sx[i] > lo:
            lo = sx[i + 1]
        if sx[i + 1] > cx and sx[i + 1] < hi:
            hi = sx[i]; break
    keep = (segs[:, 0] >= lo - 1e-6) & (segs[:, 0] <= hi + 1e-6)
    return float(segs[keep, 2].sum())


def main():
    import smplx
    import lhm_anthropometry as A
    subjects = sys.argv[1:] or ["s1", "s2", "s3", "s4", "s5"]
    model = smplx.create(A.HUMAN_MODELS, model_type="smplx", gender="neutral", num_betas=10)
    faces = model.faces.astype(np.int64)

    agg = {}
    for s in subjects:
        bp = os.path.join(REPO, "runs", f"fit_{s}", "fused_betas.npy")
        if not os.path.exists(bp):
            print(f"skip {s}: no fused_betas"); continue
        betas = torch.tensor(np.load(bp)[:10], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(betas=betas)
        v = out.vertices[0].numpy()            # shaped, rest-pose (v_shaped equivalent)
        height = abs(v[LM["HEAD_TOP"], 1] - v[LM["LEFT_HEEL"], 1])
        mass = mesh_volume(v, faces) * DENSITY
        chest = plane_perimeter(v, faces, v[LM["LEFT_NIPPLE"], 1])
        waist = plane_perimeter(v, faces, v[LM["BELLY_BUTTON"], 1])
        hips = plane_perimeter(v, faces, v[LM["PUBIC_BONE"], 1])
        rec = {"height_cm": round(height * 100, 1), "mass_kg": round(mass, 1),
               "chest_cm": round(chest * 100, 1), "waist_cm": round(waist * 100, 1),
               "hips_cm": round(hips * 100, 1)}
        json.dump(rec, open(os.path.join(REPO, "runs", f"fit_{s}", "shapy_measurements.json"), "w"), indent=2)
        agg[s] = rec
        print(f"SHAPY_MEAS {s}: height={rec['height_cm']}cm mass={rec['mass_kg']}kg "
              f"chest={rec['chest_cm']} waist={rec['waist_cm']} hips={rec['hips_cm']}")
    json.dump(agg, open(os.path.join(REPO, "runs", "SHAPY_measurements.json"), "w"), indent=2)
    print("SHAPY_MEAS_DONE")


if __name__ == "__main__":
    main()
