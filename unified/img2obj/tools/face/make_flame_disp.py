"""EXACT FLAME->SMPL-X head displacement. SMPL-X__FLAME_vertex_ids.npy maps each FLAME vertex to
its SMPL-X vertex index (the head region). So we:
  1. Procrustes-align DECA's FLAME head to the SMPL-X FLAME-region verts (same correspondence),
  2. displacement[smplx_idx[i]] = aligned_DECA_flame[i] - smplx_neutral[smplx_idx[i]].
Because the correspondence is EXACT (not nearest-neighbour), the head deforms to the real face with
no smear and no poke-through. texture_uv_bake applies it -> sharp, undistorted face.

Run (lhm env): python tools/face/make_flame_disp.py <subject>
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

import sys, os
import numpy as np
for a, t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
             ("object", object), ("str", str), ("nan", float("nan")), ("inf", float("inf"))]:
    if not hasattr(np, a):
        setattr(np, a, t)

REPO = _repo
sys.path.insert(0, REPO + "/tools")
S = sys.argv[1] if len(sys.argv) > 1 else "ssp3d_bodybuilder"
GENDER_IMAGE = f"{REPO}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png"
FID = os.path.expanduser("~/LHM/pretrained_models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy")

ids = np.load(FID).astype(int)                            # (5023,) FLAME i -> SMPL-X vertex index
flame = np.load(f"{REPO}/runs/flame_{S}_verts.npy")       # (5023,3) DECA FLAME verts
import lhm_anthropometry as A
betas = np.load(f"{REPO}/runs/camerahmr_{S}_smplx_betas.npy")[:10]
gender, _ = A.estimate_gender([GENDER_IMAGE])
av, aj, af, named = A.smplx_apose(betas, gender=gender, arm_deg=45.0)
av = np.asarray(av)

sx_face = av[ids]                                         # SMPL-X head verts in FLAME-correspondence order


def procrustes(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S0, D0 = src - ms, dst - md
    U, _, Vt = np.linalg.svd(S0.T @ D0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    sc = np.linalg.norm(D0) / (np.linalg.norm(S0) + 1e-9)
    return (sc * (R @ (src - ms).T).T) + md


flame_aligned = procrustes(flame, sx_face)               # DECA FLAME -> SMPL-X frame
disp = np.zeros_like(av)
raw = flame_aligned - sx_face
# limit to the FACE (exclude FLAME scalp/neck rings that don't match SMPL-X well): keep verts whose
# displacement is moderate, taper large ones (robustness)
mag = np.linalg.norm(raw, axis=1)
keep = mag < np.percentile(mag, 96)
disp[ids[keep]] = raw[keep]
np.save(f"{REPO}/runs/flame_disp_{S}.npy", disp.astype(np.float32))
print(f"{S}: FLAME-welded {int(keep.sum())}/{len(ids)} head verts, max disp {mag[keep].max()*1000:.0f}mm")
