"""Compute the atlas texels that belong to the HEAD (fixed across subjects: UV + head vertex set
are fixed). Used to PROTECT the face from the body's dark-patch cleanup (eyes/brows/lips are dark
and must not be replaced with skin). Saves runs/head_texel_mask_<A>.npy (A,A bool)."""

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
import texture_uv_bake as TB

A_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
vt, fv, fvt = TB.load_uv_obj(TB.UV_OBJ)
texel_v, bary = TB.rasterize_uv(vt, fvt, fv, A_size)
head_vert = TB._head_vertex_mask(np.zeros(10))            # fixed head vertex set (neutral)
valid = texel_v[:, :, 0] >= 0
head_texel = np.zeros((A_size, A_size), bool)
head_texel[valid] = head_vert[texel_v[valid]].any(1)
# dilate a little so the whole face island is protected
from scipy.ndimage import binary_dilation
head_texel = binary_dilation(head_texel, iterations=3)
np.save(f"{REPO}/runs/head_texel_mask_{A_size}.npy", head_texel)
print(f"head texels {int(head_texel.sum())}/{A_size*A_size} ({head_texel.mean():.1%}) saved")
