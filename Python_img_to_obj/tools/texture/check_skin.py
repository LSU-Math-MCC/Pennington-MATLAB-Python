"""Visualize SKIN-ONLY + depth-fused person matte for honest body-appearance fusion."""

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

import sys, glob
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "src"))
import texture_uv_bake as TB

def panel(path, out):
    img = np.asarray(Image.open(path).convert("RGB"))
    pm = TB.person_mask(img)            # SAM2 (+depth fuse)
    sk = TB.skin_mask(img)
    skin_in = pm & sk
    vis = img.copy()
    vis[~pm] = (vis[~pm] * 0.25).astype(np.uint8)          # dim background
    over = img.copy()
    over[skin_in] = (0.4 * over[skin_in] + np.array([0, 153, 102]) * 0.6).astype(np.uint8)  # skin=green
    grid = np.concatenate([vis, over], 1)
    Image.fromarray(grid).save(out)
    print(f"{path}: person={pm.mean():.3f} skin={sk.mean():.3f} skin_in_person={skin_in.mean():.3f} -> {out}")

if __name__ == "__main__":
    root = REPO
    gs = sorted(glob.glob(f"{root}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png"))
    panel(gs[0], f"{root}/runs/skin_ssp3d_bodybuilder.png")
