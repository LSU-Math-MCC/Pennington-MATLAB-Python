"""Fill unmapped (default-grey) texels in a baked UV atlas with the median observed SKIN tone, so
unobserved body regions render as skin instead of grey. Overwrites the glb's texture in place.

Run (lhm env): python tools/texture/fill_grey_texture.py <subject>
"""
import sys, os
import numpy as np
for a, t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
             ("object", object), ("str", str), ("nan", float("nan")), ("inf", float("inf"))]:
    if not hasattr(np, a):
        setattr(np, a, t)
import trimesh
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
S = sys.argv[1] if len(sys.argv) > 1 else "s1"
glb = f"{REPO}/runs/uv_{S}_1v/apose_textured_uv.glb"

m = trimesh.load(glb, process=False, force="mesh")
mat = m.visual.material
img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
if img is None:
    print("no texture image"); sys.exit(1)
a = np.asarray(img.convert("RGB")).astype(np.float32)

# grey/unmapped texels = near (128,128,128) and low saturation
mx = a.max(2); mn = a.min(2)
grey = (np.abs(a - 128).sum(2) < 60) & ((mx - mn) < 18)
# observed skin = bright, low-blue (exclude swimsuit), not grey
sat = mx - mn
obs = (~grey) & (a.sum(2) > 160)
notblue = ~((a[:, :, 2] > a[:, :, 0] + 20) & (a[:, :, 2] > a[:, :, 1] + 20))
skinmask = obs & notblue & (a[:, :, 0] > a[:, :, 2])      # reddish = skin
skin = a[skinmask]
skin_med = np.median(skin, 0) if len(skin) > 200 else np.array([200., 160., 140.])
print(f"{S}: grey texels {int(grey.sum())}, skin median {skin_med.round().astype(int)}")

a[grey] = skin_med

# DARK-PATCH cleanup: texels that are much darker than this subject's skin AND not the (blue)
# swimsuit are shadow/projection artifacts (e.g. s3 chest, s4 thigh) -> replace with skin.
# CRITICAL: PROTECT the HEAD region -- eyes/brows/nostrils/lips are legitimately dark and must
# NOT be replaced with skin (that's what made faces patchy).
hm_path = os.path.join(REPO, f"runs/head_texel_mask_{a.shape[0]}.npy")
head_mask = np.load(hm_path) if os.path.exists(hm_path) else np.zeros(a.shape[:2], bool)
skin_bright = skin_med.sum()
isblue = (a[:, :, 2] > a[:, :, 0] + 18) & (a[:, :, 2] > a[:, :, 1] + 18)
near_white = a.sum(2) > 660
dark_patch = (a.sum(2) < skin_bright * 0.62) & (~isblue) & (~near_white) & (~grey) & (~head_mask)
a[dark_patch] = skin_med
print(f"{S}: dark-patch texels {int(dark_patch.sum())} -> skin (head protected: {int(head_mask.sum())} texels)")

filled = Image.fromarray(a.clip(0, 255).astype(np.uint8))
# write back into the material and re-export
if hasattr(mat, "baseColorTexture"):
    m.visual.material.baseColorTexture = filled
else:
    m.visual.material.image = filled
m.export(glb)
print("overwrote", glb)
