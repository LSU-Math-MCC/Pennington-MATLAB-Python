"""Minimal DECA: run only the FLAME encoder (ResNet -> FLAME shape/exp/pose) + FLAME forward to
get the subject's FLAME head verts from a single image. No pytorch3d (we skip DECA's renderer).
Output: runs/flame_<S>_verts.npy (5023 FLAME verts in FLAME canonical frame).

Run (camerahmr env): python tools/face/deca_flame.py <image> <out_npy>
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/DECA"))
from decalib.models.encoders import ResnetEncoder
from decalib.models.FLAME import FLAME
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util
from decalib.datasets import datasets

img_path, out = sys.argv[1], sys.argv[2]
device = "cuda"
mcfg = deca_cfg.model
mcfg.use_tex = False
mcfg.flame_model_path = os.path.expanduser("~/DECA/data/generic_model.pkl")
mcfg.flame_lmk_embedding_path = os.path.expanduser("~/DECA/data/landmark_embedding.npy")

n_param = sum(mcfg.get("n_" + k) for k in mcfg.param_list)
E = ResnetEncoder(outsize=n_param).to(device).eval()
ck = torch.load(os.path.expanduser("~/DECA/data/deca_model.tar"), map_location=device)
util.copy_state_dict(E.state_dict(), ck["E_flame"])
flame = FLAME(mcfg).to(device).eval()

td = datasets.TestData(img_path, iscrop=True, face_detector="fan")
img = td[0]["image"].to(device)[None]
with torch.no_grad():
    params = E(img)
    code = {}; i = 0
    for k in mcfg.param_list:
        n = mcfg.get("n_" + k); code[k] = params[:, i:i + n]; i += n
    verts, lm2d, lm3d = flame(shape_params=code["shape"], expression_params=code["exp"], pose_params=code["pose"])
v = verts[0].cpu().numpy()
np.save(out, v)
print(f"FLAME verts {v.shape} -> {out}  (shape range {v.min(0).round(2)}..{v.max(0).round(2)})")
