"""Slim HaMeR runner — SOTA finger-level MANO hands WITHOUT detectron2/vitpose.

HaMeR's demo uses detectron2 (person) + ViTPose (hand keypoints) ONLY to produce hand
bounding boxes. We already have hand keypoints from Multi-HMR (the pose estimator). So we
build the per-hand bbox from Multi-HMR's SMPL-X hand joints and feed it straight into the
HaMeR model -> pred_vertices (778 MANO verts/hand, articulated fingers). detectron2 never runs.

Step 1 (Multi-HMR) and step 2 (HaMeR) load sequentially so only ONE model is on the GPU at a
time (simultaneous loads previously OOM'd WSL).

Run (WSL lhm): python tools/hands/run_hamer_slim.py <img> <out_dir>
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

import os, sys, json
import numpy as np
# chumpy (MANO .pkl loader) does `from numpy import bool,int,float,...` -> removed in numpy>=1.24.
for _a, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str), ("nan", float("nan")), ("inf", float("inf"))]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)

REPO = _repo
sys.path.insert(0, REPO + "/tools")
IMG = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else REPO + "/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png"
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else REPO + "/runs/hamer_out"
os.makedirs(OUT, exist_ok=True)

# SMPL-X 127-keypoint layout (Multi-HMR): 0-21 body, 22-36 L-hand, 37-51 R-hand.
LWR, RWR = 20, 21
LHAND = [20] + list(range(22, 37))
RHAND = [21] + list(range(37, 52))


def hand_bbox(j2d, idxs, W, H):
    p = j2d[idxs]
    p = p[(p[:, 0] > 0) & (p[:, 1] > 0) & (p[:, 0] < W) & (p[:, 1] < H)]
    if len(p) < 4:
        return None
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    x1, y1 = p[:, 0].max(), p[:, 1].max()
    return [float(x0), float(y0), float(x1), float(y1)]


def main():
    import cv2
    img_cv2 = cv2.imread(IMG)
    H, W = img_cv2.shape[:2]

    # ---- Step 1: Multi-HMR hand keypoints (then free GPU) ----
    import lhm_anthropometry as A, texture_uv_bake as TB
    _, h, j2d = TB.posed_view(A._estimator(), IMG)
    j2d = np.asarray(j2d, dtype=np.float32)
    boxes, right = [], []
    lb = hand_bbox(j2d, LHAND, W, H)
    rb = hand_bbox(j2d, RHAND, W, H)
    if lb: boxes.append(lb); right.append(0)
    if rb: boxes.append(rb); right.append(1)
    if not boxes:
        print("no hands found in Multi-HMR j2d"); return
    boxes = np.array(boxes, np.float32); right = np.array(right)
    print(f"hand boxes: L={lb is not None} R={rb is not None}")
    import torch, gc
    del A, TB
    gc.collect(); torch.cuda.empty_cache()

    # ---- Step 2: HaMeR ----
    sys.path.insert(0, os.path.expanduser("~/hamer"))
    os.chdir(os.path.expanduser("~/hamer"))   # HaMeR loads config/ckpt by RELATIVE path
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to
    model, cfg = load_hamer(DEFAULT_CHECKPOINT)
    dev = torch.device("cuda")
    model = model.to(dev).eval()

    ds = ViTDetDataset(cfg, img_cv2, boxes, right, rescale_factor=2.0)
    import trimesh
    faces = model.mano.faces
    allr = []
    for batch in torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0):
        batch = recursive_to(batch, dev)
        with torch.no_grad():
            out = model(batch)
        v = out["pred_vertices"].cpu().numpy()
        t = out["pred_cam_t"].cpu().numpy()
        mp = out["pred_mano_params"]
        betas = mp["betas"].cpu().numpy()               # MANO SHAPE (finger proportions)
        for n in range(v.shape[0]):
            isr = int(batch["right"][n].cpu())
            tag = "R" if isr else "L"
            vv = v[n].copy()
            f = faces if isr else faces[:, ::-1]
            if not isr:
                vv[:, 0] = -vv[:, 0]                     # left = mirror right-hand MANO
            trimesh.Trimesh(vv + t[n], f, process=False).export(os.path.join(OUT, f"hand_{tag}.obj"))
            # FLAT A-pose hand: subject's MANO betas, hand_pose=0 (open rest pose)
            with torch.no_grad():
                eye = torch.eye(3, device=dev)
                flat = model.mano(betas=torch.tensor(betas[n:n + 1], dtype=torch.float32, device=dev),
                                  global_orient=eye.reshape(1, 1, 3, 3),
                                  hand_pose=eye.reshape(1, 1, 3, 3).repeat(1, 15, 1, 1)).vertices[0].cpu().numpy()
            if not isr:
                flat[:, 0] = -flat[:, 0]
            trimesh.Trimesh(flat, f, process=False).export(os.path.join(OUT, f"hand_{tag}_flat.obj"))
            np.save(os.path.join(OUT, f"hand_{tag}_betas.npy"), betas[n])
            allr.append(isr)
            print(f"  saved hand_{tag}.obj + hand_{tag}_flat.obj  verts={len(vv)}")
    json.dump({"img": IMG, "hands": [("R" if r else "L") for r in allr],
               "n_verts_per_hand": 778}, open(os.path.join(OUT, "hamer_meta.json"), "w"), indent=2)
    print("HAMER_SLIM_DONE")


if __name__ == "__main__":
    main()
