"""PROPER MANO->SMPL-X hand weld. SMPL-X hands ARE MANO topology; MANO_SMPLX_vertex_ids.pkl gives
the exact SMPL-X vertex index for each of MANO's 778 verts. So we:
  1. align HaMeR's MANO hand to the body's SMPL-X hand verts by 778-point Procrustes (rot+scale+
     trans, det=+1 -> NO orientation ambiguity, the thing that broke the ICP graft),
  2. replace the SMPL-X hand verts with the aligned HaMeR verts.
The articulated HaMeR fingers now sit exactly where the mitten hand was, welded into the body mesh.

Run (lhm env): python tools/hands/weld_hands.py <subject>
"""
import sys, os, pickle
import numpy as np
for a, t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
             ("object", object), ("str", str), ("nan", float("nan")), ("inf", float("inf"))]:
    if not hasattr(np, a):
        setattr(np, a, t)
import trimesh

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
S = sys.argv[1] if len(sys.argv) > 1 else "s1"
ids = pickle.load(open(os.path.expanduser("~/LHM/pretrained_models/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl"), "rb"), encoding="latin1")
LID = np.asarray(ids["left_hand"]).ravel()
RID = np.asarray(ids["right_hand"]).ravel()
hamer_dir = f"{REPO}/runs/hamer_out" if S == "s1" else f"{REPO}/runs/hamer_{S}"

body = trimesh.load(f"{REPO}/runs/uv_{S}_1v/apose_textured_uv.glb", process=False, force="mesh")
bv = np.asarray(body.vertices).copy()


def procrustes(src, dst):
    """similarity src->dst (rot+uniform scale+trans), proper rotation (det=+1)."""
    ms, md = src.mean(0), dst.mean(0)
    S0, D0 = src - ms, dst - md
    U, _, Vt = np.linalg.svd(S0.T @ D0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = (np.linalg.norm(D0) / (np.linalg.norm(S0) + 1e-9))
    return lambda P: (scale * (R @ (P - ms).T).T) + md


for tag, idx in [("L", LID), ("R", RID)]:
    hp = f"{hamer_dir}/hand_{tag}_flat.obj"       # FLAT/open HaMeR hand (rest pose ~ SMPL-X flat -> welds cleanly)
    if not os.path.exists(hp):
        print("missing", hp); continue
    hand = trimesh.load(hp, process=False)
    hv = np.asarray(hand.vertices)
    if len(hv) != len(idx):
        print(f"vert mismatch {tag}: hamer {len(hv)} vs smplx hand {len(idx)}"); continue
    target = bv[idx]                              # SMPL-X mitten-hand verts (correspondence order)
    warp = procrustes(hv, target)
    bv[idx] = warp(hv)                            # weld: replace with aligned articulated fingers
    print(f"welded {tag} hand: {len(idx)} verts")

welded = trimesh.Trimesh(bv, body.faces, visual=body.visual, process=False)
welded.export(f"{REPO}/runs/uv_{S}_1v/welded.glb")
print("saved welded.glb")
