"""Transfer CameraHMR's SOTA SMPL shape -> SMPL-X betas via T-pose VERTEX correspondence.

Correct method (no gated files): compute, once on the NEUTRAL meshes, a nearest-vertex map from
each SMPL-X body vertex to a SMPL vertex (keep only SMPL-X verts close to the SMPL surface =>
excludes SMPL-X-only hands/face). Then fit SMPL-X betas so the SMPL-X body matches SMPL(cam_betas)
at those corresponding vertices. Faithful at the vertex level => realistic betas, no band drift.

Run (lhm env): python tools/hmr/camerahmr/camerahmr_to_smplx.py
"""
import os, sys
import numpy as np
for _a, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str), ("nan", float("nan")), ("inf", float("inf"))]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
SMPL_DIR = os.path.expanduser("~/shapy/data/body_models/smpl")
SMPLX_DIR = os.path.expanduser("~/LHM/pretrained_models/human_model_files/smplx")
SUBJECTS = ["s1", "s2", "s3", "s4", "s5"]


def main():
    import smplx
    from scipy.spatial import cKDTree
    from scipy.optimize import least_squares
    smpl = smplx.SMPL(SMPL_DIR, gender="neutral")
    smplx_m = smplx.SMPLX(SMPLX_DIR, gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=10)

    # neutral T-pose meshes -> correspondence (align by pelvis/scale first; both already ~aligned)
    with torch.no_grad():
        vs0 = smpl(betas=torch.zeros(1, 10)).vertices[0].numpy()
        vx0 = smplx_m(betas=torch.zeros(1, 10)).vertices[0].numpy()
    # center both at pelvis-ish (mean) for a clean nearest-vertex map
    vs0c = vs0 - vs0.mean(0); vx0c = vx0 - vx0.mean(0)
    tree = cKDTree(vs0c)
    dist, corr = tree.query(vx0c)
    body = dist < 0.03                      # SMPL-X verts lying on the SMPL body surface (drop hands/face)
    print(f"correspondence: {body.sum()}/{len(vx0)} SMPL-X body verts matched (median d={np.median(dist[body])*1000:.1f}mm)")
    bidx = np.where(body)[0]
    cidx = corr[body]

    for s in SUBJECTS:
        bp = f"{REPO}/runs/camerahmr_{s}_betas.npy"
        if not os.path.exists(bp):
            continue
        cam_betas = np.load(bp)
        with torch.no_grad():
            vs = smpl(betas=torch.tensor(cam_betas).float().unsqueeze(0)).vertices[0].numpy()
        vs = vs - vs.mean(0)
        target = vs[cidx]                   # SMPL target positions for the matched SMPL-X verts

        def resid(b):
            with torch.no_grad():
                vx = smplx_m(betas=torch.tensor(b).float().unsqueeze(0)).vertices[0].numpy()
            vx = vx - vx.mean(0)
            return (vx[bidx] - target).reshape(-1)
        sol = least_squares(resid, np.full(10, 0.1), method="trf", max_nfev=120,
                            bounds=(-4 * np.ones(10), 4 * np.ones(10)), diff_step=np.full(10, 0.1))
        np.save(f"{REPO}/runs/camerahmr_{s}_smplx_betas.npy", sol.x)
        rms = np.sqrt((resid(sol.x) ** 2).mean()) * 1000
        with torch.no_grad():
            vx = smplx_m(betas=torch.tensor(sol.x).float().unsqueeze(0)).vertices[0].numpy()
        h_s = vs[:, 1].ptp(); h_x = vx[:, 1].ptp()
        print(f"{s}: SMPL-X betas {sol.x[:5].round(2)} | vtx RMS {rms:.1f}mm | height SMPL {h_s*1000:.0f} vs SMPL-X {h_x*1000:.0f}mm")
    print("TRANSFER_DONE")


if __name__ == "__main__":
    main()
