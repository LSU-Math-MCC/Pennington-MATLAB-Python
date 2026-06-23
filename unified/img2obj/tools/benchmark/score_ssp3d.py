"""Score CameraHMR / SHAPY / BLADE shape accuracy on SSP-3D.

Primary metric (all three, topology-agnostic): mean per-vertex distance from the predicted body
SURFACE to the GT body surface (mm), in a shared canonical A-pose, after height-scale alignment
(removes the depth/scale ambiguity, like PVE-T-SC's scale correction). Reported as mean +/- std over
rows each method has betas for. Coverage is shown at the subject level because BLADE is often run
once per unique SSP-3D subject, while SSP-3D contains repeated frames for the same body shape.

Also reports CameraHMR's canonical PVE-T-SC (gendered SMPL T-pose, RMSD scale+trans align) so it
ties to the published SSP-3D leaderboard (CameraHMR 11.6; SHAPY 19.2; STRAPS 15.9; ...).

Run (camerahmr env):  python tools/benchmark/score_ssp3d.py
"""
import os, sys, json
import numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SMPL_DIR = os.path.expanduser("~/shapy/data/body_models/smpl")
SMPLX_DIR = os.path.expanduser("~/shapy/data/body_models/smplx")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO, "tools", "smplx"))
import backends  # noqa: E402
backends.reexec_in_wsl()  # on Windows: run in the WSL camerahmr env (SMPL models live there)
from ssp3d_subjects import subject_groups


def smpl_apose():
    bp = np.zeros((23, 3), np.float32)
    bp[15, 2] = -np.deg2rad(55); bp[16, 2] = np.deg2rad(55)
    return bp.reshape(-1)


def smplx_apose():
    bp = np.zeros((21, 3), np.float32)
    bp[15, 2] = -np.deg2rad(55); bp[16, 2] = np.deg2rad(55)
    return bp.reshape(-1)


def scale_trans_align(P, T):
    Pm = P.mean(1, keepdims=True); Pt = P - Pm
    Ps = np.sqrt((Pt ** 2).sum((1, 2), keepdims=True) / P.shape[1]); Pn = Pt / Ps
    Tm = T.mean(1, keepdims=True); Ts = np.sqrt(((T - Tm) ** 2).sum((1, 2), keepdims=True) / T.shape[1])
    return Pn * Ts + Tm


def main():
    import torch, smplx, trimesh
    D = np.load(os.path.join(REPO, "runs", "CAMERAHMR_ssp3d.npz"))
    gt, gen, fn, cam = D["shapes"], D["genders"], D["fnames"], D["betas"]
    groups = subject_groups(gt)
    src = {"CameraHMR": (cam, "smpl")}
    for name, f in [("SHAPY", "SHAPY_ssp3d.npz"), ("BLADE", "BLADE_ssp3d_crop.npz")]:
        p = os.path.join(REPO, "runs", f)
        if name == "BLADE" and not os.path.exists(p):
            p = os.path.join(REPO, "runs", "BLADE_ssp3d.npz")
        if os.path.exists(p):
            src[name] = (np.load(p)["betas"], "smplx")

    smpl = {"m": smplx.SMPL(SMPL_DIR, gender="male"), "f": smplx.SMPL(SMPL_DIR, gender="female")}
    smplx_m = smplx.SMPLX(SMPLX_DIR, gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=10)
    bps, bpx = torch.tensor(smpl_apose())[None], torch.tensor(smplx_apose())[None]

    def smpl_v(b, g):
        with torch.no_grad():
            o = smpl[g](betas=torch.tensor(b[:10]).float()[None], body_pose=bps)
        return o.vertices[0].numpy() - o.joints[0, 0].numpy()

    def smplx_v(b):
        with torch.no_grad():
            o = smplx_m(betas=torch.tensor(b[:10]).float()[None], body_pose=bpx)
        return o.vertices[0].numpy() - o.joints[0, 0].numpy()

    def surf_err(pv, gv, gf):
        sh = gv[:, 1].ptp() / (pv[:, 1].ptp() + 1e-9)        # height-scale align (kill scale ambiguity)
        prox = trimesh.proximity.ProximityQuery(trimesh.Trimesh(gv, gf, process=False))
        return float(np.abs(prox.signed_distance(pv * sh)).mean() * 1000.0)

    results = {}
    for name, (betas, mtype) in src.items():
        errs = []
        covered_rows = []
        for i in range(len(fn)):
            if np.any(np.isnan(betas[i][:10])):
                continue
            g = str(gen[i]); gv = smpl_v(gt[i], g); gf = smpl[g].faces
            pv = smpl_v(betas[i], g) if mtype == "smpl" else smplx_v(betas[i])
            errs.append(surf_err(pv, gv, gf))
            covered_rows.append(i)
        errs = np.array(errs)
        covered = set(covered_rows)
        covered_subjects = sum(1 for mem in groups if any(i in covered for i in mem))
        results[name] = dict(surface_err_mm=round(float(errs.mean()), 2),
                             std=round(float(errs.std()), 2), n_frames=int(len(errs)),
                             row_coverage=f"{len(errs)}/{len(fn)}",
                             n_subjects=int(covered_subjects),
                             subject_coverage=f"{covered_subjects}/{len(groups)}",
                             coverage=f"{covered_subjects}/{len(groups)} subj")

    # canonical PVE-T-SC for CameraHMR (SMPL betas -> SMPL T-pose, scale+trans aligned)
    pves = []
    for i in range(len(fn)):
        g = str(gen[i])
        with torch.no_grad():
            pv = smpl[g](betas=torch.tensor(cam[i][:10]).float()[None]).vertices.numpy()
            gv = smpl[g](betas=torch.tensor(gt[i][:10]).float()[None]).vertices.numpy()
        pves.append(np.linalg.norm(scale_trans_align(pv, gv) - gv, axis=-1).mean() * 1000)
    results["CameraHMR"]["PVE_T_SC_mm"] = round(float(np.mean(pves)), 2)

    print("\n=== SSP-3D shape score: vertex-to-GT-surface (mm, height-scale aligned, A-pose) ===")
    for n, r in results.items():
        extra = f" | PVE-T-SC {r['PVE_T_SC_mm']}mm" if "PVE_T_SC_mm" in r else ""
        print(f"  {n:10s}  {r['surface_err_mm']:5.2f} +/- {r['std']:.2f} mm   "
              f"(subjects={r['subject_coverage']}, rows={r['row_coverage']}){extra}")
    print("  published PVE-T-SC: CameraHMR 11.6 | SHAPY 19.2 | STRAPS 15.9 | Sengupta 13.6")
    json.dump(results, open(os.path.join(REPO, "runs", "SSP3D_SCORES.json"), "w"), indent=2)
    print("saved runs/SSP3D_SCORES.json")


if __name__ == "__main__":
    main()
