"""Recreate SHAPY's teaser-style body-SHAPE figure.

Per subject, a row: [ input image | Ground truth | <method> ... ] where each predicted mesh is
colored by its PER-VERTEX deviation from the GT body SURFACE (mm), in a shared canonical A-pose
(so only shape differs). 0-50 mm colorbar, as in Choutas et al., SHAPY (CVPR'22).

Why point-to-surface (not per-vertex beta diff): methods use different body models -- CameraHMR is
SMPL (6890 v), SHAPY is SMPL-X (10475 v) -- so we color each predicted vertex by its distance to
the GT (gendered SMPL) surface. Topology-agnostic and directly comparable in mm. Every vertex
(incl. hands/face) is coloured; errors above the 50 mm cap clamp to the top colour (no grey).
Subjects are chosen by farthest-point sampling in GT-shape space for body-type diversity.

Data: SSP-3D GT betas + per-method betas. CameraHMR <- runs/CAMERAHMR_ssp3d.npz; SHAPY (SMPL-X) <-
runs/SHAPY_ssp3d.npz (from tools/hmr/shapy/run_shapy_ssp3d.py).

Run (camerahmr env):
  python tools/hmr/shapy/plots.py --methods SHAPY CameraHMR --n 4 --out runs/SHAPY_TEASER.png
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

import os, sys, argparse
import numpy as np
import backends  # noqa: E402  (sibling tools/smplx, on path via bootstrap)
backends.reexec_in_wsl()  # on Windows: run this whole script in the WSL camerahmr env
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SMPL_DIR = os.path.expanduser("~/shapy/data/body_models/smpl")
SMPLX_DIR = os.path.expanduser("~/shapy/data/body_models/smplx")
SSP = os.path.expanduser("~/SSP-3D/data_ext/ssp_3d")
ERR_MAX = 50.0


def smpl_apose():
    bp = np.zeros((23, 3), np.float32)
    bp[15, 2] = -np.deg2rad(55); bp[16, 2] = np.deg2rad(55); bp[15, 1] = -np.deg2rad(8); bp[16, 1] = np.deg2rad(8)
    return bp.reshape(-1)


def smplx_apose():
    bp = np.zeros((21, 3), np.float32)
    bp[15, 2] = -np.deg2rad(55); bp[16, 2] = np.deg2rad(55); bp[15, 1] = -np.deg2rad(8); bp[16, 1] = np.deg2rad(8)
    return bp.reshape(-1)


def render_mesh(verts, faces, vert_colors, w=300, h=540):
    import pyrender, trimesh
    m = trimesh.Trimesh(verts, faces, vertex_colors=vert_colors, process=False)
    sc = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[.45, .45, .45])
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=True))
    c = verts.mean(0); ext = np.ptp(verts, 0).max()
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.4)
    p = np.eye(4); p[0, 3] = c[0]; p[1, 3] = c[1]; p[2, 3] = c[2] + ext * 1.5
    sc.add(cam, pose=p)
    for dx, inten in [(0.6, 3.5), (-0.7, 2.0)]:
        lp = np.eye(4); lp[0, 3] = c[0] + dx * ext; lp[1, 3] = c[1] + 0.3 * ext; lp[2, 3] = c[2] + ext * 1.5
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=inten), pose=lp)
    r = pyrender.OffscreenRenderer(w, h); col, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["SHAPY", "CameraHMR", "BLADE"])
    ap.add_argument("--shapy-npz", default=os.path.join(REPO, "runs", "SHAPY_ssp3d.npz"))
    ap.add_argument("--blade-npz", default=None,
                    help="BLADE betas npz; defaults to cropped cache when it covers enough rows")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--skinny", action="store_true", help="skinniest subjects instead of diverse")
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "SHAPY_TEASER.png"))
    args = ap.parse_args()

    import torch, smplx, trimesh, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from PIL import Image

    D = np.load(os.path.join(REPO, "runs", "CAMERAHMR_ssp3d.npz"))
    gt_all, gen, fn = D["shapes"], D["genders"], D["fnames"]
    # method -> (betas[311,10], model_type)
    src = {"CameraHMR": (D["betas"], "smpl")}
    if os.path.exists(args.shapy_npz):
        S = np.load(args.shapy_npz, allow_pickle=True)
        src["SHAPY"] = (S["betas"], "smplx")
        print(f"loaded SHAPY betas {S['betas'].shape}")
    blade_npz = args.blade_npz
    if blade_npz is None:
        crop_npz = os.path.join(REPO, "runs", "BLADE_ssp3d_crop.npz")
        base_npz = os.path.join(REPO, "runs", "BLADE_ssp3d.npz")
        blade_npz = base_npz
        if os.path.exists(crop_npz):
            Bc = np.load(crop_npz, allow_pickle=True)
            if int(np.sum(~np.isnan(Bc["betas"][:, 0]))) >= args.n:
                blade_npz = crop_npz
    if os.path.exists(blade_npz):
        Bz = np.load(blade_npz, allow_pickle=True)
        src["BLADE"] = (Bz["betas"], "smplx")
        print(f"loaded BLADE betas {Bz['betas'].shape} from {os.path.basename(blade_npz)}")
    methods = [m for m in args.methods if m in src]
    if not methods:
        sys.exit(f"none of {args.methods} available; have {list(src)}")

    smpl = {"m": smplx.SMPL(SMPL_DIR, gender="male"), "f": smplx.SMPL(SMPL_DIR, gender="female")}
    smplx_m = smplx.SMPLX(SMPLX_DIR, gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=10)
    bp_smpl = torch.tensor(smpl_apose()).float().unsqueeze(0)
    bp_smplx = torch.tensor(smplx_apose()).float().unsqueeze(0)

    def smpl_verts(betas, g):
        with torch.no_grad():
            o = smpl[g](betas=torch.tensor(betas[:10]).float().unsqueeze(0), body_pose=bp_smpl)
        return o.vertices[0].numpy() - o.joints[0, 0].numpy(), smpl[g].faces

    def smplx_verts(betas):
        with torch.no_grad():
            o = smplx_m(betas=torch.tensor(betas[:10]).float().unsqueeze(0), body_pose=bp_smplx)
        return o.vertices[0].numpy() - o.joints[0, 0].numpy(), smplx_m.faces.astype(np.int64)

    cmap = LinearSegmentedColormap.from_list(
        "meshmap_signed_rwb",
        [
            (0.00, "#1f4eaa"),
            (0.18, "#60a5fa"),
            (0.35, "#c7e9ff"),
            (0.50, "#ffffff"),
            (0.65, "#ffd6c2"),
            (0.82, "#f9735b"),
            (1.00, "#b2182b"),
        ],
    )
    norm = TwoSlopeNorm(vmin=-ERR_MAX, vcenter=0.0, vmax=ERR_MAX)
    # diverse subjects (farthest-point sampling) that every method has a beta for (no "n/a"),
    # then sort by overall size for a readable big->small column. select more than n so we can
    # drop any uncovered ones and still fill n.
    from ssp3d_subjects import select_diverse, select_skinny
    cand = (select_skinny(gt_all, max(args.n * 3, 18), SMPL_DIR) if args.skinny
            else select_diverse(gt_all, max(args.n * 3, 12)))
    full = [m for m in cand if all(not np.any(np.isnan(src[mth][0][m[0]][:10])) for mth in methods)]
    subj = sorted(full, key=lambda m: -np.linalg.norm(gt_all[m[0], :10]))[:args.n]
    if not subj:
        sys.exit("no subject has all methods covered")

    ncol = 2 + len(methods)
    headers = ["Input", "Ground truth"] + methods
    fig, axes = plt.subplots(len(subj), ncol, figsize=(2.3 * ncol, 4.1 * len(subj)))
    axes = np.atleast_2d(axes)
    for r, mem in enumerate(subj):
        i0 = mem[0]; g = str(gen[i0])
        gv, gfaces = smpl_verts(gt_all[i0], g)
        prox = trimesh.proximity.ProximityQuery(trimesh.Trimesh(gv, gfaces, process=False))
        ip = os.path.join(SSP, "images", str(fn[i0]))
        if os.path.exists(ip):
            axes[r, 0].imshow(Image.open(ip))
        axes[r, 0].axis("off")
        axes[r, 1].imshow(render_mesh(gv, gfaces, np.tile([180, 180, 185, 255], (gv.shape[0], 1)).astype(np.uint8)))
        axes[r, 1].axis("off")
        for c, mth in enumerate(methods):
            betas, mtype = src[mth][0][i0], src[mth][1]
            pv, pf = (smpl_verts(betas, g) if mtype == "smpl" else smplx_verts(betas))
            err = prox.signed_distance(pv) * 1000.0
            err = np.nan_to_num(err, nan=0.0, posinf=ERR_MAX, neginf=-ERR_MAX)
            cc = cmap(norm(np.clip(err, -ERR_MAX, ERR_MAX)))[:, :4]   # clamp extremes, no grey
            ax = axes[r, 2 + c]
            ax.imshow(render_mesh(pv, pf, (cc * 255).astype(np.uint8)))
            ax.axis("off")
            ax.text(0.5, 0.01, f"mean |err| {np.abs(err).mean():.1f} mm", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8, color=(0.15, 0.15, 0.15))
    for c, t in enumerate(headers):
        axes[0, c].set_title(t, fontsize=12, pad=6)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.012, shrink=0.96)
    cbar.set_ticks([-ERR_MAX, -ERR_MAX / 2, 0, ERR_MAX / 2, ERR_MAX])
    cbar.set_label("signed vertex-to-GT-surface (mm)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle("Body-shape accuracy vs SSP-3D ground truth (canonical A-pose)", fontsize=12)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}  ({len(subj)} subjects x {ncol} cols; methods={methods})")


if __name__ == "__main__":
    main()
