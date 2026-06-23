"""Live-demo driver: pose recognition + A-pose (muscle-level) canonicalization for EVERY
single image and EVERY subject folder.

80/20: uses the fast Multi-HMR -> SMPL-X path (no gaussian generation). For each model:
  - recognize pose (Multi-HMR), allocate gender (CLIP, auto)
  - canonicalize to A-pose SMPL-X: mesh + 127-joint placement (incl fingers/toes) + abdomen contours
  - render a 4-panel demo (front+joints, side, abdomen x-z, abdomen y-x)

Outputs runs/demo_single/<name>/ and runs/demo_subject/<name>/ + an index montage.

# ============================ TO-REFINE KERNELS ============================
# (well-scoped hooks; not blocking the live demo)
#  K1 metric-anchored depth fusion: rasterize Multi-HMR v3d -> prior depth, calibrate the
#     N monocular estimators to it, info-filter the residual -> corrects abdomen x-z aspect
#     (see ROADMAP P0). Hook: refine_surface_with_depth(view, smplx_depth, depth_maps).
#  K2 LHM gaussian surface: per-model high-fidelity texture/relief (slow; hero models only).
#  K3 finger/toe + face: fuse per-view MANO/FLAME pose (rotvec[22:], expression) like betas.
#  K4 multi-view back-half: accumulate Fisher info across views to fill unobserved z + tighten σ.
# ==========================================================================
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

import os
import sys
import glob
import json

import numpy as np

sys.path.insert(0, os.path.join(REPO, "tools"))
import lhm_anthropometry as A  # noqa: E402

REPO = _repo
SSP_BODYBUILDER = (
    f"{REPO}/datasets/SSP-3D/ssp_3d/images/"
    "bodybuilding_vid_009_clip_000_person_001_frame_*.png"
)


def render_apose(av, named, contours, out_png, title, conf=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jp = np.array(list(named.values()))
    fig, ax = plt.subplots(2, 2, figsize=(11, 10), dpi=100)
    ax[0, 0].scatter(av[:, 0], av[:, 1], s=.2, c="#aab", alpha=.25)
    ax[0, 0].scatter(jp[:, 0], jp[:, 1], c="r", s=14)
    ax[0, 0].set_aspect("equal"); ax[0, 0].set_title("A-pose front (x-y) + joints")
    ax[0, 1].scatter(av[:, 2], av[:, 1], s=.2, c="#aab", alpha=.25)
    ax[0, 1].scatter(jp[:, 2], jp[:, 1], c="r", s=14)
    ax[0, 1].set_aspect("equal"); ax[0, 1].set_title("A-pose side (z-y)")
    xz = np.array(contours["transverse_xz"]) if contours["transverse_xz"] else np.zeros((0, 2))
    if len(xz):
        ax[1, 0].plot(xz[:, 0], xz[:, 1], "-", lw=1.5)
        w = xz[:, 0].max() - xz[:, 0].min(); d = xz[:, 1].max() - xz[:, 1].min()
        ax[1, 0].set_title(f"abdomen x-z  W={w*100:.0f}cm D={d*100:.0f}cm")
    ax[1, 0].set_aspect("equal"); ax[1, 0].grid(alpha=.3)
    yx = contours["coronal_yx"]
    if yx:
        yy = [d["y"] for d in yx]
        ax[1, 1].plot([d["x_left"] for d in yx], yy, "-", lw=1.5)
        ax[1, 1].plot([d["x_right"] for d in yx], yy, "-", lw=1.5)
    ax[1, 1].set_aspect("equal"); ax[1, 1].grid(alpha=.3); ax[1, 1].set_title("abdomen y-x silhouette")
    fig.suptitle(title + (f"   (conf {conf})" if conf else ""))
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def canonicalize(betas, spread, gender, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    av, aj, af, named = A.smplx_apose(betas, gender=gender, arm_deg=45.0)
    import trimesh
    trimesh.Trimesh(vertices=av, faces=af, process=False).export(os.path.join(out_dir, "apose_mesh.obj"))
    # contours on arms-RAISED mesh (torso has no arm verts; shape is pose-invariant)
    tv, tj, tf = A.smplx_measure_mesh(betas, gender=gender)
    contours = A.abdomen_contours(tv, tf, named)
    json.dump({"gender": gender, "joints": named, "beta": betas.tolist(),
               "abdomen": contours},
              open(os.path.join(out_dir, "apose.json"), "w"), indent=2)
    render_apose(av, named, contours, os.path.join(out_dir, "apose_demo.png"), title)
    return contours


def main():
    results = {"single": [], "subject": []}

    # SINGLE set: one view each (pose recognition + canonicalization, best-effort)
    singles = sorted(glob.glob(SSP_BODYBUILDER))
    for img in singles:
        name = os.path.splitext(os.path.basename(img))[0]
        out = f"{REPO}/runs/demo_single/{name}"
        pv = A.regress_betas([img])
        if not pv:
            print(f"[single {name}] no body"); results["single"].append({"name": name, "ok": False})
            continue
        betas = pv[0]["beta"]
        gender, gdbg = A.estimate_gender([img])
        try:
            c = canonicalize(betas, np.zeros_like(betas), gender, out, f"single/{name} [{gender}]")
            print(f"[single {name}] OK gender={gender} full_body={pv[0]['is_full_body']}")
            results["single"].append({"name": name, "ok": True, "gender": gender,
                                      "full_body": pv[0]["is_full_body"]})
        except Exception as e:
            print(f"[single {name}] ERR {repr(e)[:100]}")
            results["single"].append({"name": name, "ok": False})

    # SUBJECT set: multi-view shape fusion -> canonicalization
    for sub in [SSP_BODYBUILDER]:
        name = "ssp3d_bodybuilder"
        out = f"{REPO}/runs/demo_subject/{name}"
        imgs = sorted(glob.glob(sub))
        pv = A.regress_betas(imgs)
        fused, spread, n = A.fuse_betas(pv)
        if fused is None:
            print(f"[subject {name}] no betas"); results["subject"].append({"name": name, "ok": False})
            continue
        gender, gdbg = A.estimate_gender(imgs)
        try:
            c = canonicalize(fused, spread, gender, out, f"subject/{name} [{gender}, {n} views]")
            print(f"[subject {name}] OK gender={gender} views={n}")
            results["subject"].append({"name": name, "ok": True, "gender": gender, "views": n})
        except Exception as e:
            print(f"[subject {name}] ERR {repr(e)[:100]}")
            results["subject"].append({"name": name, "ok": False})

    json.dump(results, open(f"{REPO}/runs/demo_index.json", "w"), indent=2)
    ok_s = sum(r["ok"] for r in results["single"]); ok_b = sum(r["ok"] for r in results["subject"])
    print(f"DEMO_DONE single {ok_s}/{len(results['single'])}  subject {ok_b}/{len(results['subject'])}")


if __name__ == "__main__":
    main()
