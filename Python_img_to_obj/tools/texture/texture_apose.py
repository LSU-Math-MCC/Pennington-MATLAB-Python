"""Map observed image pixels onto the canonical A-pose SMPL-X surface (per-vertex bake).

Because SMPL-X topology is FIXED, a color sampled at posed vertex v transfers to the SAME
vertex v in the A-pose -> correct L/R, fingers, toes, face by construction (no Poisson, no
tumors). Multi-view accumulates: each vertex's color is a visibility/quality-weighted mean
over all views in which it is front-facing (information filter over texels; nothing wasted).

Per view:
  1. Multi-HMR inner call -> posed metric v3d, j3d (cam frame), j2d (-> original image px).
  2. Fit pinhole (fx,fy,cx,cy) from j3d->j2d (absorbs all preprocessing); project v3d.
  3. visibility weight = max(0, -n·v_dir) (front-facing) * in-image; sample bilinear color.
  4. accumulate color_sum += w*c, w_sum += w   per vertex.
Then color the A-pose mesh (same vertices) and export colored OBJ/GLB + render.

# TO-REFINE: z-buffer occlusion (pytorch3d rasterize) so back-facing-through verts don't
# bleed; per-texel UV atlas (not per-vertex) for full pixel resolution; normal/displacement
# bake; Sapiens-normal micro-relief; metric-anchored depth displacement on torso.

Usage (WSL lhm): python tools/texture/texture_apose.py --subject <dir|image> --out <dir>
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
from PIL import Image


def posed_view(est, img_path):
    """Replicate PoseEstimator.__call__ internals -> (img_np_orig, human dict, j2d_orig)."""
    img_np = np.asarray(Image.open(img_path).convert("RGB"))
    raw_h, raw_w, _ = img_np.shape
    padded, ow, oh = est.img_center_padding(img_np)
    t, ann = est._preprocess(padded)
    K = est.get_camera_parameters()
    th = est.mhmr_model(t, is_training=False, nms_kernel_size=3, det_thresh=0.3,
                        K=K, idx=None, max_dist=None)
    if len(th) != 1:
        return img_np, None, None
    h = th[0]
    pad_left, pad_top, scale_factor = ann[0], ann[1], ann[2]
    import torch
    j2d = h["j2d"].detach().cpu().numpy()
    j2d = (j2d - np.array([pad_left, pad_top])) / float(scale_factor) - np.array([ow, oh])
    return img_np, {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
                    for k, v in h.items()}, j2d


def fit_pinhole(j3d, j2d):
    """Least-squares fx,fy,cx,cy with u=fx*X/Z+cx, v=fy*Y/Z+cy."""
    X, Y, Z = j3d[:, 0], j3d[:, 1], j3d[:, 2]
    good = (Z > 1e-3) & np.isfinite(j2d).all(1)
    x, y, z = X[good], Y[good], Z[good]
    u, v = j2d[good, 0], j2d[good, 1]
    Au = np.stack([x / z, np.ones_like(x)], 1)
    fu, cu = np.linalg.lstsq(Au, u, rcond=None)[0]
    Av = np.stack([y / z, np.ones_like(y)], 1)
    fv, cv = np.linalg.lstsq(Av, v, rcond=None)[0]
    return float(fu), float(fv), float(cu), float(cv)


def sample_bilinear(img, uv):
    H, W = img.shape[:2]
    u = np.clip(uv[:, 0], 0, W - 1.001); v = np.clip(uv[:, 1], 0, H - 1.001)
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
    fx = (u - x0)[:, None]; fy = (v - y0)[:, None]
    c = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return c / 255.0


def main():
    import argparse, trimesh
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if os.path.isdir(args.subject):
        images = sorted(sum([glob.glob(os.path.join(args.subject, "**", e), recursive=True)
                             for e in ("*.jpg", "*.jpeg", "*.png", "*.webp")], []))
    else:
        images = [args.subject]
    print("views:", len(images))

    est = A._estimator()
    import smplx
    smodel = smplx.create(A.HUMAN_MODELS, model_type="smplx", gender="neutral",
                          num_betas=10, use_pca=False, flat_hand_mean=True)
    faces = smodel.faces.astype(np.int64)

    nV = 10475
    col_sum = np.zeros((nV, 3)); w_sum = np.zeros(nV)
    betas_list = []
    for img_path in images:
        img, h, j2d = posed_view(est, img_path)
        if h is None:
            print("  skip", os.path.basename(img_path)); continue
        v3d = h["v3d"]; j3d = h["j3d"]
        betas_list.append(h["shape"].reshape(-1))
        fx, fy, cx, cy = fit_pinhole(j3d, j2d)
        Z = np.clip(v3d[:, 2], 1e-3, None)
        uv = np.stack([fx * v3d[:, 0] / Z + cx, fy * v3d[:, 1] / Z + cy], 1)
        # vertex normals + front-facing weight
        m = trimesh.Trimesh(vertices=v3d, faces=faces, process=False)
        vn = np.asarray(m.vertex_normals)
        vdir = v3d / (np.linalg.norm(v3d, axis=1, keepdims=True) + 1e-9)
        facing = np.clip(-(vn * vdir).sum(1), 0, 1)         # 1 = squarely facing camera
        H, W = img.shape[:2]
        inimg = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        w = facing * inimg
        cols = sample_bilinear(img, uv)
        col_sum += (cols * w[:, None]); w_sum += w
        print(f"  {os.path.basename(img_path)} textured {int((w>0.05).sum())} verts "
              f"cam fx={fx:.0f}")

    seen = w_sum > 1e-3
    vcol = np.full((nV, 3), 0.6)
    vcol[seen] = col_sum[seen] / w_sum[seen, None]
    coverage = float(seen.mean())

    # color the A-POSE mesh (same topology -> correct L/R, digits, face)
    betas = np.mean(betas_list, 0) if betas_list else np.zeros(10)
    gender, _ = A.estimate_gender(images)
    av, aj, af, named = A.smplx_apose(betas, gender=gender, arm_deg=45.0)
    rgba = (np.clip(vcol, 0, 1) * 255).astype(np.uint8)
    amesh = trimesh.Trimesh(vertices=av, faces=af, vertex_colors=rgba, process=False)
    amesh.export(os.path.join(args.out, "apose_textured.glb"))
    amesh.export(os.path.join(args.out, "apose_textured.obj"))
    json.dump({"coverage": round(coverage, 4), "views": len(images), "gender": gender,
               "seen_verts": int(seen.sum())},
              open(os.path.join(args.out, "texture_report.json"), "w"), indent=2)
    print(f"TEXTURE_OK coverage={coverage:.1%} gender={gender}")


if __name__ == "__main__":
    main()
