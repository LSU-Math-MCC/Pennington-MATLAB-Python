"""Fit SMPL-X betas to the observed person SILHOUETTE so the mesh takes the person's REAL
body contours (Multi-HMR betas alone are generic; this conforms them to the image).

Method (robust, no differentiable renderer): pose SMPL-X with Multi-HMR's detected pose,
project to the image with the fitted pinhole, and optimize the 10 betas (scipy least_squares)
so the mesh's per-height silhouette WIDTH PROFILE matches the mask's, scale-invariantly
(width/height vs normalized height). Multi-view: average residuals across views.

Outputs: fitted_betas.npy, silhouette_overlay.png (mask edge vs fitted mesh edge), report.

Usage (WSL lhm): python tools/geometry/fit_silhouette.py --subject <dir|img|glob> --out <dir> [--gender female]
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
import argparse

import numpy as np

REPO = _repo
sys.path.insert(0, REPO + "/tools")
import lhm_anthropometry as A      # noqa: E402
import texture_uv_bake as TB       # noqa: E402
from PIL import Image              # noqa: E402

NB = 10
NBINS = 28


def image_paths(subject):
    if any(ch in subject for ch in "*?[]"):
        return sorted(glob.glob(subject))
    if os.path.isdir(subject):
        return sorted(sum([glob.glob(os.path.join(subject, "**", e), recursive=True)
                           for e in ("*.jpg", "*.jpeg", "*.png", "*.webp")], []))
    return [subject]


def person_mask(img):
    """Largest-person mask via YOLOv8-seg, hole-filled."""
    from ultralytics import YOLO
    import cv2
    if not hasattr(person_mask, "m"):
        person_mask.m = YOLO("yolov8x-seg.pt")
    r = person_mask.m(img[:, :, ::-1], verbose=False)[0]
    if r.masks is None:
        return None
    H, W = img.shape[:2]
    best = None; ba = 0
    cls = r.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(r.masks.data)):
        if cls[i] != 0:
            continue
        m = cv2.resize(r.masks.data[i].cpu().numpy(), (W, H)) > 0.5
        from scipy.ndimage import binary_fill_holes
        m = binary_fill_holes(m)
        if m.sum() > ba:
            ba = m.sum(); best = m
    if best is not None:
        # ERODE to strip the background halo (loose YOLO edge) -> the silhouette is the
        # BODY, not body+background; a fat mask was widening the fit (heavier shape).
        best = cv2.erode(best.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2).astype(bool)
    return best


def width_profile(ys, xs_lo, xs_hi, y0, y1, nbins=NBINS):
    """Per normalized-height bin, the width (hi-lo). Returns width array, length nbins."""
    out = np.zeros(nbins)
    if y1 <= y0:
        return out
    t = (ys - y0) / (y1 - y0)
    for k in range(nbins):
        m = (t >= k / nbins) & (t < (k + 1) / nbins)
        if m.any():
            out[k] = np.percentile(xs_hi[m], 95) - np.percentile(xs_lo[m], 5)
    return out


def clothing_weight(img, mask, y0, y1, nbins=NBINS, low=0.25):
    """Per-row weight in [low,1]: rows dominated by GARMENT (high color saturation inside the
    body mask) are down-weighted (clothing inflates the silhouette beyond the true body, so
    those rows are unreliable for tightening shape). Build-on-SHAPY: clothing-robust shape."""
    import cv2
    from scipy.ndimage import gaussian_filter1d
    m = np.asarray(mask, bool)
    rgb = img.astype(np.float32)
    mx = rgb.max(2); mn = rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    w = np.ones(nbins)
    ys = np.arange(img.shape[0])
    for k in range(nbins):
        lo = y0 + (y1 - y0) * k / nbins; hi = y0 + (y1 - y0) * (k + 1) / nbins
        rows = (ys >= lo) & (ys < hi)
        sub = m[rows]
        if sub.sum() < 5:
            continue
        frac_garment = float((sat[rows][sub] > 0.4).mean())   # colored garment fraction
        w[k] = 1.0 - (1.0 - low) * min(frac_garment * 1.5, 1.0)
    return gaussian_filter1d(w, 1.0)


def mask_width_profile(mask, nbins=NBINS):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros(nbins), 0, 0, 1
    y0, y1 = ys.min(), ys.max()
    height = y1 - y0
    # per row, left/right edge
    lo = np.full_like(ys, 0, float); hi = np.full_like(ys, 0, float)
    # build per-bin width using all mask pixels
    t = (ys - y0) / max(y1 - y0, 1)
    w = np.zeros(nbins)
    for k in range(nbins):
        m = (t >= k / nbins) & (t < (k + 1) / nbins)
        if m.any():
            w[k] = xs[m].max() - xs[m].min()
    return w, y0, y1, height


_SMODELS = {}


def _smodel(gender):
    if gender not in _SMODELS:
        import smplx
        _SMODELS[gender] = smplx.create(A.HUMAN_MODELS, model_type="smplx", gender=gender,
                                        num_betas=NB, use_pca=False, flat_hand_mean=True)
    return _SMODELS[gender]


def _forward_verts(gender, betas, gpose):
    import torch
    model = _smodel(gender)
    go, bp = gpose
    with torch.no_grad():
        out = model(betas=torch.tensor(betas[:NB], dtype=torch.float32).unsqueeze(0),
                    global_orient=torch.tensor(go, dtype=torch.float32).unsqueeze(0),
                    body_pose=torch.tensor(bp, dtype=torch.float32).unsqueeze(0))
    return out.vertices[0].detach().cpu().numpy()


def mesh_silhouette_profile(gpose, betas, gender, cam, y0, y1, offset, nbins=NBINS):
    """Forward SMPL-X(betas, detected pose) + camera-frame translation offset, project,
    return ABSOLUTE pixel width profile over rows [y0,y1] (betas -> pixel width -> gradient)."""
    v = _forward_verts(gender, betas, gpose) + offset    # place in camera frame (z ~ metres)
    fx, fy, cx, cy = cam
    Z = np.clip(v[:, 2], 1e-3, None)
    u = fx * v[:, 0] / Z + cx; vy = fy * v[:, 1] / Z + cy
    w = np.zeros(nbins)
    for k in range(nbins):
        lo = y0 + (y1 - y0) * k / nbins; hi = y0 + (y1 - y0) * (k + 1) / nbins
        m = (vy >= lo) & (vy < hi)
        if m.sum() > 2:
            w[k] = np.percentile(u[m], 97) - np.percentile(u[m], 3)
    mesh_full_h = float(vy.max() - vy.min())     # for scale-invariant slimness (width/height)
    return w, mesh_full_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gender", default=None)
    ap.add_argument("--clothing-weight", type=float, default=0.25,
                    help="low-prior weight for garment-dominated rows (0=ignore clothes, 1=trust)")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    import cv2
    from scipy.optimize import least_squares

    imgs = image_paths(args.subject)
    if not imgs:
        raise SystemExit(f"no images matched --subject {args.subject}")
    est = A._estimator()

    views = []
    for ip in imgs:
        img, h, j2d = TB.posed_view(est, ip)
        if h is None:
            continue
        # SAM2 box-prompt matte (clean body silhouette) — the YOLO mask leaks onto text/bg and
        # ruins the abdomen fit. Fall back to local YOLO person_mask if SAM2 unavailable.
        try:
            mask = TB.person_mask(img)
        except Exception:
            mask = person_mask(img)
        if mask is None or mask.sum() < 500:
            continue
        fx, fy, cx, cy = TB.fit_pinhole(h["j3d"], j2d)
        rot = h["rotvec"]
        go = rot[0:1].reshape(-1).astype(np.float32)
        bp = rot[1:22].reshape(-1).astype(np.float32)
        ow, y0, y1, oh = mask_width_profile(mask)    # absolute pixel widths over [y0,y1]
        b0 = h["shape"].reshape(-1)[:NB]
        # align my SMPL-X forward to Multi-HMR's camera-frame verts via a translation offset
        offset = h["v3d"].mean(0) - _forward_verts("neutral", b0, (go, bp)).mean(0)
        clothw = clothing_weight(img, mask, y0, y1, low=args.clothing_weight)
        views.append(dict(img=img, mask=mask, cam=(fx, fy, cx, cy), gpose=(go, bp),
                          own=ow, y0=y0, y1=y1, oh=oh, offset=offset, clothw=clothw,
                          betas0=b0, path=ip))
        if len(views) >= 4:
            break
    if not views:
        print("FIT_FAIL no views"); return

    gender = args.gender
    if gender is None:
        gender, _ = A.estimate_gender([v["path"] for v in views])
    beta0 = np.mean([v["betas0"] for v in views], 0)

    from scipy.ndimage import gaussian_filter1d

    # SCALE-INVARIANT slimness: compare width/height (not pixels) so girth is fit independent
    # of depth/scale; clothing rows down-weighted (garment inflates silhouette beyond body).
    def residual(betas):
        res = []
        for v in views:
            mw, mh = mesh_silhouette_profile(v["gpose"], betas, gender, v["cam"],
                                             v["y0"], v["y1"], v["offset"])
            mwn = mw / (mh + 1e-6)
            own = v["own"] / (v["oh"] + 1e-6)
            res.append(gaussian_filter1d((mwn - own) * v["clothw"], 1.0))
        res.append(0.15 * (betas - beta0))           # gentle prior anchor
        return np.concatenate(res)

    r0 = float(np.mean(np.abs(residual(beta0))))
    # diff_step=0.08: betas must perturb enough to move the (smoothed) silhouette profile,
    # else finite-diff gradients are ~0 and the optimizer stalls (the 0% bug).
    sol = least_squares(residual, beta0, method="trf", max_nfev=200, diff_step=0.08,
                        bounds=(-4 * np.ones(NB), 4 * np.ones(NB)))
    r1 = float(np.mean(np.abs(residual(sol.x))))
    np.save(os.path.join(args.out, "fitted_betas.npy"), sol.x)
    np.save(os.path.join(args.out, "prior_betas.npy"), beta0)   # Multi-HMR prior (for fusion)

    # overlay: mask edge vs fitted-mesh silhouette edge on view 0
    v0 = views[0]
    import smplx, torch
    model = smplx.create(A.HUMAN_MODELS, model_type="smplx", gender=gender, num_betas=NB,
                         use_pca=False, flat_hand_mean=True)
    go, bp = v0["gpose"]
    out = model(betas=torch.tensor(sol.x[:NB], dtype=torch.float32).unsqueeze(0),
                global_orient=torch.tensor(go, dtype=torch.float32).unsqueeze(0),
                body_pose=torch.tensor(bp, dtype=torch.float32).unsqueeze(0))
    vv = out.vertices[0].detach().cpu().numpy() + v0["offset"]
    fx, fy, cx, cy = v0["cam"]; Z = np.clip(vv[:, 2], 1e-3, None)
    uu = (fx * vv[:, 0] / Z + cx).astype(int); vy = (fy * vv[:, 1] / Z + cy).astype(int)
    vis = v0["img"].copy()
    Hh, Ww = vis.shape[:2]
    cont, _ = cv2.findContours(v0["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cont, -1, (60, 255, 60), 2)        # observed mask edge (green)
    ok = (uu >= 0) & (uu < Ww) & (vy >= 0) & (vy < Hh)
    vis[np.clip(vy[ok], 0, Hh - 1), np.clip(uu[ok], 0, Ww - 1)] = [255, 80, 80]  # mesh verts (red)
    Image.fromarray(vis).save(os.path.join(args.out, "silhouette_overlay.png"))

    import json
    json.dump({"gender": gender, "n_views": len(views),
               "mean_abs_residual_before": round(r0, 4), "after": round(r1, 4),
               "improvement_pct": round(100 * (r0 - r1) / (r0 + 1e-9), 1)},
              open(os.path.join(args.out, "fit_report.json"), "w"), indent=2)
    print(f"FIT_OK before={r0:.4f} after={r1:.4f} improve={100*(r0-r1)/(r0+1e-9):.0f}% gender={gender}")


if __name__ == "__main__":
    main()
