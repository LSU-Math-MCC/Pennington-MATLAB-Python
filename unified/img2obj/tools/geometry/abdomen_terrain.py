"""Dense MLE TERRAIN MAP of the abdomen (not a single contour).

Information-filter fusion of N monocular depth estimators, metrically ANCHORED by the
SMPL-X prior depth (kills the shared monocular z-scale bias). Output is a 2D height field
z(x,y) over the torso with per-pixel posterior sigma -- a terrain of every stomach feature
(navel, linea, muscle relief, indents), plus the relief (high-pass) and the sigma map.

  prior: rasterize Multi-HMR posed v3d -> dense metric depth over the torso ROI.
  z_nm = a_n*depth_n + b_n + e, e~N(0,sigma_nm) ; calibrate (a_n,b_n) to prior.
  fused z_hat = info-weighted mean (+ prior as a measurement); var = 1/info (posterior).

Usage (WSL lhm): python tools/geometry/abdomen_terrain.py --image <frontal> --out <dir>
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
import argparse

import numpy as np

REPO = _repo
sys.path.insert(0, REPO + "/tools")
sys.path.insert(0, REPO + "/src")
import lhm_anthropometry as A           # noqa: E402
import texture_uv_bake as TB            # noqa: E402
from pipeline.geometry.depth_fusion import robust_affine, per_pixel_sigma, information_fuse  # noqa: E402
from pipeline.backends import real      # noqa: E402

# SMPL-X joint indices (j3d/j2d order)
JIDX = {"l_sh": 16, "r_sh": 17, "l_hip": 1, "r_hip": 2, "spine1": 3, "spine3": 9, "pelvis": 0}


def _frankot_chellappa(p, q):
    """Integrate a gradient field (p=dz/dx, q=dz/dy) to a height field z (FFT, least-sq)."""
    H, W = p.shape
    wx = np.fft.fftfreq(W).reshape(1, W) * 2 * np.pi
    wy = np.fft.fftfreq(H).reshape(H, 1) * 2 * np.pi
    P = np.fft.fft2(p); Q = np.fft.fft2(q)
    denom = wx ** 2 + wy ** 2; denom[0, 0] = 1.0
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0
    return np.real(np.fft.ifft2(Z))


def torso_roi(j2d, H, W):
    sx = [j2d[JIDX[k], 0] for k in ("l_sh", "r_sh", "l_hip", "r_hip")]
    sy_top = min(j2d[JIDX["l_sh"], 1], j2d[JIDX["r_sh"], 1])
    sy_bot = max(j2d[JIDX["l_hip"], 1], j2d[JIDX["r_hip"], 1])
    x0 = int(max(0, min(sx) - 0.10 * (max(sx) - min(sx))))
    x1 = int(min(W, max(sx) + 0.10 * (max(sx) - min(sx))))
    y0 = int(max(0, sy_top)); y1 = int(min(H, sy_bot))
    return x0, y0, x1, y1


def prior_depth(v3d, fx, fy, cx, cy, x0, y0, x1, y1):
    """Dense metric prior depth over the ROI by interpolating projected vertex depths."""
    from scipy.interpolate import griddata
    Z = np.clip(v3d[:, 2], 1e-3, None)
    u = fx * v3d[:, 0] / Z + cx; v = fy * v3d[:, 1] / Z + cy
    sel = (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
    gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    grid = griddata(np.stack([u[sel], v[sel]], 1), v3d[sel, 2], (gx, gy), method="linear")
    grid_n = griddata(np.stack([u[sel], v[sel]], 1), v3d[sel, 2], (gx, gy), method="nearest")
    grid = np.where(np.isfinite(grid), grid, grid_n)
    return grid                                          # (h,w) metric depth, +Z away


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="small,base,large")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    est = A._estimator()
    img, h, j2d = TB.posed_view(est, args.image)
    if h is None:
        print("TERRAIN_FAIL no person"); return
    v3d = h["v3d"]; H, W = img.shape[:2]
    fx, fy, cx, cy = TB.fit_pinhole(h["j3d"], j2d)
    x0, y0, x1, y1 = torso_roi(j2d, H, W)
    print(f"ROI {x0},{y0}-{x1},{y1}  cam fx={fx:.0f}")

    prior = prior_depth(v3d, fx, fy, cx, cy, x0, y0, x1, y1)     # (h,w)
    roi_mask = np.isfinite(prior)

    # N estimators over ROI, calibrate each to the metric prior
    sizes = args.sizes.split(",")
    aligned = []; params = []
    for s in sizes:
        d_full = real._Models.depth_map(img, size=s).astype(np.float64)
        d = d_full[y0:y1, x0:x1]
        a, b, sg = robust_affine(d, prior, roi_mask)
        aligned.append(a * d + b); params.append((a, b, sg))
        print(f"calib[{s}] a={a:.3f} b={b:.3f} sigma={sg*100:.2f}cm")

    ens_std = np.nanstd(np.stack(aligned, 0), axis=0)
    sigmas = [per_pixel_sigma(al, prior, roi_mask, p[2], ens_std) for al, p in zip(aligned, params)]
    # prior as an anchoring measurement (sigma ~2cm); residual recovers real relief
    z_hat, var, info = information_fuse(aligned, sigmas, prior=prior, prior_sigma=0.02)
    z_hat = np.where(roi_mask, z_hat, np.nan)
    sigma = np.sqrt(var)

    # FINE relief from SHADING NORMALS (navel/linea/muscle live in shading, not in
    # cm-resolution monocular depth). Integrate normals via Frankot-Chellappa; the metric
    # fused depth only anchors absolute scale (low freq).
    from scipy.ndimage import gaussian_filter
    roi_rgb = img[y0:y1, x0:x1].astype(np.float32)
    L = roi_rgb.mean(2) / 255.0
    L = gaussian_filter(L, 1.0)
    # suppress ALBEDO edges (garment): shading-from-shading assumes uniform skin albedo, so
    # mask high-saturation (colored garment) pixels -> their gradients don't create false relief
    mx = roi_rgb.max(2); mn = roi_rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    skin = gaussian_filter((sat < 0.35).astype(np.float32), 2.0)   # 1=skin, 0=garment
    gy, gx = np.gradient(L)
    gx *= skin; gy *= skin                               # zero garment-edge gradients
    strength = 6.0                                       # normal slope per luminance unit
    p = -gx * strength; q = -gy * strength               # dz/dx, dz/dy from shading
    relief_shade = _frankot_chellappa(p, q)              # integrated height (rel units)
    relief_shade -= gaussian_filter(relief_shade, 20)    # remove integration drift
    # normalize to PHYSICAL scale: 1 robust-std -> ~1cm, clamp to +-4cm (no drift blowup)
    inb = relief_shade[roi_mask]
    rstd = np.median(np.abs(inb - np.median(inb))) * 1.4826 + 1e-9
    relief = (relief_shade - np.median(inb)) / rstd * 0.01      # metres
    relief = np.clip(relief, -0.04, 0.04)
    relief = np.where(roi_mask, relief, np.nan)
    np.savez(os.path.join(args.out, "terrain.npz"), depth=z_hat, relief=relief,
             sigma=sigma, roi=[x0, y0, x1, y1])

    # closer surface = smaller z; show -relief so bumps point up
    R = -np.nan_to_num(relief, nan=0.0) * 100.0          # cm, +up = protruding
    fig = plt.figure(figsize=(18, 5.5), dpi=110)
    ax0 = fig.add_subplot(1, 4, 1); ax0.imshow(img[y0:y1, x0:x1]); ax0.set_title("abdomen ROI"); ax0.axis("off")
    ax1 = fig.add_subplot(1, 4, 2); im1 = ax1.imshow(R, cmap="terrain"); ax1.set_title("MLE relief terrain (cm, +out)"); ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    ax2 = fig.add_subplot(1, 4, 3); im2 = ax2.imshow(sigma * 100, cmap="magma"); ax2.set_title("posterior σ (cm)"); ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    ax3 = fig.add_subplot(1, 4, 4, projection="3d")
    hh, ww = R.shape
    step = max(1, min(hh, ww) // 120)
    gy, gx = np.mgrid[0:hh:step, 0:ww:step]
    ax3.plot_surface(gx, gy, R[::step, ::step], cmap="terrain", linewidth=0, antialiased=True)
    ax3.set_title("3D abdomen terrain"); ax3.view_init(elev=55, azim=-60)
    fig.suptitle(f"Abdomen MLE terrain — {len(sizes)} estimators + SMPL-X anchor — "
                 f"median σ={np.nanmedian(sigma)*100:.2f}cm")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "terrain.png")); plt.close(fig)
    print(f"TERRAIN_OK median_sigma_cm={np.nanmedian(sigma)*100:.2f} "
          f"relief_range_cm={np.nanmax(R)-np.nanmin(R):.1f}")


if __name__ == "__main__":
    main()
