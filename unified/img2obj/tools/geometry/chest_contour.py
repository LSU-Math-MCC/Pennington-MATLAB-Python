"""Chest-contour reconstruction by information-filter fusion of N depth estimators.

Demonstrates the estimation framework on a real chest:
  1. N monocular depth estimators (Depth-Anything small/base/large) -> N depth maps.
  2. Person mask (YOLO) + chest band from 2D pose (shoulders..mid-torso).
  3. Robust affine-calibrate each estimator to the inverse-variance ensemble, build a
     per-pixel a-priori sigma (edge / boundary / cross-estimator-disagreement inflated).
  4. Information-filter MLE fuse -> fused depth + POSTERIOR SIGMA per pixel.
  5. Extract the chest cross-section at chest height: back-project the fused front-surface
     -> (lateral x, depth z) front contour, each point carrying its propagated sigma.

A single frontal view recovers the FRONT half-contour (the back is unobserved -> infinite
sigma there); adding views fills the rest, and the posterior sigma is the honest precision.
Metric scale is pinned by shoulder breadth (--shoulder-cm) or left relative.

Usage:
  python tools/geometry/chest_contour.py --image datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png --out runs/chest_ssp3d_bodybuilder [--shoulder-cm 39]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pipeline.config import Config
from pipeline import io as pio
from pipeline.backends import real
from pipeline.geometry import camera as camlib
from pipeline.geometry.depth_fusion import fuse_estimators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shoulder-cm", type=float, default=None,
                    help="metric calibration: known biacromial (shoulder) breadth in cm")
    ap.add_argument("--sizes", default="small,base,large")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    img = pio.load_image(args.image, max_edge=768)
    H, W = img.shape[:2]
    cam = camlib.default_camera(W, H)

    segs = real.RealSeg().segment_people(img, out)
    if not segs:
        print("CHEST_FAIL no person"); return
    seg = max(segs, key=lambda s: s.person_mask.sum())
    mask = seg.person_mask
    pose = real.RealPose().estimate_pose(img, out, bbox=seg.bbox)
    kp = pose.keypoints

    # N depth estimators
    sizes = args.sizes.split(",")
    depth_maps = []
    for s in sizes:
        z = real._Models.depth_map(img, size=s).astype(np.float64)
        depth_maps.append(z)
        print(f"depth[{s}] range {z.min():.2f}..{z.max():.2f}")

    fz = fuse_estimators(depth_maps, mask)
    fused, sigma = fz["depth"], fz["sigma"]
    np.savez(out / "fused_depth.npz", depth=fused, sigma=sigma, info=fz["info"])
    for p, s in zip(fz["params"], sizes):
        print(f"calib[{s}] a={p['a']:.3f} b={p['b']:.3f} sigma_global={p['sigma_global']:.4f}")

    # chest band: vertical between shoulders and mid (shoulder->hip midpoint)
    def K(n):
        return np.array(kp[n][:2]) if n in kp and kp[n][2] > 0.2 else None
    ls, rs = K("left_shoulder"), K("right_shoulder")
    lh, rh = K("left_hip"), K("right_hip")
    if ls is None or rs is None:
        print("CHEST_FAIL no shoulders"); return
    sh_y = 0.5 * (ls[1] + rs[1])
    hip_y = 0.5 * ((lh[1] if lh is not None else sh_y + 0.4 * H) +
                   (rh[1] if rh is not None else sh_y + 0.4 * H))
    chest_y = int(round(sh_y + 0.28 * (hip_y - sh_y)))     # ~chest height
    chest_y = int(np.clip(chest_y, 1, H - 2))

    # contour: at chest row band, for each in-mask column take median fused depth+sigma
    band = slice(max(0, chest_y - 2), min(H, chest_y + 3))
    cols = []
    for u in range(W):
        col_m = mask[band, u]
        if not col_m.any():
            continue
        d = fused[band, u][col_m]
        sg = sigma[band, u][col_m]
        good = np.isfinite(d)
        if not good.any():
            continue
        du = float(np.nanmedian(d[good]))
        su = float(np.nanmedian(sg[good]))
        cols.append((u, du, su))
    if len(cols) < 5:
        print("CHEST_FAIL contour too short"); return
    cols = np.array(cols)
    uvs = np.stack([cols[:, 0], np.full(len(cols), chest_y)], 1)
    pts_cam = camlib.backproject(cam, uvs, cols[:, 1])     # front surface 3D (relative)
    # propagate depth sigma to z (z ~ depth for near-axis); lateral x from u
    x = pts_cam[:, 0]; z = pts_cam[:, 2]; zs = cols[:, 2]

    # metric calibration from shoulder breadth
    scale = 1.0; unit = "relative"
    if args.shoulder_cm:
        # biacromial pixel width -> 3D width at shoulder depth
        sh_d = float(np.nanmedian(fused[max(0, int(sh_y) - 2):int(sh_y) + 3,
                                        int(min(ls[0], rs[0])):int(max(ls[0], rs[0])) + 1]))
        pa = camlib.backproject(cam, np.array([ls[:2], rs[:2]]), np.array([sh_d, sh_d]))
        w3d = np.linalg.norm(pa[0] - pa[1]) + 1e-9
        scale = (args.shoulder_cm / 100.0) / w3d
        unit = "m"
        x = x * scale; z = z * scale; zs = zs * scale

    contour = [{"x": round(float(xi), 5), "z": round(float(zi), 5),
                "sigma": round(float(si), 5)} for xi, zi, si in zip(x, z, zs)]
    depth_extent = float(np.nanmax(z) - np.nanmin(z))
    json.dump({"unit": unit, "scale_to_m": scale, "chest_y_px": chest_y,
               "n_points": len(contour),
               "front_depth_extent": round(depth_extent, 4),
               "median_sigma": round(float(np.nanmedian(zs)), 5),
               "n_estimators": len(sizes), "contour": contour},
              open(out / "chest_contour.json", "w"), indent=2)

    _plot(out, img, mask, chest_y, x, z, zs, fused, sigma, unit)
    print("CHEST_OK", json.dumps({"n_points": len(contour), "unit": unit,
                                  "front_depth_extent": round(depth_extent, 4),
                                  "median_sigma": round(float(np.nanmedian(zs)), 5)}))


def _plot(out, img, mask, chest_y, x, z, zs, fused, sigma, unit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=110)
    a = img.copy(); a[chest_y - 1:chest_y + 2, :] = [255, 60, 60]
    ax[0].imshow(a); ax[0].set_title("chest slice line"); ax[0].axis("off")
    # front contour with sigma envelope
    order = np.argsort(x)
    xo, zo, so = x[order], z[order], zs[order]
    ax[1].plot(xo, zo, "-o", ms=3, color="#1f77b4", label="front chest contour")
    ax[1].fill_between(xo, zo - so, zo + so, alpha=0.3, color="#1f77b4", label="±1σ posterior")
    ax[1].set_aspect("equal"); ax[1].invert_yaxis()
    ax[1].set_xlabel(f"lateral x ({unit})"); ax[1].set_ylabel(f"depth z ({unit})")
    ax[1].set_title("MLE-fused chest cross-section"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    sm = np.where(np.asarray(mask, bool), sigma, np.nan)
    im = ax[2].imshow(sm, cmap="magma"); ax[2].set_title("posterior σ map"); ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout(); fig.savefig(out / "chest_contour.png"); plt.close(fig)


if __name__ == "__main__":
    main()
