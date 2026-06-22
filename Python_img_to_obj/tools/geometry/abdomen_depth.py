"""Recover the NON-GENERIC abdomen transverse (x-z) contour from a single image by fusing
Depth-Anything-V2 over the masked abdomen band. The front surface depth across the abdomen
width = the person-specific front contour (back is symmetry-inferred). Calibrated to metric via
the measured abdomen coronal width. Demonstrates the mission crux on one image.

Run (WSL lhm): PYTHONPATH=src python tools/geometry/abdomen_depth.py <image> [waist_cm]
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = _repo
sys.path.insert(0, REPO + "/tools"); sys.path.insert(0, REPO + "/src")


def main():
    import texture_uv_bake as TB, lhm_anthropometry as A
    from pipeline.backends import real
    img_path = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png"
    width_cm = float(sys.argv[2]) if len(sys.argv) > 2 else 74.0   # coronal abdomen width prior
    img = np.asarray(Image.open(img_path).convert("RGB"))
    est = A._estimator()
    _, h, j2d = TB.posed_view(est, img_path)
    if h is None:
        print("no detection"); return
    j2d = np.asarray(j2d)
    pm = TB.person_mask(img)
    z = real._Models.depth_map(img, size="base")           # HxW, smaller=closer

    # abdomen band: between spine and pelvis (navel height), a few rows
    pelvis_y = j2d[0, 1]; spine_y = j2d[6, 1] if j2d.shape[0] > 6 else j2d[3, 1]
    navel_y = int(0.55 * pelvis_y + 0.45 * spine_y)
    band = slice(max(navel_y - 4, 0), navel_y + 5)
    cols = []
    for x in range(img.shape[1]):
        col = pm[band, x]
        if col.sum() >= 3:
            cols.append((x, float(np.median(z[band, x][col]))))
    if len(cols) < 10:
        print("abdomen band too small"); return
    cols = np.array(cols)
    order = np.argsort(cols[:, 0]); xs = cols[order, 0]; zf = cols[order, 1]
    # OCCLUSION-ROBUST: a hand-on-hip is CLOSER than the belly -> a localized depth SPIKE. Reject
    # columns that deviate from a robust (median) baseline; never let them indent the contour.
    from scipy.ndimage import median_filter, gaussian_filter1d
    base = median_filter(zf, size=max(5, len(zf) // 6))    # spike-robust baseline
    resid = zf - base
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    good = np.abs(resid) < 3.0 * mad                       # drop hand/occlusion spikes
    if good.sum() >= 6:
        zf = np.interp(xs, xs[good], zf[good])             # interpolate over occluded columns
    zf = gaussian_filter1d(zf, 2.0)                        # smooth belly front
    px_w = xs.max() - xs.min()
    cm_per_px = width_cm / px_w
    xc = (xs - xs.mean()) * cm_per_px                      # centered x in cm
    # front depth relative to nearest point, scaled to cm (monocular relative -> proportional)
    zc = (zf - zf.min())
    zc = zc / (zc.max() + 1e-6)                            # 0..1 across abdomen
    front_depth_cm = zc * (width_cm * 0.42)                # depth ~ 0.42*width (anthro prior)
    front = -front_depth_cm                                # front bulges toward camera (-z)

    # generic SMPL-X abdomen transverse (ellipse) for comparison
    th = np.linspace(0, np.pi, 60)
    ex = (width_cm / 2) * np.cos(th); ez = -(width_cm * 0.30) * np.sin(th)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ex, ez, "r--", label="generic SMPL-X (ellipse)")
    ax.plot(xc, front, "g-", lw=2, label="recovered FRONT (depth-fused, real)")
    # The back (spine side) is NOT a mirror of the belly — it is roughly FLAT. From one frontal
    # image the back is UNOBSERVED; we do NOT fabricate it by symmetry. Show it flat/unknown.
    ax.plot(xc, np.zeros_like(xc), "k:", lw=1, label="back: UNOBSERVED (not mirrored)")
    ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("z depth (cm)")
    ax.set_title("Abdomen transverse @ navel — REAL front only (back unobserved, never mirrored)")
    ax.legend(fontsize=8)
    out = f"{REPO}/runs/bench_single/abdomen_transverse.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("front depth range cm:", round(float(front_depth_cm.max()), 1),
          "abdomen width cm:", round(float(px_w * cm_per_px), 1))
    print("saved", out)


if __name__ == "__main__":
    main()
