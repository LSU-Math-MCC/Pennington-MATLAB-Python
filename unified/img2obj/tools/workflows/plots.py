"""Modular full-pipeline illustration for a single subject: one labeled figure showing EACH
module's REAL output, end to end. No mock-ups — every panel is computed from the actual stage.

Stages: source -> SAM2+depth person matte -> skin mask -> de-lit albedo -> Multi-HMR 2D pose
-> CLIP build attribute -> A-pose SMPL-X mesh (orthographic) -> 17 anatomical markers ->
textured A-pose render -> reproduced-SHAPY metric measurements.

Run (WSL lhm): python tools/workflows/plots.py [s1]
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
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = _repo
sys.path.insert(0, REPO + "/tools")


def main():
    import texture_uv_bake as TB
    import lhm_anthropometry as A
    import clip_shape as CS
    s = sys.argv[1] if len(sys.argv) > 1 else "ssp3d_bodybuilder"
    ssp_glob = f"{REPO}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png"
    subject_glob = s if any(ch in s for ch in "*?[]/\\") else ssp_glob
    img_path = sorted(glob.glob(subject_glob))[0]
    img = np.asarray(Image.open(img_path).convert("RGB"))

    pm = TB.person_mask(img)                       # 2: SAM2 (+depth) matte
    sk = TB.skin_mask(img) & pm                    # 3: skin
    albedo = TB.delight(img, pm)                   # 4: lighting-corrected albedo

    est = A._estimator()                            # 5: Multi-HMR 2D pose
    _, h, j2d = TB.posed_view(est, img_path)
    j2d = np.asarray(j2d)

    all_imgs = sorted(glob.glob(subject_glob))[:5]
    bmi, post = CS.clip_build_bmi(all_imgs)         # 6: CLIP attribute (fused across views)
    labels = ["v.slim", "slim", "avg", "heavy", "v.heavy"]

    betas = np.load(f"{REPO}/runs/fit_{s}/fused_betas.npy")[:10]   # 7: A-pose mesh
    v, j, faces, named = A.smplx_apose(betas, gender="female")

    marks = json.load(open(f"{REPO}/runs/penn_integration/{s}_markers_smplx.json"))["markers"]
    meas = json.load(open(f"{REPO}/runs/fit_{s}/shapy_measurements.json"))
    render = f"{REPO}/runs/uv_{s}_skin/blender_studio.png"

    fig = plt.figure(figsize=(18, 9))
    def add_img(pos, arr, title):
        ax = fig.add_subplot(2, 5, pos); ax.imshow(arr); ax.set_title(title, fontsize=10); ax.axis("off")

    add_img(1, img, "1. source image")
    ov = img.copy(); ov[~pm] = (ov[~pm] * 0.25).astype(np.uint8)
    add_img(2, ov, "2. SAM2+depth person matte")
    ov2 = img.copy(); ov2[sk] = (0.4 * ov2[sk] + np.array([0, 160, 90]) * 0.6).astype(np.uint8)
    add_img(3, ov2, "3. skin mask (clothes excluded)")
    add_img(4, albedo, "4. de-lit albedo (WB+shading)")

    ax5 = fig.add_subplot(2, 5, 5); ax5.imshow(img)
    if j2d.ndim == 2 and len(j2d):
        ax5.scatter(j2d[:, 0], j2d[:, 1], s=8, c="lime")
    ax5.set_title("5. Multi-HMR 2D pose", fontsize=10); ax5.axis("off")

    ax6 = fig.add_subplot(2, 5, 6); ax6.bar(range(5), post, color="teal")
    ax6.set_xticks(range(5)); ax6.set_xticklabels(labels, fontsize=7)
    ax6.set_title(f"6. CLIP build -> BMI~{bmi:.1f}", fontsize=10)

    ax7 = fig.add_subplot(2, 5, 7); ax7.scatter(v[::20, 0], v[::20, 1], s=0.5, c="lightgray")
    ax7.set_aspect("equal"); ax7.set_title("7. A-pose SMPL-X (front)", fontsize=10); ax7.axis("off")
    ax8 = fig.add_subplot(2, 5, 8); ax8.scatter(v[::20, 2], v[::20, 1], s=0.5, c="lightgray")
    for m in marks:
        p = m["xyz"]; ax8.scatter([p[2]], [p[1]], s=22, c="red")
    ax8.set_aspect("equal"); ax8.set_title("8. 17 markers (side)", fontsize=10); ax8.axis("off")

    if os.path.exists(render):
        add_img(9, np.asarray(Image.open(render).convert("RGB")), "9. textured A-pose (de-lit)")
    ax10 = fig.add_subplot(2, 5, 10); ax10.axis("off")
    txt = (f"10. reproduced-SHAPY metrics\n\nheight {meas['height_cm']} cm\n"
           f"mass   {meas['mass_kg']} kg\nchest  {meas['chest_cm']} cm\n"
           f"waist  {meas['waist_cm']} cm\nhips   {meas['hips_cm']} cm")
    ax10.text(0.05, 0.9, txt, va="top", fontsize=11, family="monospace")

    fig.suptitle(f"meshmap modular pipeline — {s}  ({os.path.basename(img_path)})", fontsize=14)
    out = f"{REPO}/runs/PIPELINE_{s}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
