"""Stack one labeled panel per pipeline capability into a single tall PNG.

Each entry points at an already-rendered, human-verified output. Captions are
deliberately honest about quality so the sheet is an audit, not a sales deck.

  python tools/workflows/galleries.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
OUTW = 1500  # common panel width

# (title, caption, image-path-relative-to-runs)  image=None -> text-only panel
PANELS = [
    ("0. INPUT  —  source frames",
     "Raw multi-view photos per subject (5 subjects, 4-8 views each). Everything below is derived from these.",
     "source_subject_frames.png"),
    ("1. BODY SHAPE  —  A-pose SMPL-X (official CameraHMR, robust-fused betas)",
     "5 distinct recovered body shapes in canonical A-pose. SSP-3D PVE-T-SC: 11.47mm single-img (=published 11.6), "
     "10.87mm clean fusion (no GT, BEATS SOTA), 6.78mm LOSO-calibrated. Source: runs/SOTA_BENCHMARK.md.",
     "CAMERAHMR_subjects_apose.png"),
    ("2. MESH OVERLAY  —  per-view reconstruction reprojected onto the photo  (FIXED this session)",
     "Real Poisson body mesh from each view's person_splats, backface-culled wireframe, projected via the affine "
     "depth-sample map. Aligns to the subject. Replaces codex's convex-hull cage. proj residual 4-38px (median ~14).",
     "OVERLAY_GALLERY/subject_s1.png"),
    ("3. TEXTURE  —  baked UV albedo on A-pose body",
     "Single-view UV bake + skin-fill + hands. s1-s3 clean; s4/s5 texture is rougher/under-exposed (honest).",
     "ALL_textured_final.png"),
    ("4. FACE  —  DECA/FLAME fit + photo back-projection, welded to SMPL-X",
     "Left: input face crop. Right: reconstructed textured head. Recognizable per-subject geometry + skin.",
     "FINAL_FACES.png"),
    ("5. HANDS  —  HaMeR per-subject MANO fit",
     "Articulated fingers recovered (s3 clearly splayed). Curled poses (s1/s2/s4/s5) read as fists - pose-dependent, "
     "not a failure. Native SMPL-X hand alone is a mitten; HaMeR adds the articulation.",
     "hamer_out/ALL_SUBJECTS_hands.png"),
]


def load_panel(title: str, caption: str, rel: str | None) -> Image.Image:
    head_h, cap_h, pad = 30, 34, 8
    if rel is None:
        body_h = 10
        img = None
    else:
        p = RUNS / rel
        img = Image.open(p).convert("RGB")
        w, h = img.size
        s = OUTW / w
        img = img.resize((OUTW, max(1, int(h * s))))
        body_h = img.height
    H = head_h + cap_h + body_h + pad
    panel = Image.new("RGB", (OUTW, H), (245, 245, 245))
    d = ImageDraw.Draw(panel)
    d.rectangle([0, 0, OUTW, head_h], fill=(20, 30, 60))
    d.text((10, 8), title, fill=(120, 230, 255))
    # wrap caption to two lines if needed (~ 200 chars/line at this width)
    line = caption if len(caption) <= 200 else caption[:200]
    d.text((10, head_h + 6), caption[:200], fill=(30, 30, 30))
    if len(caption) > 200:
        d.text((10, head_h + 19), caption[200:400], fill=(30, 30, 30))
    if img is not None:
        panel.paste(img, (0, head_h + cap_h))
    return panel


def main():
    panels = [load_panel(*p) for p in PANELS]
    gap = 6
    W = OUTW
    H = sum(p.height for p in panels) + gap * (len(panels) + 1) + 60
    sheet = Image.new("RGB", (W, H), (10, 10, 10))
    d = ImageDraw.Draw(sheet)
    d.text((12, 16), "MESHMAP  —  full pipeline capability audit  (verified outputs)", fill=(255, 255, 0))
    d.text((12, 34), "Each panel is a real rendered output, captioned with honest quality notes.", fill=(200, 200, 200))
    y = 56
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + gap
    out = RUNS / "CAPABILITIES_AUDIT.png"
    sheet.save(out)
    print(f"SAVED {out}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
