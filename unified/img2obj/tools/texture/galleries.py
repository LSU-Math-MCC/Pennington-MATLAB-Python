"""Texture-stage figure galleries (subcommands).

Subcommands:
  eval <root>      texture-variant montages from eval_texture_variants.sh outputs
  beforeafter      CLIP de-bias before/after measurement table -> runs/BEFORE_AFTER.json

To add a gallery: write a function and a thin ``_<name>_cli`` wrapper, then add it
to ``COMMANDS`` at the bottom. Heavy deps (torch/smplx) are imported lazily inside
``beforeafter`` so the module stays cheap to import.

Usage: python tools/texture/galleries.py <subcommand> [args...]
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

_REPO = Path(__file__).resolve().parents[2]


# -------------------------------------------------------------------------------- eval
def load_thumb(path, size):
    if not path.exists():
        return Image.new("RGB", size, (245, 245, 245))
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def label(draw, xy, text):
    draw.rectangle((xy[0], xy[1], xy[0] + 210, xy[1] + 20), fill="white")
    draw.text((xy[0] + 4, xy[1] + 3), text, fill=(180, 0, 0))


def make(root, kind, tile):
    entries = sorted(p for p in root.iterdir() if p.is_dir() and "_" in p.name and
                     "_crop" not in p.name and (p / f"render_{kind}.png").exists())
    subjects = sorted({p.name.split("_", 1)[0] for p in entries})
    preferred = ["off", "face", "face-head"]
    present = {p.name.split("_", 1)[1] for p in entries}
    modes = [m for m in preferred if m in present] + sorted(present - set(preferred))
    W = tile[0] * len(modes)
    H = tile[1] * len(subjects)
    out = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(out)
    for r, subj in enumerate(subjects):
        for c, mode in enumerate(modes):
            d = root / f"{subj}_{mode}"
            im = load_thumb(d / f"render_{kind}.png", tile)
            x, y = c * tile[0], r * tile[1]
            out.paste(im, (x, y))
            rep = {}
            try:
                rep = json.loads((d / "uv_report.json").read_text())
            except Exception:
                pass
            txt = f"{subj} {mode}"
            if rep:
                txt += (f" cov={rep.get('coverage_texels', 0):.2f}"
                        f" face={rep.get('face_repair_filled_texels', 0)}"
                        f" occ={rep.get('face_occlusion_repair_texels', 0)}")
            label(draw, (x, y), txt)
    out_path = root / f"montage_{kind}.png"
    out.save(out_path)
    print(out_path)


def _eval_cli():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/texture_eval_latest")
    make(root, "face", (360, 360))
    make(root, "front", (300, 440))


# -------------------------------------------------------------------------- beforeafter
_LM = dict(HEAD_TOP=8976, LEFT_HEEL=8847, LEFT_NIPPLE=3572, BELLY_BUTTON=5939, PUBIC_BONE=5949)


def _beforeafter_cli():
    """Measure HEAVY silhouette-fused betas vs CLIP-corrected betas with the reproduced
    SHAPY measurement module; prints a table + writes runs/BEFORE_AFTER.json."""
    import os
    import numpy as np
    import torch
    import smplx

    sys.path.insert(0, str(_REPO / "tools" / "anthro"))
    import shapy_measure as SM
    import lhm_anthropometry as A

    def meas(model, faces, betas):
        with torch.no_grad():
            v = model(betas=torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0)).vertices[0].numpy()
        return dict(height_cm=round(abs(v[_LM["HEAD_TOP"], 1] - v[_LM["LEFT_HEEL"], 1]) * 100, 1),
                    mass_kg=round(SM.mesh_volume(v, faces) * 985, 1),
                    waist_cm=round(SM.plane_perimeter(v, faces, v[_LM["BELLY_BUTTON"], 1]) * 100, 1),
                    hips_cm=round(SM.plane_perimeter(v, faces, v[_LM["PUBIC_BONE"], 1]) * 100, 1))

    model = smplx.create(A.HUMAN_MODELS, model_type="smplx", gender="neutral", num_betas=10)
    faces = model.faces.astype(np.int64)
    out = {}
    print(f"{'subj':4} {'HEAVY (Multi-HMR+sil)':28} {'AFTER (+CLIP attributes)':28}")
    for s in ["s1", "s2", "s3", "s4", "s5"]:
        d = str(_REPO / "runs" / f"fit_{s}")
        heavy = f"{d}/fused_betas_heavy.npy"; cur = f"{d}/fused_betas.npy"
        if not (os.path.exists(heavy) and os.path.exists(cur)):
            continue
        b = meas(model, faces, np.load(heavy)); a = meas(model, faces, np.load(cur))
        out[s] = dict(before=b, after=a)
        print(f"{s:4} {b['mass_kg']}kg w{b['waist_cm']} h{b['hips_cm']:<10}    "
              f"{a['mass_kg']}kg w{a['waist_cm']} h{a['hips_cm']}")
    json.dump(out, open(str(_REPO / "runs" / "BEFORE_AFTER.json"), "w"), indent=2)
    print("BEFORE_AFTER_DONE")


# ----------------------------------------------------------------------------- dispatch
COMMANDS = {"eval": _eval_cli, "beforeafter": _beforeafter_cli}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <{'|'.join(COMMANDS)}> [args...]")
    _name = sys.argv[1]
    sys.argv = [f"{sys.argv[0]} {_name}", *sys.argv[2:]]
    COMMANDS[_name]()
