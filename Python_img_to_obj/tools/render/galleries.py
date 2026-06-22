"""Render-stage figure galleries (subcommands).

Subcommands:
  montage <run_dir>        labeled per-stage debug contact sheet for a run
  overlay [run_dir ...]    original | mesh-overlay pairs per subject + combined sheet

To add a gallery: write a function and a thin ``_<name>_cli`` wrapper, then add it
to ``COMMANDS`` at the bottom. ``montage`` is also importable as a library
(``from galleries import montage_run``).

Usage: python tools/render/galleries.py <subcommand> [args...]
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

_REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- montage
STEPS = [
    ("input.png", "1 input"),
    ("mask.png", "2 person mask"),
    ("pose_overlay.png", "3 pose (MediaPipe)"),
    ("depth_preview.png", "4 depth (Depth-Anything)"),
    ("selected_depth_overlay.png", "5 mask-gated depth"),
    ("splat_projection_overlay.png", "6 splat assignment"),
    ("face_region_mask.png", "7 face region"),
    ("face_landmarks_overlay.png", "8 face landmarks"),
    ("canonical_3d.png", "9 canonical 3D"),
    ("canonical_ortho.png", "10 ortho A-pose"),
]


def montage(debug_dir: Path, out: Path, cell=300, cols=4):
    rows = (len(STEPS) + cols - 1) // cols
    pad = 26
    canvas = Image.new("RGB", (cols * cell, rows * (cell + pad)), (20, 20, 28))
    dr = ImageDraw.Draw(canvas)
    for i, (fname, label) in enumerate(STEPS):
        p = debug_dir / fname
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell, cell))
        c, r = i % cols, i // cols
        x = c * cell + (cell - im.width) // 2
        y = r * (cell + pad) + pad + (cell - im.height) // 2
        canvas.paste(im, (x, y))
        dr.text((c * cell + 6, r * (cell + pad) + 6), label, fill=(140, 230, 140))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("wrote", out)


def montage_run(run_dir):
    """Build a step montage for every debug dir under a run (library entrypoint)."""
    run = Path(run_dir)
    debug_dirs = sorted(run.glob("views/*/*/debug")) or sorted(run.glob("images/*/debug"))
    if (run / "debug" / "input.png").exists():
        debug_dirs.append(run / "debug")
    if not debug_dirs:
        print("no debug dirs found under", run)
        return
    for d in debug_dirs:
        tag = "_".join(d.parts[-3:-1]) if d.parent != run else "single"
        montage(d, run / "debug" / f"steps_{tag}.png")


def _montage_cli():
    montage_run(sys.argv[1] if len(sys.argv) > 1 else "runs/subject_s5")


# ----------------------------------------------------------------------------- overlay
OUT = _REPO / "runs" / "OVERLAY_GALLERY"
CELL = 320  # thumbnail height per panel


def thumb(path: Path, label: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = CELL / h
    img = img.resize((max(1, int(w * s)), CELL))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, len(label) * 7 + 6, 16], fill=(0, 0, 0))
    draw.text((3, 3), label, fill=(255, 255, 0))
    return img


def pair(orig: Path, over: Path, tag: str) -> Image.Image:
    left = thumb(orig, f"{tag} orig")
    right = thumb(over, f"{tag} mesh")
    gap = 4
    canvas = Image.new("RGB", (left.width + gap + right.width, CELL), (20, 20, 20))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def grid(cells: list[Image.Image], cols: int, title: str) -> Image.Image:
    if not cells:
        return Image.new("RGB", (400, 60), (20, 20, 20))
    cw = max(c.width for c in cells)
    rows = (len(cells) + cols - 1) // cols
    pad, top = 6, 26
    W = cols * cw + (cols + 1) * pad
    H = top + rows * (CELL + pad) + pad
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    ImageDraw.Draw(canvas).text((8, 6), title, fill=(255, 255, 255))
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        x = pad + col * (cw + pad)
        y = top + pad + r * (CELL + pad)
        canvas.paste(c, (x, y))
    return canvas


def overlay_gallery(run_dirs: list[str]):
    import json

    OUT.mkdir(parents=True, exist_ok=True)
    subject_sheets = []
    for rd in run_dirs:
        run = Path(rd)
        rep = run / "overlays" / "overlay_report.json"
        if not rep.exists():
            print(f"SKIP {run}: no overlay_report.json")
            continue
        data = json.loads(rep.read_text())
        cells = []
        for v in data["views"]:
            if "error" in v:
                print(f"  view error {v['view']}: {v['error']}")
                continue
            tag = Path(v["out"]).stem.replace("_mesh_overlay", "")
            res = v.get("projection", {}).get("mean_px")
            tag = f"{tag} {res:.0f}px" if res is not None else tag
            cells.append(pair(Path(v["image"]), Path(v["out"]), tag))
        sheet = grid(cells, cols=min(3, len(cells)) or 1, title=f"{run.name}  ({len(cells)} views)")
        sp = OUT / f"{run.name}.png"
        sheet.save(sp)
        print(f"SAVED {sp}  ({len(cells)} views)")
        subject_sheets.append(sheet)

    if subject_sheets:
        w = max(s.width for s in subject_sheets)
        h = sum(s.height for s in subject_sheets) + 8 * (len(subject_sheets) + 1)
        combined = Image.new("RGB", (w, h), (10, 10, 10))
        y = 8
        for s in subject_sheets:
            combined.paste(s, (0, y))
            y += s.height + 8
        cp = OUT / "ALL_SUBJECTS.png"
        combined.save(cp)
        print(f"SAVED {cp}  ({combined.size[0]}x{combined.size[1]})")


def _overlay_cli():
    overlay_gallery(sys.argv[1:] or [f"runs/subject_s{i}" for i in range(1, 6)])


# ----------------------------------------------------------------------------- dispatch
COMMANDS = {"montage": _montage_cli, "overlay": _overlay_cli}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <{'|'.join(COMMANDS)}> [args...]")
    _name = sys.argv[1]
    sys.argv = [f"{sys.argv[0]} {_name}", *sys.argv[2:]]
    COMMANDS[_name]()
