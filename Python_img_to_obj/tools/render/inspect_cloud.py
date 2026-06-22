"""One-command inspector for the render->inspect->fix loop.

Takes a 3DGS .ply, a SplatCloud .npz, or a .glb and produces, in <out>/:
  stats.json            - point count, bbox, axis extents (z/y ratio flags billboards),
                          color/opacity ranges
  inspect_3d.png        - matplotlib 3D scatter
  inspect_ortho.png     - front/side/top + 3D ortho panel
  apose_mesh.glb        - Poisson relightable mesh (open3d)
  blender_studio.png    - Blender studio relight
  blender_raking.png    - Blender raking side-light (reveals micro-relief)
  montage.png           - all of the above tiled, for a single eyeball pass

Usage:
  python tools/render/inspect_cloud.py <input.ply|.npz|.glb> <out_dir> [--no-blender]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from pipeline.types import SplatCloud
from pipeline.export import render3d, relight
from pipeline.backends import lhm_backend

BLENDER = ROOT / "vendor" / "blender" / "blender-5.1.2-windows-x64" / "blender.exe"


def load_any(path: Path):
    path = Path(path)
    if path.suffix == ".npz":
        from pipeline.export.ply import load_splat_npz
        return load_splat_npz(path), None
    if path.suffix == ".ply":
        return lhm_backend.load_gaussian_ply(path), None
    if path.suffix == ".glb":
        return None, path
    raise ValueError(f"unsupported input: {path}")


def cloud_stats(sp: SplatCloud) -> dict:
    c = np.asarray(sp.centers, float)
    if c.shape[0] == 0:
        return {"n": 0}
    mn, mx = c.min(0), c.max(0)
    ext = mx - mn
    col = np.asarray(sp.colors, float)
    s = {
        "n": int(c.shape[0]),
        "bbox_min": [round(float(v), 4) for v in mn],
        "bbox_max": [round(float(v), 4) for v in mx],
        "extent_xyz": [round(float(v), 4) for v in ext],
        "z_over_y": round(float(ext[2] / (ext[1] + 1e-9)), 4),   # ~<0.1 => billboard
        "x_over_y": round(float(ext[0] / (ext[1] + 1e-9)), 4),
        "color_min": [round(float(v), 3) for v in col.min(0)] if col.size else None,
        "color_max": [round(float(v), 3) for v in col.max(0)] if col.size else None,
        "opacity_mean": round(float(np.mean(sp.opacities)), 3) if len(sp.opacities) else None,
    }
    s["likely_billboard"] = s["z_over_y"] < 0.12
    return s


def run_blender(glb: Path, out: Path):
    if not BLENDER.exists():
        return {"blender": "not installed"}
    r = subprocess.run(
        [str(BLENDER), "--background", "--python", str(ROOT / "tools" / "blender_render.py"),
         "--", "--glb", str(glb.resolve()), "--out", str(out.resolve())],
        capture_output=True, text=True, timeout=600)
    ok = "RENDER_DONE" in r.stdout
    return {"blender_ok": ok, "blender_tail": r.stdout.strip().splitlines()[-3:]}


def montage(out: Path, panels):
    imgs = [(lbl, out / f) for lbl, f in panels if (out / f).exists()]
    if not imgs:
        return
    cell = 460
    cols = min(3, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * (cell + 24)), (18, 18, 24))
    dr = ImageDraw.Draw(canvas)
    for i, (lbl, p) in enumerate(imgs):
        im = Image.open(p).convert("RGB"); im.thumbnail((cell, cell))
        cx, cy = i % cols, i // cols
        x = cx * cell + (cell - im.width) // 2
        y = cy * (cell + 24) + 24 + (cell - im.height) // 2
        canvas.paste(im, (x, y)); dr.text((cx * cell + 6, cy * (cell + 24) + 6), lbl, fill=(150, 230, 150))
    canvas.save(out / "montage.png")


def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    inp = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    no_blender = "--no-blender" in sys.argv

    sp, glb_in = load_any(inp)
    info = {}
    if sp is not None:
        info["stats"] = cloud_stats(sp)
        (out / "stats.json").write_text(json.dumps(info["stats"], indent=2))
        print("STATS", json.dumps(info["stats"]))
        try:
            render3d.render_3d_plot(out / "inspect_3d.png", sp.centers, sp.colors, title=inp.stem)
            render3d.render_ortho_views(out / "inspect_ortho.png", sp.centers, sp.colors, title=inp.stem)
        except Exception as e:  # noqa: BLE001
            print("render3d error", e)
        try:
            relight.save_gaussian_ply_full(out / "apose_splats.ply", sp)
            mesh = relight.build_relight_mesh(sp, poisson_depth=10)
            if mesh is not None:
                relight.save_glb(mesh, out / "apose_mesh.glb")
                info["mesh_vertices"] = int(len(mesh.vertices))
                print("MESH verts", info["mesh_vertices"])
        except Exception as e:  # noqa: BLE001
            print("mesh error", e)
    glb = glb_in or (out / "apose_mesh.glb")
    if not no_blender and glb.exists():
        info["blender"] = run_blender(glb, out)
        print("BLENDER", info["blender"])

    montage(out, [("3D", "inspect_3d.png"), ("ortho", "inspect_ortho.png"),
                  ("studio", "blender_studio.png"), ("raking", "blender_raking.png")])
    print("MONTAGE", (out / "montage.png").exists())


if __name__ == "__main__":
    main()
