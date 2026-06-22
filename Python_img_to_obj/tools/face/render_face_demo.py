"""Render the reconstructed FLAME face meshes from several yaw angles.

Pure-numpy z-buffer rasterizer with barycentric vertex-color interpolation and
a touch of Lambertian shading, so each subject gets a real multi-angle 3D demo
(not a single flat screenshot). Builds one row per subject + a combined sheet.

  python tools/face/render_face_demo.py            # all subjects
  python tools/face/render_face_demo.py 1 4 5      # pick subjects
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
SZ = 360
YAWS = (-40, -20, 0, 20, 40)
BG = np.array([245, 245, 245], np.uint8)


def roty(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rasterize(verts: np.ndarray, faces: np.ndarray, vcols: np.ndarray) -> np.ndarray:
    """Orthographic z-buffer render of a vertex-colored mesh. verts already
    centered/scaled into roughly [-1,1]; +Y up, viewer looks down -Z."""
    img = np.repeat(BG[None, None, :], SZ, 0).repeat(SZ, 1).astype(np.float32)
    zbuf = np.full((SZ, SZ), -np.inf, np.float32)

    # screen coords: x -> col, y -> row (flip y), keep z for depth + shading
    m = 0.92
    sx = (verts[:, 0] * m * 0.5 + 0.5) * (SZ - 1)
    sy = (0.5 - verts[:, 1] * m * 0.5) * (SZ - 1)
    sz = verts[:, 2]

    # per-face flat normal for shading; light from camera+upper-left
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nl > 1e-9, nl, 1.0)
    light = np.array([0.3, 0.4, 1.0]); light = light / np.linalg.norm(light)
    shade = np.clip(0.45 + 0.55 * (n @ light), 0.25, 1.0)  # cull-free, abs-ish

    fc = vcols[faces].mean(1)  # per-face base color (fast); good enough at this res
    for i in range(len(faces)):
        a, b, c = faces[i]
        xs = np.array([sx[a], sx[b], sx[c]])
        ys = np.array([sy[a], sy[b], sy[c]])
        x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
        y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
        if x1 < 0 or y1 < 0 or x0 >= SZ or y0 >= SZ:
            continue
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, SZ - 1), min(y1, SZ - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        # barycentric
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-9:
            continue
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        z = w0 * sz[a] + w1 * sz[b] + w2 * sz[c]
        gyi, gxi = gy[inside], gx[inside]
        zi = z[inside]
        cur = zbuf[gyi, gxi]
        win = zi > cur
        if not win.any():
            continue
        yy, xx, zz = gyi[win], gxi[win], zi[win]
        zbuf[yy, xx] = zz
        img[yy, xx] = np.clip(fc[i] * shade[i], 0, 255)
    return img.astype(np.uint8)


def center_scale(v: np.ndarray) -> np.ndarray:
    v = v - v.mean(0)
    v = v / (np.abs(v).max() + 1e-9)
    return v


def render_subject(s: int) -> Image.Image | None:
    try:
        v = center_scale(np.load(RUNS / f"decafull_s{s}_verts.npy").astype(float))
        f = np.load(RUNS / f"decafull_s{s}_faces.npy").astype(int)
        c = np.load(RUNS / f"decafull_s{s}_vcols.npy").astype(float)
    except Exception as e:
        print(f"s{s}: missing assets ({e})")
        return None
    tiles = []
    for yaw in YAWS:
        rv = v @ roty(yaw).T
        tiles.append(Image.fromarray(rasterize(rv, f, c)))
    pad = 4
    row = Image.new("RGB", (len(tiles) * SZ + (len(tiles) + 1) * pad, SZ + pad + 22), (25, 25, 25))
    ImageDraw.Draw(row).text((6, 4), f"subject s{s}  -  FLAME face recon, yaw {YAWS} deg", fill=(255, 255, 0))
    for i, t in enumerate(tiles):
        row.paste(t, (pad + i * (SZ + pad), 22))
    print(f"s{s}: rendered {len(tiles)} angles")
    return row


def main(subjects: list[int]):
    out = RUNS / "FACE_DEMO"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in subjects:
        r = render_subject(s)
        if r is None:
            continue
        r.save(out / f"face_demo_s{s}.png")
        rows.append(r)
    if rows:
        w = max(r.width for r in rows)
        h = sum(r.height for r in rows) + 6 * (len(rows) + 1) + 24
        sheet = Image.new("RGB", (w, h), (10, 10, 10))
        ImageDraw.Draw(sheet).text((10, 6), "MESHMAP - FACE-ONLY DEMO (3D FLAME recon, multi-angle)", fill=(120, 230, 255))
        y = 26
        for r in rows:
            sheet.paste(r, (0, y)); y += r.height + 6
        sheet.save(out / "FACE_DEMO_ALL.png")
        print(f"SAVED {out/'FACE_DEMO_ALL.png'} ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
    main(args)
