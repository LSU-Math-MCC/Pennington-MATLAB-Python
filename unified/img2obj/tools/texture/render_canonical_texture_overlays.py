"""Render a canonical UV texture bake back onto the original photos.

This is the texture acceptance loop: bake in canonical SMPL-X UV space, then
repose/project the textured mesh into each source view and alpha-composite it
over the original image. Unlike standalone A-pose renders, this shows the exact
photo/mesh agreement problem.

Run in the lhm env:
  python tools/texture/render_canonical_texture_overlays.py \
    --subject "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png" \
    --bake runs/mesh_texture_notebook_s4 \
    --out runs/mesh_texture_notebook_s4/photo_overlays
"""

from __future__ import annotations
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


import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import texture_uv_bake as TB  # noqa: E402
import lhm_anthropometry as A  # noqa: E402


def _font(size=18):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def image_paths(subject: Path):
    subject_s = str(subject)
    if any(ch in subject_s for ch in "*?[]"):
        return [Path(p) for p in sorted(glob.glob(subject_s))]
    if subject.is_dir():
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            paths.extend(glob.glob(str(subject / "**" / ext), recursive=True))
        return [Path(p) for p in sorted(paths)]
    return [subject]


def vt_to_vertex(fv: np.ndarray, fvt: np.ndarray, n_vt: int):
    out = np.zeros(n_vt, np.int64)
    for fi in range(fv.shape[0]):
        for k in range(3):
            out[fvt[fi, k]] = fv[fi, k]
    return out


def bilinear_texture(atlas: np.ndarray, uv: np.ndarray):
    h, w = atlas.shape[:2]
    x = np.clip(uv[:, 0] * (w - 1), 0, w - 1.001)
    y = np.clip((1.0 - uv[:, 1]) * (h - 1), 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    return (
        atlas[y0, x0] * (1 - fx) * (1 - fy)
        + atlas[y0, x0 + 1] * fx * (1 - fy)
        + atlas[y0 + 1, x0] * (1 - fx) * fy
        + atlas[y0 + 1, x0 + 1] * fx * fy
    )


def rasterize_textured(img: np.ndarray, xy: np.ndarray, z: np.ndarray, uv: np.ndarray,
                       faces: np.ndarray, atlas: np.ndarray):
    """Rasterize a textured mesh with per-fragment UV lookup.

    The earlier diagnostic renderer sampled the atlas only at mesh vertices and
    interpolated those RGB values across each triangle. That is fine for vertex
    colors but wrong for a UV texture, especially a sparse per-view atlas: it
    creates smears/scars that are renderer artifacts, not bake artifacts.
    """
    h, w = img.shape[:2]
    color = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    depth = np.full((h, w), np.inf, np.float32)

    atlas_f = atlas.astype(np.float32)
    has_tex_alpha = atlas_f.ndim == 3 and atlas_f.shape[2] > 3
    for tri in faces:
        pts = xy[tri].astype(np.float32)
        zz = z[tri].astype(np.float32)
        tuv = uv[tri].astype(np.float32)
        if not np.isfinite(pts).all() or not np.isfinite(zz).all() or np.any(zz <= 1e-4):
            continue
        area = cv2.contourArea(pts)
        if area < 0.5:
            continue
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1).astype(np.float32)
        a, b, c = pts
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(float(den)) < 1e-6:
            continue
        vp = p - a
        b1 = (vp[:, 0] * v1[1] - vp[:, 1] * v1[0]) / den
        b2 = (v0[0] * vp[:, 1] - v0[1] * vp[:, 0]) / den
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-4) & (b1 >= -1e-4) & (b2 >= -1e-4)
        if not inside.any():
            continue
        pix = p[inside].astype(int)
        bary = np.stack([b0, b1, b2], 1)[inside]
        dz = (zz[None, :] * bary).sum(1)
        yy, xx = pix[:, 1], pix[:, 0]
        closer = dz < depth[yy, xx]
        if not closer.any():
            continue
        yy, xx, bary, dz = yy[closer], xx[closer], bary[closer], dz[closer]
        frag_uv = (tuv[None, :, :] * bary[:, :, None]).sum(1)
        tex = bilinear_texture(atlas_f, frag_uv)
        col = tex[:, :3]
        aout = tex[:, 3] / 255.0 if has_tex_alpha else np.ones(len(tex), np.float32)
        keep_alpha = aout > 0.05
        if not keep_alpha.any():
            continue
        yy, xx, dz, aout, col = yy[keep_alpha], xx[keep_alpha], dz[keep_alpha], aout[keep_alpha], col[keep_alpha]
        depth[yy, xx] = dz
        color[yy, xx] = col
        alpha[yy, xx] = aout

    if alpha.any() and not has_tex_alpha:
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.65)
    return np.clip(color, 0, 255).astype(np.uint8), np.clip(alpha, 0, 1)


def make_view_conditioned_atlas(img: np.ndarray, v3d: np.ndarray, vnormals: np.ndarray,
                                xy_v: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                                texel_v: np.ndarray, bary: np.ndarray, valid: np.ndarray,
                                faces: np.ndarray, atlas_size: int):
    """Bake one source view into canonical UV texels, with alpha for observed texels only."""
    tv = texel_v[valid]
    tb = bary[valid]
    p3d = (v3d[tv] * tb[:, :, None]).sum(1)
    n3d = (vnormals[tv] * tb[:, :, None]).sum(1)
    z = np.clip(p3d[:, 2], 1e-5, None)
    uv_img = np.stack([fx * p3d[:, 0] / z + cx, fy * p3d[:, 1] / z + cy], 1)
    h, w = img.shape[:2]
    in_img = (uv_img[:, 0] >= 0) & (uv_img[:, 0] < w) & (uv_img[:, 1] >= 0) & (uv_img[:, 1] < h)

    zbuf, ds = TB.triangle_depth_buffer(xy_v, v3d[:, 2], faces, h, w, ds=1)
    bu = np.clip((uv_img[:, 0] / ds).astype(int), 0, zbuf.shape[1] - 1)
    bv = np.clip((uv_img[:, 1] / ds).astype(int), 0, zbuf.shape[0] - 1)
    front = np.isfinite(zbuf[bv, bu]) & (np.abs(z - zbuf[bv, bu]) <= 0.018)

    pdir = p3d / (np.linalg.norm(p3d, axis=1, keepdims=True) + 1e-9)
    nrm = n3d / (np.linalg.norm(n3d, axis=1, keepdims=True) + 1e-9)
    facing = np.clip(-(nrm * pdir).sum(1), 0, 1)
    visible = in_img & front & (facing > 0.35)

    cols = TB.sample_bilinear(img, np.clip(uv_img, [0, 0], [w - 1, h - 1]))
    rgba = np.zeros((atlas_size, atlas_size, 4), np.uint8)
    yy, xx = np.where(valid)
    rgba[yy[visible], xx[visible], :3] = np.clip(cols[visible] * 255, 0, 255).astype(np.uint8)
    rgba[yy[visible], xx[visible], 3] = 255
    return rgba, {"view_observed_texels": int(visible.sum()), "view_observed_fraction": round(float(visible.mean()), 4)}


def rasterize_projective(img: np.ndarray, xy: np.ndarray, z: np.ndarray, faces: np.ndarray):
    """Rasterize the posed mesh using the source photo itself as a projective texture.

    This is the geometry/projection acceptance ceiling: if this overlay is visible, the mesh
    pose/shape/camera are wrong; if only the canonical-atlas overlay is visible, the bake/fusion
    texture is wrong.
    """
    h, w = img.shape[:2]
    color = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    depth = np.full((h, w), np.inf, np.float32)
    for tri in faces:
        pts = xy[tri].astype(np.float32)
        zz = z[tri].astype(np.float32)
        if not np.isfinite(pts).all() or not np.isfinite(zz).all() or np.any(zz <= 1e-4):
            continue
        if cv2.contourArea(pts) < 0.5:
            continue
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1).astype(np.float32)
        a, b, c = pts
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(float(den)) < 1e-6:
            continue
        vp = p - a
        b1 = (vp[:, 0] * v1[1] - vp[:, 1] * v1[0]) / den
        b2 = (v0[0] * vp[:, 1] - v0[1] * vp[:, 0]) / den
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-4) & (b1 >= -1e-4) & (b2 >= -1e-4)
        if not inside.any():
            continue
        pix = p[inside].astype(int)
        bary = np.stack([b0, b1, b2], 1)[inside]
        dz = (zz[None, :] * bary).sum(1)
        yy, xx = pix[:, 1], pix[:, 0]
        closer = dz < depth[yy, xx]
        if not closer.any():
            continue
        yy, xx, dz = yy[closer], xx[closer], dz[closer]
        depth[yy, xx] = dz
        color[yy, xx] = img[yy, xx]
        alpha[yy, xx] = 1.0
    if alpha.any():
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.65)
    return np.clip(color, 0, 255).astype(np.uint8), np.clip(alpha, 0, 1)


def overlay_one(est, img_path: Path, atlas: np.ndarray, vt: np.ndarray, fv: np.ndarray,
                fvt: np.ndarray, texel_v: np.ndarray, bary: np.ndarray, valid: np.ndarray,
                vt2v: np.ndarray, out_dir: Path):
    img, h, j2d = TB.posed_view(est, str(img_path))
    if h is None:
        return {"image": str(img_path), "ok": False, "reason": "no LHM person"}
    v3d = h["v3d"]
    fx, fy, cx, cy = TB.fit_pinhole(h["j3d"], j2d)
    uverts3 = v3d[vt2v]
    z = np.clip(uverts3[:, 2], 1e-5, None)
    xy = np.stack([fx * uverts3[:, 0] / z + cx, fy * uverts3[:, 1] / z + cy], 1)
    render, mask = rasterize_textured(img, xy, z, vt, fvt, atlas)
    import trimesh
    vn = np.asarray(trimesh.Trimesh(vertices=v3d, faces=fv, process=False).vertex_normals)
    view_atlas, view_stats = make_view_conditioned_atlas(
        img, v3d, vn, xy, fx, fy, cx, cy, texel_v, bary, valid, fv, atlas.shape[0])
    view_render, view_mask = rasterize_textured(img, xy, z, vt, fvt, view_atlas)
    proj_render, proj_mask = rasterize_projective(img, xy, z, fvt)
    # Photographic overlay: high alpha proves alignment; if it is wrong, it is obvious.
    comp = (render.astype(np.float32) * mask[:, :, None] * 0.88
            + img.astype(np.float32) * (1.0 - mask[:, :, None] * 0.88))
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    stem = img_path.parent.name + "_" + img_path.stem
    overlay_path = out_dir / f"{stem}_textured_overlay.png"
    mesh_path = out_dir / f"{stem}_mesh_only.png"
    proj_path = out_dir / f"{stem}_projective_overlay.png"
    view_path = out_dir / f"{stem}_view_conditioned_overlay.png"
    view_atlas_path = out_dir / f"{stem}_view_atlas.png"
    Image.fromarray(comp).save(overlay_path)
    Image.fromarray(render).save(mesh_path)
    Image.fromarray(view_atlas).save(view_atlas_path)
    view_comp = (view_render.astype(np.float32) * view_mask[:, :, None] * 0.88
                 + img.astype(np.float32) * (1.0 - view_mask[:, :, None] * 0.88))
    Image.fromarray(np.clip(view_comp, 0, 255).astype(np.uint8)).save(view_path)
    proj_comp = (proj_render.astype(np.float32) * proj_mask[:, :, None] * 0.88
                 + img.astype(np.float32) * (1.0 - proj_mask[:, :, None] * 0.88))
    Image.fromarray(np.clip(proj_comp, 0, 255).astype(np.uint8)).save(proj_path)
    covered = mask > 0.05
    diff = float(np.mean(np.abs(comp[covered].astype(np.float32) - img[covered].astype(np.float32)))) if covered.any() else 0.0
    view_covered = view_mask > 0.05
    view_diff = float(np.mean(np.abs(view_comp[view_covered].astype(np.float32) - img[view_covered].astype(np.float32)))) if view_covered.any() else 0.0
    proj_covered = proj_mask > 0.05
    proj_diff = float(np.mean(np.abs(proj_comp[proj_covered].astype(np.float32) - img[proj_covered].astype(np.float32)))) if proj_covered.any() else 0.0
    return {
        "image": str(img_path), "ok": True, "overlay": str(overlay_path), "mesh": str(mesh_path),
        "projective_overlay": str(proj_path), "view_conditioned_overlay": str(view_path),
        "view_atlas": str(view_atlas_path),
        "coverage_px": round(float(covered.mean()), 4), "mean_overlay_absdiff": round(diff, 2),
        "mean_view_conditioned_absdiff": round(view_diff, 2),
        "mean_projective_absdiff": round(proj_diff, 2),
        "focal": round(float((fx + fy) * 0.5), 2), **view_stats,
    }


def make_contact(rows: list[dict], out_path: Path):
    tiles = []
    for row in rows:
        if not row.get("ok"):
            continue
        src = Image.open(row["image"]).convert("RGB")
        mesh = Image.open(row["mesh"]).convert("RGB")
        proj = Image.open(row["projective_overlay"]).convert("RGB")
        view = Image.open(row["view_conditioned_overlay"]).convert("RGB")
        over = Image.open(row["overlay"]).convert("RGB")
        for im in (src, mesh, view, over):
            im.thumbnail((300, 390), Image.LANCZOS)
        proj.thumbnail((300, 390), Image.LANCZOS)
        tile = Image.new("RGB", (1550, 430), (20, 22, 24))
        d = ImageDraw.Draw(tile)
        x = 10
        for title, im in (("original", src), ("posed textured mesh", mesh),
                          ("projective ceiling", proj), ("view UV atlas overlay", view),
                          ("global atlas overlay", over)):
            tile.paste(im, (x + (300 - im.width) // 2, 32 + (390 - im.height) // 2))
            d.text((x, 8), title, font=_font(18), fill=(235, 238, 240))
            x += 310
        d.text((10, 408), Path(row["image"]).as_posix(), font=_font(13), fill=(185, 195, 205))
        tiles.append(tile)
    if not tiles:
        return
    sheet = Image.new("RGB", (1550, 430 * len(tiles)), (16, 18, 20))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += 430
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--bake", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--atlas-mode", choices=["auto", "observed", "full"], default="auto",
                    help="auto/observed uses atlas_observed_rgba.png when present; full renders atlas.png")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    bake_dir = Path(args.bake)
    atlas_path = bake_dir / "atlas_observed_rgba.png"
    if args.atlas_mode == "full":
        atlas = np.asarray(Image.open(bake_dir / "atlas.png").convert("RGB"))
        atlas_mode = "rgb_full"
    elif atlas_path.exists() and args.atlas_mode in ("auto", "observed"):
        atlas = np.asarray(Image.open(atlas_path).convert("RGBA"))
        atlas_mode = "observed_rgba"
    else:
        atlas = np.asarray(Image.open(bake_dir / "atlas.png").convert("RGB"))
        atlas_mode = "rgb_full"
    vt, fv, fvt = TB.load_uv_obj(TB.UV_OBJ)
    texel_v, bary = TB.rasterize_uv(vt, fvt, fv, atlas.shape[0])
    valid = texel_v[:, :, 0] >= 0
    vt2v = vt_to_vertex(fv, fvt, len(vt))
    est = A._estimator()

    rows = []
    for img_path in image_paths(Path(args.subject))[:args.limit]:
        print(f"overlay {img_path}", flush=True)
        try:
            rows.append(overlay_one(est, img_path, atlas, vt, fv, fvt, texel_v, bary, valid, vt2v, out))
        except Exception as e:
            rows.append({"image": str(img_path), "ok": False, "reason": repr(e)})
            print(f"  failed: {e}", flush=True)
    make_contact(rows, out / "TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png")
    report = {"atlas_mode": atlas_mode, "rows": rows}
    (out / "overlay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"OVERLAY_DONE {out / 'TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png'}")


if __name__ == "__main__":
    main()
