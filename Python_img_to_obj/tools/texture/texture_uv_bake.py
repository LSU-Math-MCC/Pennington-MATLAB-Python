"""Per-TEXEL UV-atlas texture bake -> pixel-resolution appearance on the canonical A-pose.

Inverse bake (atlas->image), which uses every relevant image pixel and avoids pytorch3d
camera-convention pitfalls by reusing our own fitted pinhole:

  Step 1 (once): rasterize the SMPL-X UV layout -> for each atlas texel, its face + bary.
  Step 2 (per view): texel 3D point/normal = bary-interp of POSED v3d/vn; project to image
     (fitted fx,fy,cx,cy); visible = front-facing AND passes a vertex z-buffer occlusion test;
     sample image color; accumulate per texel (quality-weighted) across all views.
  Step 3: atlas = weighted mean; hole-fill; apply to A-pose mesh (same UV) -> textured GLB.

This puts an individual vein/blemish/tattoo at its exact texel on the canonical surface, with
correct L/R + digit + face by SMPL-X topology. Occlusion via z-buffer (no back-bleed).

Usage (WSL lhm): python tools/texture/texture_uv_bake.py --subject <dir|img|glob> --out <dir> [--atlas 1024]
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
import glob
import json
import argparse

import numpy as np

# Register HEIC/AVIF openers so EVERY image loads (some .jpg are actually HEIC/AVIF -> PIL
# UnidentifiedImageError otherwise). Shared module: importing this fixes loading everywhere.
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except Exception:
    pass
try:
    import pillow_avif  # noqa: F401  (registers AVIF on import)
except Exception:
    pass

sys.path.insert(0, os.path.join(REPO, "tools"))
import lhm_anthropometry as A  # noqa: E402
from PIL import Image

UV_OBJ = os.path.join(A.HUMAN_MODELS, "smplx/smplx_uv/smplx_uv.obj")


def image_paths(subject):
    if any(ch in subject for ch in "*?[]"):
        return sorted(glob.glob(subject))
    if os.path.isdir(subject):
        return sorted(sum([glob.glob(os.path.join(subject, "**", e), recursive=True)
                           for e in ("*.jpg", "*.jpeg", "*.png", "*.webp")], []))
    return [subject]


def load_uv_obj(path):
    vt = []; fv = []; fvt = []
    with open(path) as f:
        for line in f:
            if line.startswith("vt "):
                _, u, v = line.split()[:3]
                vt.append((float(u), float(v)))
            elif line.startswith("f "):
                parts = line.split()[1:]
                vi = []; ti = []
                for p in parts:
                    a = p.split("/")
                    vi.append(int(a[0]) - 1)
                    ti.append(int(a[1]) - 1 if len(a) > 1 and a[1] else 0)
                # triangulate fan if quad
                for k in range(1, len(vi) - 1):
                    fv.append((vi[0], vi[k], vi[k + 1]))
                    fvt.append((ti[0], ti[k], ti[k + 1]))
    return np.array(vt, float), np.array(fv, np.int64), np.array(fvt, np.int64)


def rasterize_uv(vt, fvt, fv, A_size):
    """Per atlas texel -> (face_v3 indices, bary). Returns texel_v (A,A,3) int, bary (A,A,3)."""
    texel_v = np.full((A_size, A_size, 3), -1, np.int64)
    bary = np.zeros((A_size, A_size, 3), np.float32)
    # uv in [0,1]; atlas pixel: x=u*A, y=(1-v)*A
    P = vt.copy()
    P[:, 0] *= (A_size - 1)
    P[:, 1] = (1 - P[:, 1]) * (A_size - 1)
    for fi in range(fvt.shape[0]):
        t = P[fvt[fi]]                                   # 3x2 pixel coords
        minx, miny = np.floor(t.min(0)).astype(int)
        maxx, maxy = np.ceil(t.max(0)).astype(int)
        minx = max(minx, 0); miny = max(miny, 0)
        maxx = min(maxx, A_size - 1); maxy = min(maxy, A_size - 1)
        if maxx < minx or maxy < miny:
            continue
        xs, ys = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
        px = np.stack([xs.ravel(), ys.ravel()], 1).astype(float) + 0.5
        # barycentric
        d = t[1] - t[0]; e = t[2] - t[0]; den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-9:
            continue
        w = px - t[0]
        b1 = (w[:, 0] * e[1] - w[:, 1] * e[0]) / den
        b2 = (d[0] * w[:, 1] - d[1] * w[:, 0]) / den
        b0 = 1 - b1 - b2
        inside = (b0 >= -1e-4) & (b1 >= -1e-4) & (b2 >= -1e-4)
        if not inside.any():
            continue
        pix = px[inside].astype(int)
        texel_v[pix[:, 1], pix[:, 0]] = fv[fi]
        bary[pix[:, 1], pix[:, 0]] = np.stack([b0, b1, b2], 1)[inside]
    return texel_v, bary


def posed_view(est, img_path):
    img_np = np.asarray(Image.open(img_path).convert("RGB"))
    padded, ow, oh = est.img_center_padding(img_np)
    t, ann = est._preprocess(padded)
    K = est.get_camera_parameters()
    th = est.mhmr_model(t, is_training=False, nms_kernel_size=3, det_thresh=0.3,
                        K=K, idx=None, max_dist=None)
    if len(th) == 0:                                   # low-res/small subject: retry lower thresh
        th = est.mhmr_model(t, is_training=False, nms_kernel_size=3, det_thresh=0.1,
                            K=K, idx=None, max_dist=None)
    if len(th) == 0:
        return img_np, None, None
    # pick the LARGEST person (by 2D joint bbox) — robust to spurious small detections
    def _area(d):
        jj = d["j2d"].detach().cpu().numpy()
        return (jj[:, 0].max() - jj[:, 0].min()) * (jj[:, 1].max() - jj[:, 1].min())
    h = max(th, key=_area)
    pad_left, pad_top, scale = ann[0], ann[1], ann[2]
    j2d = h["j2d"].detach().cpu().numpy()
    j2d = (j2d - np.array([pad_left, pad_top])) / float(scale) - np.array([ow, oh])
    return img_np, {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
                    for k, v in h.items()}, j2d


def fit_pinhole(j3d, j2d):
    X, Y, Z = j3d[:, 0], j3d[:, 1], j3d[:, 2]
    good = (Z > 1e-3) & np.isfinite(j2d).all(1)
    fu, cu = np.linalg.lstsq(np.stack([X[good] / Z[good], np.ones(good.sum())], 1), j2d[good, 0], rcond=None)[0]
    fv, cv = np.linalg.lstsq(np.stack([Y[good] / Z[good], np.ones(good.sum())], 1), j2d[good, 1], rcond=None)[0]
    return float(fu), float(fv), float(cu), float(cv)


def vertex_depth_buffer(uv, z, H, W, ds=2):
    """Coarse z-buffer from projected vertices (min depth per cell) for occlusion test."""
    hb, wb = H // ds + 1, W // ds + 1
    buf = np.full((hb, wb), np.inf)
    u = np.clip((uv[:, 0] / ds).astype(int), 0, wb - 1)
    v = np.clip((uv[:, 1] / ds).astype(int), 0, hb - 1)
    for i in range(len(z)):
        if z[i] < buf[v[i], u[i]]:
            buf[v[i], u[i]] = z[i]
    return buf, ds


def triangle_depth_buffer(uv, z, faces, H, W, ds=1):
    """Rasterized triangle z-buffer in image space.

    Vertex-only depth tests miss most triangle interiors, which is exactly where
    occluded torso/limb texels can incorrectly sample foreground arms, clothes,
    hair, or background. This buffer is slower but gives the bake a real
    visibility test.
    """
    hb, wb = H // ds + 1, W // ds + 1
    pts = np.asarray(uv, np.float32) / float(ds)
    zz = np.asarray(z, np.float32)
    buf = np.full((hb, wb), np.inf, np.float32)
    for tri in np.asarray(faces, np.int64):
        p = pts[tri]
        tz = zz[tri]
        if not np.isfinite(p).all() or not np.isfinite(tz).all() or np.any(tz <= 1e-4):
            continue
        minx, miny = np.floor(p.min(0)).astype(int)
        maxx, maxy = np.ceil(p.max(0)).astype(int)
        minx, miny = max(minx, 0), max(miny, 0)
        maxx, maxy = min(maxx, wb - 1), min(maxy, hb - 1)
        if maxx <= minx or maxy <= miny:
            continue
        xs, ys = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
        q = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1).astype(np.float32)
        a, b, c = p
        v0 = b - a
        v1 = c - a
        den = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(float(den)) < 1e-8:
            continue
        vp = q - a
        b1 = (vp[:, 0] * v1[1] - vp[:, 1] * v1[0]) / den
        b2 = (v0[0] * vp[:, 1] - v0[1] * vp[:, 0]) / den
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-4) & (b1 >= -1e-4) & (b2 >= -1e-4)
        if not inside.any():
            continue
        pix = q[inside].astype(int)
        dz = b0[inside] * tz[0] + b1[inside] * tz[1] + b2[inside] * tz[2]
        yy = pix[:, 1]
        xx = pix[:, 0]
        closer = dz < buf[yy, xx]
        if closer.any():
            buf[yy[closer], xx[closer]] = dz[closer]
    return buf, ds


def sample_bilinear(img, uv):
    H, W = img.shape[:2]
    u = np.clip(uv[:, 0], 0, W - 1.001); v = np.clip(uv[:, 1], 0, H - 1.001)
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
    fx = (u - x0)[:, None]; fy = (v - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy) / 255.0


def main():
    import trimesh
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", type=int, default=2048)
    ap.add_argument("--betas", default=None, help="npy of contour-fitted/fused betas for the A-pose mesh")
    ap.add_argument("--skin-only", action="store_true",
                    help="SKIN-ONLY fusion: bake only texels projecting onto skin pixels (clothing"
                         " is non-body -> stays honestly unmapped). Body appearance, not garments.")
    ap.add_argument("--blend", action="store_true",
                    help="weighted multi-view blend (default = best-view-per-texel for MAX sharpness/detail)")
    ap.add_argument("--no-face-align", action="store_true",
                    help="disable G2 landmark-guided face alignment (default on)")
    ap.add_argument("--coherent-face", action="store_true",
                    help="opt into the old landmark-warped single-face overwrite. Default is off;"
                         " the raw skin-prior face is safer unless this passes overlay QA.")
    ap.add_argument("--no-coherent-face", action="store_true",
                    help="disable the special single-frontal-face overwrite. Useful for singleton"
                         " debugging and for cases where the warp corrupts visible face texels.")
    ap.add_argument("--skin-priority", choices=["off", "soft", "strong"], default="strong",
                    help="when multiple views observe a texel, prefer skin-colored pixels over"
                         " hair/clothes/occluders. strong is intended for body-skin texture maps.")
    ap.add_argument("--multi-face-fusion", action="store_true",
                    help="allow per-texel fusion inside the front-face mask. Default is a single"
                         " raw best face source to prevent duplicate eyes/mouths.")
    ap.add_argument("--face-repair", choices=["off", "face", "face-head"], default="face-head",
                    help="landmark-aware face atlas repair: off=strict observed only, face=repair"
                         " face/mouth holes, face-head=also neutral-fill unobserved head texels")
    ap.add_argument("--unmapped-fill", choices=["grey", "skin"], default="skin",
                    help="complete still-unmapped valid body texels after baking. grey keeps strict"
                         " missing-data visualization; skin makes a renderable avatar while"
                         " uv_report still records true observed coverage.")
    ap.add_argument("--face-occlusion-clean", choices=["off", "conservative", "aggressive"],
                    default="conservative",
                    help="clean hair/shadow streaks inside the face atlas. conservative only fixes"
                         " obvious non-feature dark streaks; aggressive also removes large dark"
                         " components and is intended for manual visual experiments.")
    ap.add_argument("--no-lr-symmetry", action="store_true",
                    help="disable filling unknown-side texels from their L-R mirror (default on)")
    ap.add_argument("--face-disp", default=None,
                    help="npy (10475,3) canonical head displacement -> deform SMPL-X head to the "
                         "detailed (DECA/FLAME) face so the projected face texture is sharp + undistorted")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    face_disp = np.load(args.face_disp).astype(np.float32) if (args.face_disp and os.path.exists(args.face_disp)) else None
    AS = args.atlas

    images = image_paths(args.subject)
    if not images:
        raise SystemExit(f"no images matched --subject {args.subject}")
    print("views:", len(images), "atlas:", AS)

    vt, fv, fvt = load_uv_obj(UV_OBJ)
    print("uv: vt", vt.shape, "faces", fv.shape)
    print("rasterizing UV layout (once)...")
    texel_v, bary = rasterize_uv(vt, fvt, fv, AS)
    valid = texel_v[:, :, 0] >= 0
    print("atlas texels covered:", int(valid.sum()), f"({valid.mean():.1%})")
    tv = texel_v[valid]; tb = bary[valid]                # (T,3),(T,3)

    est = A._estimator()
    consistent_betas = None
    if args.betas and os.path.exists(args.betas):
        consistent_betas = np.load(args.betas)[:10]      # register all views to this shape
        print(f"registering all views to consistent betas {args.betas}")
    nV = 10475
    NT = int(valid.sum())
    # Semantic atlas masks. The old coherent-face path used every texel above the neck, which
    # drags mouth/eye/hair pixels across scalp, ears, and neck. Keep the broad head mask only for
    # diagnostics/fill suppression; use a tighter front-face mask for the single-view face pass.
    mask_betas = consistent_betas if consistent_betas is not None else np.zeros(10)
    head_vert = _head_vertex_mask(mask_betas)
    face_vert = _face_core_vertex_mask(mask_betas)
    head_weight = (head_vert[tv] * tb).sum(1)
    head_texel = head_weight > 0.33
    head_fill_texel = head_weight > 0.90
    face_texel = (face_vert[tv] * tb).sum(1) > 0.45
    print(f"head texels: {int(head_texel.sum())} face-core texels: {int(face_texel.sum())}")
    coherent_face_enabled = bool(args.coherent_face and not args.no_coherent_face)
    head_best_q = -1.0; head_best_cols = None; head_best_w = None; head_best_skin = None
    head_best_view = None
    atlas_sum = np.zeros((NT, 3)); atlas_w = np.zeros(NT)
    # BEST-VIEW per texel (avoids cross-view blending -> no limb/torso drift, sharper):
    atlas_bestcol = np.full((NT, 3), 0.5); atlas_bestw = np.zeros(NT)
    atlas_bestscore = np.zeros(NT)
    atlas_bestskin = np.zeros(NT, bool)
    vcol_sum = np.zeros((nV, 3)); vcol_w = np.zeros(nV)      # per-vertex (for symmetry fill)
    betas_list = []
    mouth_reject_total = 0
    for img_path in images:
        img, h, j2d = posed_view(est, img_path)
        if h is None:
            print("  skip", os.path.basename(img_path)); continue
        betas_list.append(h["shape"].reshape(-1))
        # Register every view to ONE consistent canonical body (shared betas, re-posed to
        # this view) so the SAME UV texel maps to the SAME geometry across views -> sharp,
        # registered texture (not per-view-shape ghosting). Falls back to per-view v3d.
        if consistent_betas is not None:
            import torch
            go = h["rotvec"][0:1].reshape(-1).astype(np.float32)
            bp = h["rotvec"][1:22].reshape(-1).astype(np.float32)
            vlocal = _forward_consistent(consistent_betas, go, bp)
            if face_disp is not None:                      # deform head to detailed (DECA/FLAME) face (rotated into this view)
                import cv2
                Rgo = cv2.Rodrigues(go.astype(np.float64))[0]
                vlocal = vlocal + (Rgo @ face_disp.T).T
            offset = h["v3d"].mean(0) - vlocal.mean(0)
            v3d = vlocal + offset
        else:
            v3d = h["v3d"]
        fx, fy, cx, cy = fit_pinhole(h["j3d"], j2d)
        m = trimesh.Trimesh(vertices=v3d, faces=fv, process=False)
        vn = np.asarray(m.vertex_normals)
        # texel 3D pos + normal via bary
        p3d = (v3d[tv] * tb[:, :, None]).sum(1)
        n3d = (vn[tv] * tb[:, :, None]).sum(1)
        Z = np.clip(p3d[:, 2], 1e-3, None)
        uv = np.stack([fx * p3d[:, 0] / Z + cx, fy * p3d[:, 1] / Z + cy], 1)
        pdir = p3d / (np.linalg.norm(p3d, axis=1, keepdims=True) + 1e-9)
        nrm = n3d / (np.linalg.norm(n3d, axis=1, keepdims=True) + 1e-9)
        facing = np.clip(-(nrm * pdir).sum(1), 0, 1)
        H, W = img.shape[:2]
        inimg = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        # occlusion: texel depth must match the rasterized nearest mesh surface at its pixel.
        # A vertex-only z-buffer lets hidden triangle interiors leak through, which paints
        # crossed arms, bikini edges, hair, and background onto unrelated canonical texels.
        vuv = np.stack([fx * v3d[:, 0] / np.clip(v3d[:, 2], 1e-3, None) + cx,
                        fy * v3d[:, 1] / np.clip(v3d[:, 2], 1e-3, None) + cy], 1)
        buf, ds = triangle_depth_buffer(vuv, v3d[:, 2], fv, H, W, ds=1)
        bu = np.clip((uv[:, 0] / ds).astype(int), 0, buf.shape[1] - 1)
        bv = np.clip((uv[:, 1] / ds).astype(int), 0, buf.shape[0] - 1)
        z_near = buf[bv, bu]
        front = np.isfinite(z_near) & (np.abs(Z - z_near) <= 0.018)
        # BACKGROUND FILTER: reject texels whose pixel falls outside the (eroded) person mask
        pm = person_mask(img)
        sm = skin_mask(img)
        feature_px = _projected_face_feature_mask(uv, j2d, H, W)
        # drop the streak-prone SILHOUETTE EDGE band: texels projecting near the mask boundary
        # are grazing/mixed pixels that smear into vertical stripes. Erode ~4px for sampling.
        import cv2 as _cv2
        pm_core = _cv2.erode(pm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1); vi2 = np.clip(uv[:, 1].astype(int), 0, H - 1)
        in_person = pm_core[vi2, ui]
        if args.skin_only:
            in_person = in_person & (sm[vi2, ui] | (face_texel & feature_px))
        # Use mask-distance as a soft confidence. Near-silhouette mixed pixels are the classic
        # source of body streaks; this keeps interior pixels dominant in multi-view fusion.
        pm_dist = _cv2.distanceTransform(pm.astype(np.uint8), _cv2.DIST_L2, 3)
        edge_conf = np.clip(pm_dist[vi2, ui] / 8.0, 0.0, 1.0)
        skin_px = sm[vi2, ui]
        # Reject pixels that land in the projected inner-mouth polygon. Teeth/tongue/cavity are
        # real photo pixels, but they do not belong on the closed SMPL-X face surface.
        mouth_reject = _projected_inner_mouth_mask(uv, j2d, H, W)
        # quality weight: prefer square-on views, reject grazing body samples more aggressively,
        # and require occlusion + in-image + INSIDE PERSON (no background bleed).
        facing_floor = np.where(face_texel, 0.45, 0.60)
        qual = np.where(facing >= facing_floor, (facing ** 2) * edge_conf, 0.0)
        w = qual * inimg * front * in_person
        mouth_reject_total += int((mouth_reject & inimg).sum())
        w[mouth_reject] = 0.0
        # Sample the RAW photo pixels — keep ALL texture/colour information. Lighting is
        # corrected in POST on the assembled mesh texture (per-view delight discarded detail).
        cols = sample_bilinear(img, np.clip(uv, [0, 0], [W - 1, H - 1]))
        atlas_sum += cols * w[:, None]; atlas_w += w
        # Per-texel view choice. For body/face texture we prefer skin over hair/clothes when
        # the same canonical texel has competing observations. This prevents hair from winning
        # the scalp/face and garments from winning skin regions just because the view is square-on.
        if args.skin_priority == "off":
            score = w
        else:
            skin_gain = 2.0 if args.skin_priority == "soft" else 5.0
            score = w * (1.0 + skin_gain * skin_px.astype(np.float32))
            # On the face/head, skin beats non-skin decisively; old hair/clothing wins here are
            # the source of the grotesque fused faces.
            score = np.where(head_texel & skin_px, score * 1.8, score)
            score = np.where(head_texel & (~skin_px), score * 0.35, score)
        better = score > atlas_bestscore
        atlas_bestcol[better] = cols[better]
        atlas_bestw[better] = w[better]
        atlas_bestscore[better] = score[better]
        atlas_bestskin[better] = skin_px[better] & (w[better] > 1e-6)
        # Track the single best raw face source. This is separate from the old optional warp:
        # it prevents the fused face UV island from becoming a quilt of eyes/mouths from
        # several photos.
        face_source_score = w * face_texel
        if args.skin_priority != "off":
            face_source_score = face_source_score * (1.0 + 2.5 * skin_px.astype(np.float32))
        hq = float(face_source_score.sum())
        if hq > head_best_q:
            head_best_q = hq
            head_best_cols = cols.copy()
            head_best_w = w.copy()
            head_best_skin = skin_px.copy()
            head_best_view = dict(img=img, uv=uv.copy(), j2d=np.asarray(j2d), W=W, H=H,
                                  mouth_reject=mouth_reject.copy(),
                                  expression=np.asarray(h.get("expression")) if h.get("expression") is not None else None,
                                  jaw=np.asarray(h["rotvec"][22]) if np.asarray(h["rotvec"]).shape[0] > 22 else None)
        # scatter to per-vertex (bary-weighted) for symmetry back-fill of unseen regions
        for k in range(3):
            wk = w * tb[:, k]
            np.add.at(vcol_sum, tv[:, k], cols * wk[:, None])
            np.add.at(vcol_w, tv[:, k], wk)
        print(f"  {os.path.basename(img_path)}: {int((w>0.05).sum())} texels written, fx={fx:.0f}")

    seen = atlas_bestw > 1e-3
    # FUSION: BEST-VIEW-per-texel by default -> each texel is the REAL pixel from its sharpest,
    # most front-facing view (exact source detail, no blur). --blend opts into a weighted mean
    # (smoother seams but softer). Faces/fine detail need the sharp best-view.
    tcol = atlas_bestcol.copy()
    if args.blend:
        blended = atlas_sum / np.maximum(atlas_w, 1e-6)[:, None]
        tcol[seen] = blended[seen]
    # FACE ISLAND: by default, the front face comes from one raw source view. Per-texel
    # multi-view face fusion creates duplicate eyes/mouths whenever source poses/expressions
    # differ. Holes are handled by the face repair stage, not by mixing another face on top.
    face_single_source_texels = 0
    if head_best_cols is not None and not args.blend and not args.multi_face_fusion:
        fh = face_texel & (head_best_w > 1e-3)
        if args.skin_priority != "off" and head_best_skin is not None:
            # Non-skin facial features from the selected face (eyes/lips/brows) are still real
            # and should remain; this only blocks obvious hair/clothes from replacing skin.
            skin_or_feature = head_best_skin | _projected_face_feature_mask(
                head_best_view["uv"], head_best_view["j2d"], head_best_view["H"], head_best_view["W"])
            fh = fh & skin_or_feature
        tcol[fh] = head_best_cols[fh]
        face_unowned = face_texel & ~fh
        tcol[face_unowned] = 0.5
        atlas_bestw[face_unowned] = 0.0
        seen = atlas_bestw > 1e-3
        seen[fh] = True
        face_single_source_texels = int(fh.sum())
        print(f"single-source face: {face_single_source_texels} face texels from best view (q={head_best_q:.0f})")
    # Optional legacy coherent face: overwrite face texels with a warped single view. Kept only
    # for experiments because it can misalign eyes on some subjects.
    if head_best_cols is not None and not args.blend and coherent_face_enabled:
        fh = face_texel & (head_best_w > 1e-3)
        cols_face = head_best_cols
        # G2: landmark-align the face — warp the chosen view so its REAL eyes/nose/mouth land on
        # the mesh's projected face landmarks, then re-sample (photo features map to right places).
        if not args.no_face_align:
            warped = face_align_warp(head_best_view["img"], head_best_view["j2d"])
            if warped is not None:
                bv = head_best_view
                head_best_view["aligned_img"] = warped
                cols_face = sample_bilinear(warped, np.clip(bv["uv"], [0, 0], [bv["W"] - 1, bv["H"] - 1]))
                print("face landmark-aligned (MediaPipe -> SMPL-X face joints)")
        tcol[fh] = cols_face[fh]
        print(f"coherent face: {int(fh.sum())} face texels from best frontal view (q={head_best_q:.0f})")
    # L-R SYMMETRY FILL (valid: the body is left-right symmetric; front-back is NOT). Fill an
    # UNMAPPED texel from its MIRROR ONLY IF the mirror side is observed -> never overwrites real
    # pixels, only completes unknown sides. (front/back is never mirrored — that would be insane.)
    lr_filled = 0
    if not args.no_lr_symmetry:
        flip = _load_flip(nV)
        if flip is not None:
            cf, bc = flip
            vmean = vcol_sum / np.maximum(vcol_w, 1e-6)[:, None]
            mvcol = (vmean[cf] * bc[:, :, None]).sum(1)   # per-vertex MIRROR colour (bary)
            mvobs = (vcol_w[cf] > 1e-3).all(1)            # per-vertex: mirror side observed
            mcol = (mvcol[tv] * tb[:, :, None]).sum(1)    # texel mirror colour (bary over its verts)
            mobs = mvobs[tv].all(1)
            fillm = (~seen) & mobs & (~head_texel)          # never mirror-fill face/head detail
            tcol[fillm] = mcol[fillm]; lr_filled = int(fillm.sum())
            print(f"L-R symmetry filled {lr_filled} unknown-side texels (mirror observed)")
        else:
            print("L-R symmetry: no flip correspondences -> skipped")
    # HONEST MAPPING otherwise: no front-back symmetry, no inpaint, no fabrication. Unmapped
    # texels (neither side observed) stay neutral grey — partial coverage is honest.
    atlas = np.full((AS, AS, 3), 0.5)
    atlas[valid] = tcol
    cov = float(seen.mean())
    atlas_img = (np.clip(atlas, 0, 1) * 255).astype(np.uint8)
    face_atlas = np.zeros((AS, AS), bool); face_atlas[valid] = face_texel
    head_atlas = np.zeros((AS, AS), bool); head_atlas[valid] = head_fill_texel
    seen_atlas = np.zeros((AS, AS), bool); seen_atlas[valid] = seen
    mouth_atlas = np.zeros((AS, AS), bool)
    feature_atlas = np.zeros((AS, AS), bool)
    if head_best_view is not None and head_best_view.get("mouth_reject") is not None:
        mouth_atlas[valid] = face_texel & head_best_view["mouth_reject"]
        feature_atlas[valid] = face_texel & _projected_face_feature_mask(
            head_best_view["uv"], head_best_view["j2d"], head_best_view["H"], head_best_view["W"])
    face_repair = {"filled_texels": 0, "mouth_texels": 0, "occlusion_texels": 0,
                   "neutralized_texels": 0, "head_texels": 0}
    if args.face_repair != "off":
        repair_head = head_atlas if args.face_repair == "face-head" else np.zeros_like(head_atlas)
        atlas_img, face_repair = _repair_face_atlas(atlas_img, face_atlas, repair_head, seen_atlas,
                                                    mouth_atlas, feature_atlas, head_best_view,
                                                    args.face_occlusion_clean)
    if (face_repair["filled_texels"] or face_repair["mouth_texels"] or
            face_repair["occlusion_texels"] or face_repair["neutralized_texels"] or
            face_repair["head_texels"]):
        print("face repaired: "
              f"{face_repair['filled_texels']} holes + {face_repair['mouth_texels']} mouth texels "
              f"+ {face_repair['occlusion_texels']} occlusion texels "
              f"+ {face_repair['neutralized_texels']} neutralized texels "
              f"+ {face_repair['head_texels']} head texels")
    valid_atlas = np.zeros((AS, AS), bool); valid_atlas[valid] = True
    completion = {"near_filled_texels": 0, "skin_filled_texels": 0}
    if args.unmapped_fill == "skin":
        atlas_img, completion = _complete_unmapped_atlas(atlas_img, valid_atlas, seen_atlas,
                                                         face_atlas | head_atlas | mouth_atlas)
        if completion["near_filled_texels"] or completion["skin_filled_texels"]:
            print("texture completed: "
                  f"{completion['near_filled_texels']} near-neighbor + "
                  f"{completion['skin_filled_texels']} skin-fill texels")
    # If --unmapped-fill grey, non-face unmapped texels stay neutral grey so partial coverage is visible.
    Image.fromarray(atlas_img).save(os.path.join(args.out, "atlas.png"))
    observed_rgba = np.dstack([atlas_img, (seen_atlas.astype(np.uint8) * 255)])
    Image.fromarray(observed_rgba).save(os.path.join(args.out, "atlas_observed_rgba.png"))
    Image.fromarray((seen_atlas.astype(np.uint8) * 255)).save(os.path.join(args.out, "atlas_observed_mask.png"))

    # K-NRM: tangent-space normal map from albedo high-frequency luminance -> micro-relief
    # (veins/blemishes catch light) without changing geometry. (Sapiens-normal = refine.)
    normal_img = _detail_normal_map(atlas_img, strength=3.0)
    Image.fromarray(normal_img).save(os.path.join(args.out, "atlas_normal.png"))

    # apply to A-pose mesh with UV (albedo + normal map via PBR material)
    if args.betas and os.path.exists(args.betas):
        betas = np.load(args.betas)[:10]                 # contour-fitted/fused shape
        print(f"using fitted betas from {args.betas}")
    else:
        betas = np.mean(betas_list, 0) if betas_list else np.zeros(10)
    np.save(os.path.join(args.out, "smplx_betas.npy"), np.asarray(betas, np.float32))
    gender, _ = A.estimate_gender(images)
    # NOTE: tried transferring the view's jaw+expression to the displayed head (G3a) — it made
    # the mouth WORSE (Multi-HMR jaw-open -> dark mouth gash on the projected texture). Reverted:
    # neutral face geometry + landmark-aligned texture (G2) is the best SMPL-X-bounded result.
    av, aj, af, named = A.smplx_apose(betas, gender=gender, arm_deg=45.0)
    if face_disp is not None:                              # canonical head -> detailed (DECA/FLAME) face shape
        av = np.asarray(av) + face_disp
    uverts, ufaces, uuv = build_uv_mesh(av, fv, fvt, vt)   # seam-split -> clean seams
    try:
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(atlas_img),
            normalTexture=Image.fromarray(normal_img),
            roughnessFactor=0.6, metallicFactor=0.0)
        visual = trimesh.visual.TextureVisuals(uv=uuv, material=mat)
    except Exception:
        visual = trimesh.visual.TextureVisuals(uv=uuv, image=Image.fromarray(atlas_img))
    amesh = trimesh.Trimesh(vertices=uverts, faces=ufaces, visual=visual, process=False)
    amesh.export(os.path.join(args.out, "apose_textured_uv.glb"))
    json.dump({"coverage_texels": round(cov, 4), "atlas": AS, "views": len(images),
               "gender": gender, "head_texels": int(head_texel.sum()),
               "face_core_texels": int(face_texel.sum()),
               "projected_mouth_reject_texels": int(mouth_reject_total),
               "coherent_face_enabled": coherent_face_enabled,
               "multi_face_fusion": bool(args.multi_face_fusion),
               "single_source_face_texels": int(face_single_source_texels),
               "skin_priority": args.skin_priority,
               "skin_priority_texels": int((seen & atlas_bestskin).sum()),
               "face_repair_mode": args.face_repair,
               "face_occlusion_clean": args.face_occlusion_clean,
               "face_repair_filled_texels": int(face_repair["filled_texels"]),
               "face_repair_mouth_texels": int(face_repair["mouth_texels"]),
               "face_occlusion_repair_texels": int(face_repair["occlusion_texels"]),
               "face_neutralized_texels": int(face_repair["neutralized_texels"]),
               "head_repair_filled_texels": int(face_repair["head_texels"]),
               "unmapped_fill_mode": args.unmapped_fill,
               "near_completion_texels": int(completion["near_filled_texels"]),
               "skin_completion_texels": int(completion["skin_filled_texels"])},
              open(os.path.join(args.out, "uv_report.json"), "w"), indent=2)
    print(f"UVBAKE_OK coverage={cov:.1%} atlas={AS} gender={gender}")


_SEG = {}


_HEADM = {}


_FACEM = {}


def _smplx_neutral_vertices_joints(betas):
    import torch, smplx
    key = tuple(np.round(np.asarray(betas)[:10].astype(np.float32), 3))
    cache = _FACEM.setdefault("_neutral_cache", {})
    if key not in cache:
        model = smplx.create(A.HUMAN_MODELS, model_type="smplx", gender="neutral", num_betas=10)
        with torch.no_grad():
            out = model(betas=torch.tensor(np.asarray(betas)[:10], dtype=torch.float32).unsqueeze(0))
        cache[key] = (out.vertices[0].numpy(), out.joints[0].numpy(), model.faces.astype(np.int64))
    return cache[key]


def _head_vertex_mask(betas):
    """Boolean (10475,) mask of SMPL-X head/face vertices = vertices above the neck joint."""
    key = tuple(np.round(np.asarray(betas)[:10].astype(np.float32), 3))
    if key not in _HEADM:
        v, j, _ = _smplx_neutral_vertices_joints(betas)
        neck_y = j[12, 1]                                   # SMPL-X neck joint
        _HEADM[key] = v[:, 1] > neck_y                      # above neck = head/face
    return _HEADM[key]


def _face_core_vertex_mask(betas):
    """Tight front-face mask: cheeks/eyes/nose/lips, excluding scalp, ears, neck, and back head."""
    key = tuple(np.round(np.asarray(betas)[:10].astype(np.float32), 3))
    if key not in _FACEM:
        import trimesh
        v, j, faces = _smplx_neutral_vertices_joints(betas)
        head = _head_vertex_mask(betas)
        lm = j[76:127]                                      # SMPL-X dlib-17..67 face landmarks
        x0, y0 = lm[:, :2].min(0); x1, y1 = lm[:, :2].max(0)
        w = max(float(x1 - x0), 1e-3)
        h = max(float(y1 - y0), 1e-3)
        z_front = np.percentile(v[head, 2], 52) if head.any() else np.percentile(v[:, 2], 75)
        vn = trimesh.Trimesh(v, faces, process=False).vertex_normals
        in_face_box = (
            (v[:, 0] >= x0 - 0.42 * w) & (v[:, 0] <= x1 + 0.42 * w) &
            (v[:, 1] >= y0 - 0.48 * h) & (v[:, 1] <= y1 + 0.70 * h)
        )
        front_surface = (v[:, 2] >= z_front) & (vn[:, 2] > -0.15)
        _FACEM[key] = head & in_face_box & front_surface
    return _FACEM[key]


def _projected_inner_mouth_mask(uv, j2d, H, W, pad_px=3):
    """Texel projections that land inside the projected inner-mouth polygon."""
    if j2d is None:
        return np.zeros(len(uv), bool)
    j2d = np.asarray(j2d)
    if j2d.ndim != 2 or j2d.shape[0] < 127:
        return np.zeros(len(uv), bool)
    inner = j2d[[119, 120, 121, 122, 123, 124, 125, 126]]
    if not np.isfinite(inner).all():
        return np.zeros(len(uv), bool)
    # Ignore degenerate / closed-mouth polygons; lips still bake, cavities do not.
    if np.ptp(inner[:, 0]) < 2.0 or np.ptp(inner[:, 1]) < 2.0:
        return np.zeros(len(uv), bool)
    try:
        import cv2
        mask = np.zeros((H, W), np.uint8)
        poly = np.round(inner).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1)
        if pad_px > 0:
            k = np.ones((pad_px * 2 + 1, pad_px * 2 + 1), np.uint8)
            mask = cv2.dilate(mask, k, iterations=1)
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
        vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
        return mask[vi, ui].astype(bool)
    except Exception:
        return np.zeros(len(uv), bool)


def _projected_face_feature_mask(uv, j2d, H, W):
    """Texels landing on semantic facial features to protect during occlusion cleanup."""
    if j2d is None:
        return np.zeros(len(uv), bool)
    j2d = np.asarray(j2d)
    if j2d.ndim != 2 or j2d.shape[0] < 127:
        return np.zeros(len(uv), bool)
    try:
        import cv2
        mask = np.zeros((H, W), np.uint8)

        def pts(dlib_ids):
            idx = [76 + (k - 17) for k in dlib_ids]
            p = j2d[idx]
            return p[np.isfinite(p).all(1)]

        for group, pad in [
            (range(17, 27), 5),   # eyebrows
            (range(36, 42), 7),   # right eye
            (range(42, 48), 7),   # left eye
            (range(48, 68), 7),   # lips / mouth
            (range(30, 36), 4),   # nose lower contour
        ]:
            p = pts(group)
            if len(p) >= 3:
                hull = cv2.convexHull(np.round(p).astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 1)
                if pad:
                    k = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
                    mask = cv2.dilate(mask, k, iterations=1)
            elif len(p):
                for x, y in np.round(p).astype(int):
                    cv2.circle(mask, (int(x), int(y)), pad, 1, -1)
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
        vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
        return mask[vi, ui].astype(bool)
    except Exception:
        return np.zeros(len(uv), bool)


def _lip_fill_color(view):
    """Robust lip color from the aligned best face view; avoids teeth/mouth cavity pixels."""
    if not view:
        return None
    img = np.asarray(view.get("aligned_img", view.get("img")))
    j2d = np.asarray(view.get("j2d")) if view.get("j2d") is not None else None
    if img is None or j2d is None or j2d.ndim != 2 or j2d.shape[0] < 127:
        return None
    try:
        import cv2
        H, W = img.shape[:2]
        outer = np.round(j2d[107:119]).astype(np.int32)      # dlib mouth outer loop 48..59
        inner = np.round(j2d[119:127]).astype(np.int32)      # dlib inner mouth 60..67
        lip = np.zeros((H, W), np.uint8)
        cv2.fillPoly(lip, [outer], 1)
        cv2.fillPoly(lip, [inner], 0)
        lip = cv2.dilate(lip, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        if lip.sum() < 10:
            return None
        pix = img[lip].astype(np.float32)
        # Drop teeth/highlights and very dark shadow; keep the robust central lip tone.
        lum = pix.mean(1)
        lo, hi = np.percentile(lum, [12, 88])
        pix = pix[(lum >= lo) & (lum <= hi)]
        if len(pix) < 8:
            return None
        return np.median(pix, axis=0)
    except Exception:
        return None


def _repair_face_atlas(atlas_img, face_mask, head_mask, observed_mask, mouth_mask, feature_mask,
                       best_view, occlusion_clean="conservative"):
    """Landmark-aware face-only repair for texels we intentionally did not trust.

    This is deliberately scoped to the front-face mask. Body/neck/scalp still stay honest unless
    other pipeline steps fill them, but the face should not ship with grey holes.
    """
    stats = {"filled_texels": 0, "mouth_texels": 0, "occlusion_texels": 0,
             "neutralized_texels": 0, "head_texels": 0}
    if atlas_img is None or not np.asarray(face_mask).any():
        return atlas_img, stats
    out = atlas_img.copy()
    mx = out.max(2); mn = out.min(2)
    grey = (np.abs(out.astype(np.int16) - 128).sum(2) < 60) & ((mx - mn) < 18)
    holes = face_mask & ((~observed_mask) | grey)

    mouth_fill = holes & mouth_mask
    lip = _lip_fill_color(best_view)
    if mouth_fill.any():
        face_obs = face_mask & observed_mask & (~grey) & (~mouth_mask)
        if lip is None and face_obs.sum() > 20:
            lip = np.median(out[face_obs].astype(np.float32), axis=0) * np.array([0.96, 0.82, 0.82])
        if lip is None:
            lip = np.array([155.0, 82.0, 78.0])
        out[mouth_fill] = np.clip(lip * np.array([0.92, 0.72, 0.72]), 0, 255).astype(np.uint8)
        holes = holes & (~mouth_fill)
        stats["mouth_texels"] = int(mouth_fill.sum())

    if not holes.any():
        return out, stats
    try:
        import cv2
        from scipy.ndimage import binary_opening
        # Remove isolated one-pixel noise from the repair mask; inpaint the actual missing chunks.
        repair = binary_opening(holes, iterations=1) | (holes & grey)
        if repair.sum() == 0:
            return out, stats
        ys, xs = np.where(face_mask)
        pad = 24
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, out.shape[0])
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, out.shape[1])
        crop = out[y0:y1, x0:x1].copy()
        mask = (repair[y0:y1, x0:x1] * 255).astype(np.uint8)
        local_face = face_mask[y0:y1, x0:x1]
        local_obs = local_face & (mask == 0)
        if local_obs.sum() > 20:
            med = np.median(crop[local_obs].astype(np.float32), axis=0).astype(np.uint8)
            crop[~local_face] = med                         # keep grey outside face from bleeding in
        fixed = cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)
        dest = out[y0:y1, x0:x1]
        dest[mask > 0] = fixed[mask > 0]
        out[y0:y1, x0:x1] = dest
        stats["filled_texels"] = int((mask > 0).sum())
    except Exception:
        face_obs = face_mask & observed_mask & (~grey)
        if face_obs.sum() > 20:
            out[holes] = np.median(out[face_obs].astype(np.float32), axis=0).astype(np.uint8)
            stats["filled_texels"] = int(holes.sum())

    # Clean hair/shadow streaks that are inside the face but outside protected features.
    rgb = out.astype(np.float32)
    mx = rgb.max(2); mn = rgb.min(2)
    grey = (np.abs(rgb - 128).sum(2) < 60) & ((mx - mn) < 18)
    skin_src = face_mask & (~grey) & (~mouth_mask) & (~feature_mask) & (rgb.sum(2) > 330)
    if occlusion_clean != "off" and skin_src.sum() > 50:
        skin = np.median(rgb[skin_src], axis=0)
        lum = rgb.mean(2)
        skin_lum = float(np.mean(skin))
        dark_occ = (face_mask & observed_mask & (~mouth_mask) & (~feature_mask) &
                    (lum < skin_lum * 0.58) & ((mx - mn) < 125))
        if dark_occ.sum() > 8:
            try:
                import cv2
                from scipy.ndimage import binary_opening, binary_dilation
                repair = binary_dilation(binary_opening(dark_occ, iterations=1), iterations=1)
                ys, xs = np.where(face_mask)
                pad = 20
                y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, out.shape[0])
                x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, out.shape[1])
                crop = out[y0:y1, x0:x1].copy()
                mask = (repair[y0:y1, x0:x1] * 255).astype(np.uint8)
                crop[~face_mask[y0:y1, x0:x1]] = np.clip(skin, 0, 255).astype(np.uint8)
                fixed = cv2.inpaint(crop, mask, 4, cv2.INPAINT_TELEA)
                dest = out[y0:y1, x0:x1]
                dest[mask > 0] = fixed[mask > 0]
                out[y0:y1, x0:x1] = dest
                stats["occlusion_texels"] = int((mask > 0).sum())
            except Exception:
                out[dark_occ] = np.clip(skin, 0, 255).astype(np.uint8)
                stats["occlusion_texels"] = int(dark_occ.sum())

        if occlusion_clean == "aggressive":
            # Experimental: if a face is heavily occluded, remove large dark components that
            # conservative feature protection would preserve. This can over-inpaint bad sources,
            # so it is opt-in and excluded from the default benchmark path.
            rgb = out.astype(np.float32)
            lum = rgb.mean(2)
            mx = rgb.max(2); mn = rgb.min(2)
            dark_all = face_mask & (~mouth_mask) & (lum < skin_lum * 0.70) & ((mx - mn) < 160)
            try:
                import cv2
                n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(dark_all.astype(np.uint8), 8)
                face_area = max(int(face_mask.sum()), 1)
                large = np.zeros_like(dark_all)
                for i in range(1, n):
                    area = int(cc_stats[i, cv2.CC_STAT_AREA])
                    w = int(cc_stats[i, cv2.CC_STAT_WIDTH])
                    h = int(cc_stats[i, cv2.CC_STAT_HEIGHT])
                    if area > max(35, 0.006 * face_area) or max(w, h) > 0.18 * np.sqrt(face_area):
                        large |= labels == i
                large &= ~mouth_mask
                if large.any():
                    from scipy.ndimage import binary_dilation
                    repair = binary_dilation(large, iterations=1)
                    ys, xs = np.where(face_mask)
                    pad = 20
                    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, out.shape[0])
                    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, out.shape[1])
                    crop = out[y0:y1, x0:x1].copy()
                    mask = (repair[y0:y1, x0:x1] * 255).astype(np.uint8)
                    crop[~face_mask[y0:y1, x0:x1]] = np.clip(skin, 0, 255).astype(np.uint8)
                    fixed = cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)
                    dest = out[y0:y1, x0:x1]
                    dest[mask > 0] = fixed[mask > 0]
                    out[y0:y1, x0:x1] = dest
                    stats["occlusion_texels"] += int((mask > 0).sum())
            except Exception:
                pass

    # Head-only repair: fill unmapped scalp/side-head/upper-neck texels with a calm skin tone so
    # the repaired face is not framed by grey mesh. This does not alter observed face pixels.
    mx = out.max(2); mn = out.min(2)
    grey = (np.abs(out.astype(np.int16) - 128).sum(2) < 60) & ((mx - mn) < 18)
    head_holes = head_mask & (~face_mask) & ((~observed_mask) | grey)
    if head_holes.any():
        rgb = out.astype(np.float32)
        bright = rgb.sum(2) > 330
        not_blue = ~((rgb[:, :, 2] > rgb[:, :, 0] + 18) & (rgb[:, :, 2] > rgb[:, :, 1] + 18))
        reddish = (rgb[:, :, 0] > rgb[:, :, 2] + 6) & (rgb[:, :, 0] > rgb[:, :, 1] - 28)
        not_extreme = (rgb.max(2) - rgb.min(2)) < 135
        skin_src = ((face_mask | head_mask) & observed_mask & (~grey) & (~mouth_mask) &
                    bright & not_blue & reddish & not_extreme)
        if skin_src.sum() > 30:
            skin = np.median(out[skin_src].astype(np.float32), axis=0)
        else:
            skin = np.array([202.0, 158.0, 138.0])
        out[head_holes] = np.clip(skin, 0, 255).astype(np.uint8)
        stats["head_texels"] = int(head_holes.sum())
    return out, stats


def _skin_fill_color(atlas_img, valid_mask, protected_mask):
    rgb = atlas_img.astype(np.float32)
    mx = rgb.max(2); mn = rgb.min(2)
    grey = (np.abs(rgb - 128).sum(2) < 60) & ((mx - mn) < 18)
    bright = rgb.sum(2) > 330
    not_blue = ~((rgb[:, :, 2] > rgb[:, :, 0] + 18) & (rgb[:, :, 2] > rgb[:, :, 1] + 18))
    reddish = (rgb[:, :, 0] > rgb[:, :, 2] + 6) & (rgb[:, :, 0] > rgb[:, :, 1] - 30)
    not_extreme = (mx - mn) < 150
    skin_src = valid_mask & (~protected_mask) & (~grey) & bright & not_blue & reddish & not_extreme
    if skin_src.sum() < 80:
        skin_src = valid_mask & (~grey) & bright & not_blue & reddish
    if skin_src.sum() < 20:
        return np.array([198.0, 150.0, 130.0], np.float32)
    return np.median(rgb[skin_src], axis=0)


def _complete_unmapped_atlas(atlas_img, valid_mask, observed_mask, protected_mask):
    """Turn partial observed UV into a renderable atlas without changing true coverage.

    Small holes get nearby texture by UV distance, preserving clothing/skin edges where possible.
    Large unknown body regions get a calm subject skin color rather than neutral grey.
    """
    stats = {"near_filled_texels": 0, "skin_filled_texels": 0}
    out = atlas_img.copy()
    rgb = out.astype(np.float32)
    mx = rgb.max(2); mn = rgb.min(2)
    grey = (np.abs(rgb - 128).sum(2) < 60) & ((mx - mn) < 18)
    unknown = valid_mask & grey
    known = valid_mask & (~grey)
    if not unknown.any() or known.sum() < 10:
        return out, stats

    try:
        from scipy.ndimage import distance_transform_edt
        dist, inds = distance_transform_edt(~known, return_indices=True)
        near = unknown & (dist <= 10.0)
        if near.any():
            out[near] = out[inds[0][near], inds[1][near]]
            stats["near_filled_texels"] = int(near.sum())
            unknown = unknown & (~near)
    except Exception:
        pass

    if unknown.any():
        skin = _skin_fill_color(out, valid_mask, protected_mask)
        out[unknown] = np.clip(skin, 0, 255).astype(np.uint8)
        stats["skin_filled_texels"] = int(unknown.sum())
    return out, stats


_FAN = {}
# dlib-68 landmark k (k in 17..67) corresponds to SMPL-X face JOINT index 76 + (k - 17).
# (verified: dlib30 nose-tip->89 nose4; dlib36 R-eye-outer->95; dlib48 R-mouth->107.)
DLIB_INNER = list(range(17, 68))                       # brows+nose+eyes+mouth (skip jaw contour)
SX_FACE_JOINT = [76 + (k - 17) for k in DLIB_INNER]    # matching SMPL-X joints (51 pts)


def _fan_landmarks(img_rgb):
    """FAN (face-alignment, torch, no GL) -> (68,2) dlib-ordered pixel landmarks, or None."""
    import face_alignment
    if "fa" not in _FAN:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _FAN["fa"] = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                                  device=dev, flip_input=False)
    preds = _FAN["fa"].get_landmarks(np.ascontiguousarray(img_rgb))
    if not preds:
        return None
    return np.asarray(preds[0], np.float32)            # largest/first face (68,2)


def face_align_warp(img, j2d):
    """Warp `img` (thin-plate-spline) so its REAL face landmarks (FAN, dlib-68) land on the
    mesh's projected SMPL-X face landmarks (j2d[76:127]), so photo eyes/nose/mouth map onto the
    mesh eyes/nose/mouth. Returns warped image or None if detection/fit fails."""
    import cv2
    from scipy.interpolate import RBFInterpolator
    try:
        lm = _fan_landmarks(img)
        if lm is None or lm.shape[0] < 68 or j2d.ndim != 2 or j2d.shape[0] < 127:
            print(f"  [face_align: skip lm={None if lm is None else lm.shape} j2d={j2d.shape}]")
            return None
        src = lm[DLIB_INNER]                            # photo features (51,2)
        dst = j2d[SX_FACE_JOINT]                        # mesh face landmarks (51,2)
        ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
        if ok.sum() < 20:
            print(f"  [face_align: skip ok={ok.sum()}]")
            return None
        # inverse map: for output(=mesh) coords, where to read in the photo. f(dst_i)=src_i.
        f = RBFInterpolator(dst[ok], src[ok], kernel="thin_plate_spline", smoothing=1.0)
        H, W = img.shape[:2]
        # only remap a padded box around the face (avoid wild TPS extrapolation elsewhere)
        x0, y0 = dst[ok].min(0); x1, y1 = dst[ok].max(0)
        pad = 0.6 * max(x1 - x0, y1 - y0)
        bx0, by0 = max(int(x0 - pad), 0), max(int(y0 - pad), 0)
        bx1, by1 = min(int(x1 + pad), W), min(int(y1 + pad), H)
        ys, xs = np.mgrid[by0:by1, bx0:bx1]
        grid = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        mapped = f(grid).astype(np.float32)
        mapx = mapped[:, 0].reshape(ys.shape); mapy = mapped[:, 1].reshape(ys.shape)
        warped = img.copy()
        warped[by0:by1, bx0:bx1] = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT)
        return warped
    except Exception as e:
        print("  [face_align_warp failed -> raw face]", repr(e)[:90])
        return None


_ALB = {}


def sota_albedo(img, mask):
    """SOTA single-image delighting via intrinsic decomposition (compphoto/Intrinsic, Careaga &
    Aksoy SIGGRAPH'23/'24 ordinal-shading + colorful-diffuse). Returns the diffuse ALBEDO
    (illumination removed by a learned net, not a heuristic), brightness-matched to the source
    skin so tone stays realistic. Falls back to retinex `delight` if the package is unavailable."""
    try:
        from intrinsic.pipeline import load_models, run_pipeline
        if "m" not in _ALB:
            _ALB["m"] = load_models("v2")
        res = run_pipeline(_ALB["m"], img.astype(np.float32) / 255.0)
        alb = np.clip(np.asarray(res["hr_alb"]), 0, 1)
        if alb.ndim == 2:
            alb = np.repeat(alb[:, :, None], 3, 2)
        if alb.shape[:2] != img.shape[:2]:                # pipeline may rescale -> restore size
            import cv2
            alb = cv2.resize(alb, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        m = np.asarray(mask, bool)
        if m.sum() > 80:                                  # match albedo brightness to source skin
            src_l = (img.astype(np.float32) / 255.0)[m].mean()
            alb_l = alb[m].mean() + 1e-6
            alb = np.clip(alb * (src_l / alb_l), 0, 1)
        return (alb * 255).astype(np.uint8)
    except Exception as e:
        print("  [sota_albedo unavailable -> retinex]", repr(e)[:80])
        return delight(img, mask)


def delight(img, mask):
    """Per-view lighting correction so baked texels are ALBEDO, not shaded photo pixels:
      (1) gray-world white balance over skin -> removes the scene color cast (warm room light),
          equalizing skin colour ACROSS views (kills seam hue jumps);
      (2) homomorphic luminance flattening -> divide out the low-frequency illumination
          (directional shadow/highlight) while KEEPING high-frequency real texture detail.
    Standard retinex/gray-world; no hardcoded outputs. Returns a delit copy (skin region)."""
    import cv2
    from scipy.ndimage import gaussian_filter
    m = np.asarray(mask, bool)
    if m.sum() < 80:
        return img
    f = img.astype(np.float32) / 255.0
    skin = f[m]
    mean_rgb = skin.mean(0) + 1e-6
    wb = np.clip(mean_rgb.mean() / mean_rgb, 0.75, 1.35)     # gray-world WB
    f = f * wb[None, None, :]
    lum = f.mean(2)
    sigma = max(img.shape[:2]) / 11.0
    illum = gaussian_filter(lum, sigma=sigma)                # low-freq illumination field
    tgt = float(np.clip(lum[m].mean(), 0.35, 0.70))          # preserve overall brightness
    ratio = np.clip(tgt / (illum + 1e-3), 0.6, 1.7)          # flatten shading, bounded
    f = f * ratio[:, :, None]
    out = img.copy()
    out[m] = np.clip(f[m] * 255.0, 0, 255).astype(img.dtype)
    return out


def skin_mask(img):
    """Skin-pixel mask (YCrCb + HSV ranges, tone-robust). For SKIN-ONLY fusion: bake/fit on
    body skin, treat clothing as non-body (unobserved) -> honest body appearance + shape."""
    import cv2
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1]; cb = ycrcb[:, :, 2]
    m1 = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 130)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m2 = ((h <= 25) | (h >= 170)) & (s >= 20) & (s <= 180) & (v >= 50)
    skin = (m1 | m2)
    from scipy.ndimage import binary_opening
    return binary_opening(skin, iterations=1)


def _yolo_person_box(img):
    if "yolo" not in _SEG:
        from ultralytics import YOLO
        _SEG["yolo"] = YOLO("yolov8x-seg.pt")
    r = _SEG["yolo"](img[:, :, ::-1], verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    cls = r.boxes.cls.cpu().numpy().astype(int)
    xyxy = r.boxes.xyxy.cpu().numpy()
    best = None; ba = 0
    for i in range(len(xyxy)):
        if cls[i] != 0:
            continue
        a = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        if a > ba:
            ba = a; best = xyxy[i]
    return best


def _sam2_predictor():
    if "sam" not in _SEG:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt = os.path.expanduser("~/LHM/pretrained_models/sam2/sam2.1_hiera_large.pt")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device=dev)
        _SEG["sam"] = SAM2ImagePredictor(model)
    return _SEG["sam"]


def person_mask(img):
    """Precise person matte via SAM2 prompted by the YOLO person box. Background-agnostic
    (works on complex backgrounds — no chroma cheat). Light erode to avoid edge bleed."""
    import cv2
    from scipy.ndimage import binary_fill_holes
    H, W = img.shape[:2]
    box = _yolo_person_box(img)
    if box is None:
        return np.ones((H, W), bool)
    pred = _sam2_predictor()
    pred.set_image(np.ascontiguousarray(img))
    masks, scores, _ = pred.predict(box=np.asarray(box)[None, :], multimask_output=False)
    m = np.asarray(masks).reshape(-1, *np.asarray(masks).shape[-2:])[0] > 0.5
    m = binary_fill_holes(m)
    # MULTI-CUE: fuse with depth foreground — the person is NEARER than the background, so
    # drop SAM2 pixels whose depth is far behind the person median (removes water/scene leaks
    # on complex backgrounds; generalizes, no chroma cheat).
    try:
        from pipeline.backends import real
        z = real._Models.depth_map(img)             # smaller = closer
        if m.sum() > 50:
            med = float(np.median(z[m]))
            spread = float(np.std(z[m])) + 1e-3
            m = m & (z <= med + 2.0 * spread)       # keep near-depth (foreground) only
    except Exception:
        pass
    m = binary_fill_holes(m)
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return m


_CSMODEL = {}


def _forward_consistent(betas, go, bp):
    """Forward SMPL-X(consistent betas, this view's global_orient+body_pose) -> verts."""
    import torch, smplx
    nb = len(betas)
    if nb not in _CSMODEL:
        _CSMODEL[nb] = smplx.create(A.HUMAN_MODELS, model_type="smplx", gender="neutral",
                                    num_betas=nb, use_pca=False, flat_hand_mean=True)
    with torch.no_grad():
        out = _CSMODEL[nb](betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                           global_orient=torch.tensor(go, dtype=torch.float32).unsqueeze(0),
                           body_pose=torch.tensor(bp, dtype=torch.float32).unsqueeze(0))
    return out.vertices[0].detach().cpu().numpy()


def _load_flip(nV):
    """SMPL-X L-R mirror correspondence as (closest_faces (nV,3) int, bc (nV,3) float): the
    mirror of vertex i is the barycentric point bc[i] on face closest_faces[i]. None if absent."""
    import os
    p = os.path.join(A.HUMAN_MODELS, "smplx/smplx_flip_correspondences.npz")
    if not os.path.exists(p):
        return None
    try:
        d = np.load(p)
        if "closest_faces" in d.files and "bc" in d.files:
            cf = d["closest_faces"].astype(np.int64); bc = d["bc"].astype(np.float32)
            if cf.shape == (nV, 3):
                return cf, bc
    except Exception:
        return None
    return None


def _detail_normal_map(atlas_rgb, strength=3.0):
    """Tangent-space normal map from albedo high-frequency luminance (Sobel) so fine
    appearance detail (veins, blemishes, tattoo edges) reads as relief under lighting."""
    g = atlas_rgb.astype(np.float32).mean(2) / 255.0
    try:
        import cv2
        g = cv2.GaussianBlur(g, (0, 0), 0.8)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    except Exception:
        gy, gx = np.gradient(g)
    nx = -gx * strength; ny = -gy * strength; nz = np.ones_like(g)
    n = np.stack([nx, ny, nz], -1)
    n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def _per_vertex_uv(vt, fv, fvt, nV):
    uv = np.zeros((nV, 2))
    for f in range(fv.shape[0]):
        for k in range(3):
            uv[fv[f, k]] = vt[fvt[f, k]]
    return uv


def build_uv_mesh(av, fv, fvt, vt):
    """Build a seam-split mesh on the vt topology: each UV vertex gets its own 3D position
    (duplicating verts at seams) -> clean texture seams, one UV per vertex."""
    nVT = vt.shape[0]
    vt_to_v = np.zeros(nVT, np.int64)
    for f in range(fv.shape[0]):
        for k in range(3):
            vt_to_v[fvt[f, k]] = fv[f, k]
    return av[vt_to_v], fvt, vt.copy()


if __name__ == "__main__":
    main()
