"""Overlay the recovered body mesh on the original input images.

For each view we build a triangle mesh directly from that view's
``person_splats.npz``. The raw splats reproject *exactly* onto the subject (they
share the camera frame of ``selected_depth_samples.npz``, whose affine
world->pixel map we fit), so we triangulate them in image space (Delaunay) and
prune triangles that span gaps - long edges or depth jumps - giving a real
triangle mesh whose wireframe traces the body with no overhang. A Poisson shell
was tried first but ballooned off the body; triangulating the aligned points is
both tighter and faithfully aligned.

The splats carry a background wall/floor depth tail; depth_slab_mask keeps only
the dominant near depth slab so the mesh covers the person, not the background.
The wireframe output is cropped to the subject so the triangles are visible. The
JSON report records projection residuals so misalignment is visible.

Usage:
  python tools/render/overlay_final_mesh.py runs/subject_s1
  python tools/render/overlay_final_mesh.py runs/subject_s1 --points 3000 # denser mesh
  python tools/render/overlay_final_mesh.py runs/subject_s1 --mesh path.glb  # fixed mesh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.export.ply import load_splat_npz  # noqa: E402
from pipeline.geometry.mask_depth_select import load_samples  # noqa: E402
from pipeline.geometry.transforms import apply_T, make_T, procrustes  # noqa: E402


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def fit_affine_projection(points_world: np.ndarray, pixels: np.ndarray, max_points=12000):
    pts = np.asarray(points_world, float)
    pix = np.asarray(pixels, float)
    n = min(len(pts), len(pix))
    if n < 8:
        raise ValueError("need at least 8 samples for projection fit")
    pts, pix = pts[:n], pix[:n]
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        pts, pix = pts[idx], pix[idx]
    X = np.c_[pts, np.ones(len(pts))]
    coef, *_ = np.linalg.lstsq(X, pix, rcond=None)  # 4x2
    pred = X @ coef
    err = np.linalg.norm(pred - pix, axis=1)
    return coef, {"points": int(len(pts)), "mean_px": float(err.mean()),
                  "p95_px": float(np.percentile(err, 95))}


def project_affine(coef: np.ndarray, verts_world: np.ndarray):
    X = np.c_[np.asarray(verts_world, float), np.ones(len(verts_world))]
    return X @ coef


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def robust_inlier_mask(points: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Per-axis median/MAD rejection to drop stray splats before meshing."""
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return np.zeros(0, bool)
    med = np.median(pts, axis=0)
    mad = np.median(np.abs(pts - med), axis=0)
    scale = np.where(mad > 1e-9, mad * 1.4826, 1e-9)
    z = np.abs(pts - med) / scale
    return np.isfinite(pts).all(1) & (z <= k).all(1)


def depth_slab_mask(points: np.ndarray, axis: int = 2, bins: int = 60,
                    rel: float = 0.05) -> np.ndarray:
    """Keep only the dominant near depth slab (the person).

    The person splats carry a depth tail of background wall/floor (e.g. for
    subject_s1: a person mass at 2.6-3.2m and a separate wall cluster at
    ~3.8m). Reconstructing over both drags the mesh into the background. We
    find the histogram mode along the camera/depth axis and grow outward while
    bins stay above ``rel`` of the peak, cutting at the first deep valley."""
    pts = np.asarray(points, float)
    z = pts[:, axis]
    if len(z) < 2 or not np.isfinite(z).any():
        return np.ones(len(pts), bool)
    h, edges = np.histogram(z, bins=bins)
    mode = int(h.argmax())
    thr = h[mode] * rel
    lo = mode
    while lo > 0 and h[lo - 1] >= thr:
        lo -= 1
    hi = mode
    while hi < bins - 1 and h[hi + 1] >= thr:
        hi += 1
    return (z >= edges[lo]) & (z <= edges[hi + 1])


def triangulate_view(points: np.ndarray, proj: np.ndarray, max_points: int = 2200):
    """Build a triangle mesh that hugs the projected person points.

    The raw person splats reproject *exactly* onto the subject, so instead of a
    Poisson shell (which balloons off the body) we triangulate the points in
    image space (Delaunay) and prune triangles that span gaps - between the
    legs, arm and torso, or across a depth discontinuity. The result is a real
    triangle mesh whose wireframe traces the body with no overhang.

    Returns (verts3d, uv, faces)."""
    from scipy.spatial import Delaunay

    pts = np.asarray(points, float)
    pts = pts[depth_slab_mask(pts)]      # drop background wall/floor depth tail
    pts = pts[robust_inlier_mask(pts)]   # drop lateral stray splats
    if len(pts) < 16:
        raise ValueError(f"too few points to triangulate: {len(pts)}")
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]

    uv = project_affine(proj, pts)
    keep = np.isfinite(uv).all(1)
    pts, uv = pts[keep], uv[keep]

    tri = Delaunay(uv)
    faces = tri.simplices
    # 2D edge lengths per triangle
    e0 = np.linalg.norm(uv[faces[:, 0]] - uv[faces[:, 1]], axis=1)
    e1 = np.linalg.norm(uv[faces[:, 1]] - uv[faces[:, 2]], axis=1)
    e2 = np.linalg.norm(uv[faces[:, 2]] - uv[faces[:, 0]], axis=1)
    longest = np.maximum.reduce([e0, e1, e2])
    # depth span per triangle (camera-depth = world z in the person frame)
    z = pts[:, 2]
    zspan = np.maximum.reduce([
        np.abs(z[faces[:, 0]] - z[faces[:, 1]]),
        np.abs(z[faces[:, 1]] - z[faces[:, 2]]),
        np.abs(z[faces[:, 2]] - z[faces[:, 0]]),
    ])
    edge_thr = np.median(longest) * 2.5      # alpha-shape: kill long spanning tris
    depth_thr = np.median(zspan) * 4.0 + 1e-6
    good = (longest <= edge_thr) & (zspan <= depth_thr)
    faces = faces[good]
    if len(faces) == 0:
        raise ValueError("triangulation pruned to empty mesh")
    return pts, uv, faces


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def draw_wireframe(image_path: Path, uv: np.ndarray, faces: np.ndarray,
                   out_path: Path, color=(40, 235, 90), alpha=0.85):
    """Draw the triangle mesh as a wireframe over the photo (unique edges only)."""
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("overlay drawing requires opencv-python") from e

    img = np.asarray(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    overlay = img.copy()

    f = np.asarray(faces, int)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    e = np.unique(e, axis=0)

    uv = np.asarray(uv, float)
    pix = np.round(uv).astype(int)
    drawn = 0
    for a, b in e:
        pa, pb = pix[a], pix[b]
        cv2.line(overlay, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 color, 1, cv2.LINE_AA)
        drawn += 1

    out = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    # Crop to the subject (+margin) so the triangle mesh is actually visible
    # instead of a tiny blob in a big frame.
    finite = np.isfinite(uv).all(1)
    if finite.any():
        x0, y0 = pix[finite].min(0)
        x1, y1 = pix[finite].max(0)
        mx = int(0.18 * max(x1 - x0, 1))
        my = int(0.18 * max(y1 - y0, 1))
        x0, y0 = max(int(x0) - mx, 0), max(int(y0) - my, 0)
        x1, y1 = min(int(x1) + mx, W - 1), min(int(y1) + my, H - 1)
        if x1 > x0 and y1 > y0:
            out = out[y0:y1 + 1, x0:x1 + 1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)
    return {"edges": int(drawn), "faces": int(len(f))}


# --------------------------------------------------------------------------- #
# run discovery / IO
# --------------------------------------------------------------------------- #
def manifest_inputs(run_dir: Path):
    mf = run_dir / "manifest.json"
    if not mf.exists():
        return []
    data = json.loads(mf.read_text())
    return [Path(p) for p in data.get("inputs", [])]


def resolve_image_for_view(view_dir: Path, inputs: list[Path]):
    def fix(p: Path):
        return (REPO / p).resolve() if not p.is_absolute() else p
    if len(inputs) == 1:
        return fix(inputs[0])
    stem, parent = view_dir.name, view_dir.parent.name
    for p in inputs:
        if p.stem == stem and p.parent.name == parent:
            return fix(p)
    for p in inputs:
        if p.stem == stem:
            return fix(p)
    raise FileNotFoundError(f"could not match source image for {view_dir}")


def discover_views(run_dir: Path):
    views_root = run_dir / "views"
    if views_root.exists():
        return sorted(p for p in views_root.glob("*/*") if (p / "instances").exists())
    if (run_dir / "instances").exists():
        return [run_dir]
    return []


def first_instance_dir(view_dir: Path):
    insts = sorted(p for p in (view_dir / "instances").glob("*") if p.is_dir())
    if not insts:
        raise FileNotFoundError(f"no instance dirs under {view_dir}")
    return insts[0]


# --------------------------------------------------------------------------- #
# fixed-mesh mode (overlay a single external mesh through the view alignment)
# --------------------------------------------------------------------------- #
def load_mesh(path: Path):
    import trimesh
    mesh = trimesh.load(path, process=False, force="mesh")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"mesh has no vertices: {path}")
    return np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)


def fit_similarity(src: np.ndarray, dst: np.ndarray, max_points=8000):
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = min(len(src), len(dst))
    if n < 3:
        raise ValueError("need at least 3 paired points for similarity fit")
    src, dst = src[:n], dst[:n]
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        src, dst = src[idx], dst[idx]
    R, t, s = procrustes(src, dst, with_scale=True)
    T = make_T(R, t, scale=s)
    err = np.linalg.norm(apply_T(T, src) - dst, axis=1)
    return T, {"points": int(len(src)), "mean_m": float(err.mean()),
               "p95_m": float(np.percentile(err, 95))}


def subject_alignment_transform(run_dir: Path, view_index: int):
    report = run_dir / "debug" / "alignment_report.json"
    if not report.exists():
        return np.eye(4)
    data = json.loads(report.read_text())
    transforms = data.get("transforms", [])
    if view_index >= len(transforms):
        return np.eye(4)
    T = np.asarray(transforms[view_index], float)
    try:
        return np.linalg.inv(T)
    except np.linalg.LinAlgError:
        return np.eye(4)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def overlay_run(run_dir: Path, out_dir: Path | None, fixed_mesh: Path | None = None,
                target_points: int = 1500):
    run_dir = run_dir.resolve()
    out_dir = (out_dir or (run_dir / "overlays")).resolve()
    inputs = manifest_inputs(run_dir)
    views = discover_views(run_dir)
    if not views:
        raise SystemExit(f"OVERLAY_FAIL no view dirs found under {run_dir}")

    ext_verts = ext_faces = None
    if fixed_mesh is not None:
        ext_verts, ext_faces = load_mesh(fixed_mesh.resolve())

    report = {"run_dir": str(run_dir), "fixed_mesh": str(fixed_mesh) if fixed_mesh else None,
              "views": []}
    for i, view_dir in enumerate(views):
        try:
            inst = first_instance_dir(view_dir)
            image = resolve_image_for_view(view_dir, inputs)
            person = load_splat_npz(inst / "person_splats.npz")
            samples = load_samples(inst / "selected_depth_samples.npz")
            proj, proj_stats = fit_affine_projection(samples.points_world, samples.pixels)

            if ext_verts is not None:
                # Map the external (canonical) mesh into this view's posed frame.
                can = load_splat_npz(inst / "canonical_splats.npz")
                ref_to_view = subject_alignment_transform(run_dir, i)
                T_can_to_world, _ = fit_similarity(can.centers, person.centers)
                verts_world = apply_T(T_can_to_world, apply_T(ref_to_view, ext_verts))
                uv = project_affine(proj, verts_world)
                faces = ext_faces
                recon_stats = {"source": "fixed", "faces": int(len(faces))}
            else:
                _, uv, faces = triangulate_view(person.centers, proj, target_points)
                recon_stats = {"source": "person_splats_delaunay", "faces": int(len(faces))}

            out_path = out_dir / f"{view_dir.parent.name}_{view_dir.name}_mesh_overlay.png"
            draw_stats = draw_wireframe(image, uv, faces, out_path)
            rec = {"view": str(view_dir), "image": str(image), "out": str(out_path),
                   "projection": proj_stats, "reconstruction": recon_stats, "draw": draw_stats}
            print(f"OVERLAY_OK {out_path} proj_mean={proj_stats['mean_px']:.1f}px "
                  f"faces={draw_stats['faces']} edges={draw_stats['edges']}")
        except Exception as e:
            rec = {"view": str(view_dir), "error": repr(e)}
            print(f"OVERLAY_FAIL {view_dir}: {repr(e)}")
        report["views"].append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlay_report.json").write_text(json.dumps(report, indent=2))
    print(f"OVERLAY_REPORT {out_dir / 'overlay_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="pipeline run dir, e.g. runs/subject_s1")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mesh", default=None,
                    help="overlay this fixed mesh (canonical frame) instead of "
                         "reconstructing per view from person_splats")
    ap.add_argument("--points", type=int, default=1500,
                    help="max points fed to the per-view triangulation (mesh density)")
    args = ap.parse_args()
    overlay_run(Path(args.run_dir),
                Path(args.out_dir) if args.out_dir else None,
                Path(args.mesh) if args.mesh else None,
                target_points=args.points)


if __name__ == "__main__":
    main()
