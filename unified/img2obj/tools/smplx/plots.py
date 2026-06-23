"""Cross-backend SMPL-X comparison plots (subcommands).

Subcommands:
  comparison   qualitative per-backend overlay grid (row per HMR backend, col per image)
  matrix       pairwise inter-method shape-disagreement matrix for one subject

Both drive the polymorphic backend registry in this package (backends.py / schema.py /
render.py). To add a plot: write a function and a thin ``_<name>_cli`` wrapper, then add
it to ``COMMANDS`` at the bottom. Heavy deps (pyrender/torch/smplx/matplotlib) are
imported lazily inside the subcommands.

Usage: python tools/smplx/plots.py <subcommand> [args...]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends  # noqa: E402
backends.reexec_in_wsl()  # on Windows: run this whole script in the WSL camerahmr env


# -------------------------------------------------------------------------- comparison
ROW_H = 320            # px per row (cells scaled to this height)
GUTTER = 120           # left label gutter px


def _cell_for(work, name, img_path):
    """A backend cell is EITHER a pre-rendered overlay PNG (e.g. BLADE's own render) OR a schema
    npz of geometry we render with the shared renderer (e.g. CameraHMR). Prefer the overlay PNG."""
    import schema
    import render

    stem = os.path.splitext(os.path.basename(img_path))[0]
    png = os.path.join(work, name, stem + ".overlay.png")
    npz = os.path.join(work, name, stem + ".npz")
    if os.path.exists(png):
        return np.array(Image.open(png).convert("RGB"))
    if os.path.exists(npz):
        r = schema.load(npz)
        bg = np.array(Image.open(r["image_path"]).convert("RGB"))
        if len(r["people"]) == 0:
            return bg
        return render.render_overlay(r["people"], r["faces"], r["focal"], r["img_w"], r["img_h"], bg_rgb=bg)
    return np.array(Image.open(img_path).convert("RGB"))


def _scale(arr, h):
    im = Image.fromarray(arr)
    w = max(1, int(im.width * h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def _comparison_cli():
    from backends import REGISTRY

    ap = argparse.ArgumentParser(prog="plots.py comparison")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--backends", nargs="+", default=["camerahmr", "blade"])
    ap.add_argument("--out", default="runs/HMR_COMPARE.png")
    ap.add_argument("--work", default="runs/hmr_compare")
    ap.add_argument("--skip-run", action="store_true", help="reuse existing npz, don't re-run models")
    args = ap.parse_args()

    images = [os.path.expanduser(p) for p in args.images]
    backends = [REGISTRY[b] for b in args.backends]
    work = os.path.expanduser(args.work)

    # 1) run each backend (in its own env) -> normalized npz per image
    if not args.skip_run:
        for b in backends:
            b.run(images, work)

    # 2) build the grid: rows = [Image] + backends ; cols = images
    rows = [("Image", [_scale(np.array(Image.open(p).convert("RGB")), ROW_H) for p in images])]
    for b in backends:
        cells = [_scale(_cell_for(work, b.name, p), ROW_H) for p in images]
        rows.append((b.label, cells))

    col_w = [max(rows[r][1][c].width for r in range(len(rows))) for c in range(len(images))]
    W = GUTTER + sum(col_w)
    H = ROW_H * len(rows)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for ri, (label, cells) in enumerate(rows):
        y = ri * ROW_H
        draw.text((10, y + ROW_H // 2 - 10), label, fill=(0, 0, 0), font=font)
        x = GUTTER
        for ci, cell in enumerate(cells):
            canvas.paste(cell, (x + (col_w[ci] - cell.width) // 2, y))
            x += col_w[ci]
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    canvas.save(os.path.expanduser(args.out))
    print(f"saved {args.out}  ({len(rows)} rows x {len(images)} cols)")


# ------------------------------------------------------------------------------- matrix
ERR_MAX = 50.0
SPEC = [("CameraHMR", "camerahmr", "smpl"), ("SHAPY", "shapy", "smplx"), ("BLADE", "blade", "smplx")]
SMPL_DIR = os.path.expanduser("~/shapy/data/body_models/smpl")
SMPLX_DIR = os.path.expanduser("~/shapy/data/body_models/smplx")


def smpl_apose():
    bp = np.zeros((23, 3), np.float32); bp[15, 2] = -np.deg2rad(55); bp[16, 2] = np.deg2rad(55); return bp.reshape(-1)


def smplx_apose():
    bp = np.zeros((21, 3), np.float32); bp[15, 2] = -np.deg2rad(55); bp[16, 2] = np.deg2rad(55); return bp.reshape(-1)


def render_mesh(verts, faces, vert_colors, w=280, h=500):
    import pyrender, trimesh
    m = trimesh.Trimesh(verts, faces, vertex_colors=vert_colors, process=False)
    sc = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[.45, .45, .45])
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=True))
    c = verts.mean(0); ext = np.ptp(verts, 0).max()
    sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.4),
           pose=np.array([[1, 0, 0, c[0]], [0, 1, 0, c[1]], [0, 0, 1, c[2] + ext * 1.5], [0, 0, 0, 1]], float))
    for dx, inten in [(0.6, 3.5), (-0.7, 2.0)]:
        lp = np.eye(4); lp[0, 3] = c[0] + dx * ext; lp[1, 3] = c[1] + 0.3 * ext; lp[2, 3] = c[2] + ext * 1.5
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=inten), pose=lp)
    r = pyrender.OffscreenRenderer(w, h); col, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
    return col


def _matrix_cli():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    ap = argparse.ArgumentParser(prog="plots.py matrix")
    ap.add_argument("--subject", default="s1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(_REPO, "runs", f"PAIRWISE_{args.subject}.png")

    import torch, smplx, trimesh, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import cm

    smpl = smplx.SMPL(SMPL_DIR, gender="neutral")
    smplx_m = smplx.SMPLX(SMPLX_DIR, gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=10)
    bps, bpx = torch.tensor(smpl_apose())[None], torch.tensor(smplx_apose())[None]

    def verts_of(betas, mtype):
        with torch.no_grad():
            if mtype == "smpl":
                o = smpl(betas=torch.tensor(betas[:10]).float()[None], body_pose=bps); f = smpl.faces
            else:
                o = smplx_m(betas=torch.tensor(betas[:10]).float()[None], body_pose=bpx); f = smplx_m.faces.astype(np.int64)
        return o.vertices[0].numpy() - o.joints[0, 0].numpy(), f

    methods = []
    for label, key, mtype in SPEC:
        p = os.path.join(_REPO, "runs", f"{key}_{args.subject}_betas.npy")
        if os.path.exists(p):
            v, f = verts_of(np.load(p), mtype)
            methods.append((label, v, f))
        else:
            print(f"  missing {p} (skipping {label})")
    if len(methods) < 2:
        sys.exit("need >=2 methods")
    N = len(methods)
    cmap = cm.get_cmap("viridis")

    fig, axes = plt.subplots(N, N, figsize=(2.5 * N + 1, 4.2 * N))
    axes = np.atleast_2d(axes)
    for i, (li, vi, fi) in enumerate(methods):
        for j, (lj, vj, fj) in enumerate(methods):
            ax = axes[i, j]; ax.axis("off")
            if i == j:
                ax.imshow(render_mesh(vi, fi, np.tile([180, 180, 185, 255], (vi.shape[0], 1)).astype(np.uint8)))
            else:
                sh = vj[:, 1].ptp() / (vi[:, 1].ptp() + 1e-9)           # height-scale align i -> j
                prox = trimesh.proximity.ProximityQuery(trimesh.Trimesh(vj, fj, process=False))
                err = np.abs(prox.signed_distance(vi * sh)) * 1000.0
                cc = (cmap(np.clip(err / ERR_MAX, 0, 1))[:, :4] * 255).astype(np.uint8)
                ax.imshow(render_mesh(vi, fi, cc))
                ax.text(0.5, 0.01, f"{err.mean():.1f} mm", transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=9, color=(0.1, 0.1, 0.1))
            if i == 0:
                ax.set_title(lj, fontsize=12, pad=6)
            if j == 0:
                ax.text(-0.08, 0.5, li, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=12)
    sm = cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, ERR_MAX))
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01).set_label("row mesh -> column surface (mm)")
    fig.suptitle(f"Pairwise inter-method shape disagreement — subject {args.subject} "
                 f"(canonical A-pose, no ground truth)", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}  ({N}x{N} methods={[m[0] for m in methods]})")


# ----------------------------------------------------------------------------- dispatch
COMMANDS = {"comparison": _comparison_cli, "matrix": _matrix_cli}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {os.path.basename(sys.argv[0])} <{'|'.join(COMMANDS)}> [args...]")
    _name = sys.argv[1]
    sys.argv = [f"{sys.argv[0]} {_name}", *sys.argv[2:]]
    COMMANDS[_name]()
