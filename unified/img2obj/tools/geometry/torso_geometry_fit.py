"""Make the TORSO/abdomen geometry NON-GENERIC: deform the SMPL-X A-pose torso so its frontal
width profile exactly matches the person's silhouette (SAM2 mask, arms excluded), scale-
invariantly per normalized height. The parametric betas-fit barely moves on a single frontal
image; this directly conforms the abdomen coronal contour to the photo.

Writes <out_dir>/apose_torso_verts.npy (deformed A-pose vertices, same SMPL-X topology).
Run (WSL lhm): python tools/geometry/torso_geometry_fit.py <image> <betas.npy> <out_dir>
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
import numpy as np

REPO = _repo
sys.path.insert(0, REPO + "/tools")


def torso_halfwidths(mask_row, cx):
    """Left & right half-widths of the central body run about column cx. Occlusion (hand-on-hip)
    fattens ONE side -> we take the CLEAN (smaller) side and use L-R symmetry, so a locally
    obstructed side never blocks fitting the visible side. Returns (left_hw, right_hw)."""
    xs = np.where(mask_row)[0]
    if len(xs) < 3:
        return 0.0, 0.0
    splits = np.where(np.diff(xs) > 3)[0]
    runs = np.split(xs, splits + 1)
    best = min(runs, key=lambda r: abs((r[0] + r[-1]) / 2 - cx))
    return max(cx - best[0], 0.0), max(best[-1] - cx, 0.0)


def main():
    import texture_uv_bake as TB, lhm_anthropometry as A
    img_path, betas_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    betas = np.load(betas_path)[:10]
    img, h, j2d = TB.posed_view(TB.A._estimator() if hasattr(TB, "A") else A._estimator(), img_path)
    if h is None:
        print("no detection"); return
    j2d = np.asarray(j2d)
    mask = TB.person_mask(img)

    # observed TORSO half-width profile (central run) over the torso image-rows [shoulder..hip]
    sh_y = float(min(j2d[16, 1], j2d[17, 1]))            # shoulders
    hip_y = float(max(j2d[1, 1], j2d[2, 1]))             # hips
    cx = float(j2d[0, 0])                                 # pelvis x = body centre
    y0, y1 = int(min(sh_y, hip_y)), int(max(sh_y, hip_y))
    NB = 24
    obs = np.zeros(NB)
    for k in range(NB):
        yy = int(y0 + (y1 - y0) * (k + 0.5) / NB)
        if 0 <= yy < mask.shape[0]:
            lhw, rhw = torso_halfwidths(mask[yy], cx)
            # CLEAN SIDE via L-R symmetry: the occluded side is fatter -> take the smaller side.
            obs[k] = 2.0 * min(lhw, rhw) if min(lhw, rhw) > 1 else 2.0 * max(lhw, rhw)
    img_torso_h = (y1 - y0)
    obs_ratio = obs / (img_torso_h + 1e-6)               # scale-invariant width/torso-height

    # A-pose mesh + its torso width profile (central verts only; exclude arms by |x|)
    av, aj, faces, named = A.smplx_apose(betas, gender="female")
    sx_sh = 0.5 * (named["left_shoulder"][1] + named["right_shoulder"][1])
    sx_hip = 0.5 * (named["left_hip"][1] + named["right_hip"][1])
    ty0, ty1 = min(sx_sh, sx_hip), max(sx_sh, sx_hip)
    mesh_torso_h = ty1 - ty0
    # central torso verts: within torso y-band and |x| < 60% of the per-band max (drop arms)
    in_band = (av[:, 1] >= ty0) & (av[:, 1] <= ty1)
    t_mesh = (av[:, 1] - ty0) / (ty1 - ty0 + 1e-6)
    mesh_ratio = np.zeros(NB); halfw_band = np.zeros(NB)
    for k in range(NB):
        sel = in_band & (np.abs(t_mesh - (k + 0.5) / NB) < (0.5 / NB))
        if sel.sum() > 4:
            xc = av[sel, 0]
            torso = xc[np.abs(xc - np.median(xc)) < 0.6 * (np.abs(xc - np.median(xc)).max() + 1e-6)]
            halfw_band[k] = (torso.max() - torso.min()) / 2.0
            mesh_ratio[k] = halfw_band[k] / mesh_torso_h

    # OCCLUSION-ROBUST per-height scale. An obstructed row (arm/hand merged with the torso ->
    # NO gap, anomalously wide run) must NOT corrupt the fit. We fit VISIBLE heights exactly and
    # INTERPOLATE the occluded ones from clean neighbours -> local progress on visible regions,
    # occluded regions never block or distort them.
    from scipy.ndimage import gaussian_filter1d
    raw = np.where((mesh_ratio > 1e-4) & (obs_ratio > 1e-4), obs_ratio / np.maximum(mesh_ratio, 1e-4), np.nan)
    valid = np.isfinite(raw)
    # light outlier reject (a height where BOTH sides are occluded -> spike); interpolate those
    med = np.nanmedian(raw); mad = np.nanmedian(np.abs(raw - med)) + 1e-6
    clean = valid & (np.abs(raw - med) < 4.0 * mad)
    idx = np.arange(NB)
    scale_k = np.interp(idx, idx[clean], raw[clean]) if clean.sum() >= 2 else np.where(valid, raw, 1.0)
    scale_k = gaussian_filter1d(scale_k, 0.8)
    # NARROW-ONLY: the frontal silhouette includes arms-at-sides (the A-pose mesh torso excludes
    # them), so any "wider than mesh" is arm contamination -> cap widening. We only trust the
    # silhouette to NARROW the torso (a genuinely lean subject), never to balloon it.
    scale_k = np.clip(scale_k, 0.55, 1.03)
    print(f"  torso heights: {int(clean.sum())}/{NB} fit from clean side, {NB-int(clean.sum())} interpolated; "
          f"scale range {scale_k.min():.2f}-{scale_k.max():.2f}")
    avd = av.copy()
    cxm = np.median(av[in_band, 0])
    for k in range(NB):
        sel = in_band & (np.abs(t_mesh - (k + 0.5) / NB) < (0.5 / NB))
        # weight: 1 for central torso verts, 0 for arm verts (far in x)
        if sel.sum() == 0:
            continue
        dx = np.abs(av[:, 0] - cxm)
        w = np.clip(1.0 - dx / (1.2 * halfw_band[k] + 1e-6), 0, 1)
        m = sel
        avd[m, 0] = cxm + (av[m, 0] - cxm) * (1 + (scale_k[k] - 1) * w[m])
    np.save(os.path.join(out_dir, "apose_torso_verts.npy"), avd)
    print(f"TORSO_FIT mean coronal scale {scale_k.mean():.3f} (obs/mesh width ratio) -> apose_torso_verts.npy")


if __name__ == "__main__":
    main()
