"""Run abdomen_terrain over subjects, auto-selecting the most FRONTAL full-body view
(terrain shading-integration is reliable only on frontal skin). In-process, no shell vars."""

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

import sys
import glob
import os

import numpy as np

REPO = _repo
sys.path.insert(0, REPO + "/tools")
import abdomen_terrain  # noqa: E402
import texture_uv_bake as TB  # noqa: E402
import lhm_anthropometry as A  # noqa: E402

# SMPL-X joint indices in Multi-HMR j2d
L_SH, R_SH, L_HIP, R_HIP = 16, 17, 1, 2


def frontality(img_path, est):
    """Higher = more frontal full-body: biacromial width / torso height in 2D.
    Guards against degenerate/cropped views (tiny torso height -> bogus huge ratio)."""
    try:
        img, h, j2d = TB.posed_view(est, img_path)
        if h is None:
            return -1
        H = img.shape[0]
        sw = np.linalg.norm(j2d[L_SH] - j2d[R_SH])
        th = abs(0.5 * (j2d[L_SH, 1] + j2d[R_SH, 1]) - 0.5 * (j2d[L_HIP, 1] + j2d[R_HIP, 1]))
        if th < 0.10 * H or sw < 0.05 * img.shape[1]:     # cropped/zoomed -> reject
            return -1
        return float(min(sw / (th + 1e-6), 1.2))          # cap (frontal full-body ~0.4-0.7)
    except Exception:
        return -1


def main():
    subs = sys.argv[1:] or ["ssp3d_bodybuilder"]
    ssp_glob = f"{REPO}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png"
    est = A._estimator()
    for s in subs:
        subject_glob = s if any(ch in s for ch in "*?[]/\\") else ssp_glob
        imgs = sorted(glob.glob(subject_glob))
        if not imgs:
            print(f"[{s}] no images"); continue
        scored = [(frontality(im, est), im) for im in imgs]
        scored.sort(reverse=True)
        best = scored[0][1]
        print(f"=== terrain {s} : {os.path.basename(best)} (frontality {scored[0][0]:.2f}) ===",
              flush=True)
        sys.argv = ["abdomen_terrain.py", "--image", best, "--out", f"{REPO}/runs/terrain_{s}"]
        try:
            abdomen_terrain.main()
        except SystemExit:
            pass
        except Exception:
            import traceback
            print(f"[{s}] ERR", traceback.format_exc()[-400:], flush=True)
    print("BATCH_TERRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
