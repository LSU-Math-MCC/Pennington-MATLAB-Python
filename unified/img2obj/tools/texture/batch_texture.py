"""Texture-bake subjects in-process (models load once). No shell vars."""

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
import os

REPO = _repo
sys.path.insert(0, REPO + "/tools")
import texture_uv_bake  # noqa: E402

subjects = sys.argv[1:] or ["bodybuilding_vid_009_clip_000_person_001_frame_*.png"]
SSP_IMAGES = f"{REPO}/datasets/SSP-3D/ssp_3d/images"
atlas = "2048"
for s in subjects:
    sub = s if any(ch in s for ch in "*?[]/\\") else f"{SSP_IMAGES}/{s}"
    if not texture_uv_bake.image_paths(sub):
        print(f"[{s}] missing"); continue
    print(f"=== texture {s} ===", flush=True)
    name = os.path.splitext(os.path.basename(s.replace("*", "all")))[0]
    sys.argv = ["texture_uv_bake.py", "--subject", sub, "--out", f"{REPO}/runs/uv_{name}", "--atlas", atlas]
    try:
        texture_uv_bake.main()
    except SystemExit:
        pass
    except Exception:
        import traceback
        print(f"[{s}] ERR", traceback.format_exc()[-400:], flush=True)
print("BATCH_TEX_DONE", flush=True)
