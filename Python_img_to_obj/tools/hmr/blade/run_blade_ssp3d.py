"""Collect BLADE's predicted SMPL-X betas on SSP-3D (for the shape teaser).

LOADS BLADE ONCE and runs all representative frames in a single process (one BLADE_API,
samples_per_gpu=1 so no OOM; cur_id == image stem so each `<stem>.pth` maps straight back). BLADE
saves SMPL-X opti betas in that .pth. Subjects are the diverse FPS set (ssp3d_subjects), so the
teaser isn't all one body type. Output: runs/BLADE_ssp3d.npz aligned to SSP-3D's 311 rows.

Run (blade_env):  cd ~/blade && python <this> [--n-subjects 8]
Run every unique SSP-3D subject once: python <this> --all-subjects --crop
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

import os, sys, glob, argparse, shutil
import numpy as np

REPO = _repo
SSP = os.path.expanduser("~/SSP-3D/data_ext/ssp_3d")
BLADE = os.path.expanduser("~/blade")
sys.path.insert(0, BLADE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssp3d_subjects import select_rep_frames, subject_groups
TMP = os.path.join(REPO, "runs", "blade_ssp3d_tmp")


SMPL_DIR = os.path.expanduser("~/shapy/data/body_models/smpl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=8)
    ap.add_argument("--all-subjects", action="store_true",
                    help="run one representative frame for every unique SSP-3D subject")
    ap.add_argument("--skinny", action="store_true", help="run the skinniest subjects instead of diverse")
    ap.add_argument("--crop", action="store_true", help="use close-range crops for BLADE")
    ap.add_argument("--crop-mode", choices=("target", "gt-pad"), default="target",
                    help="target=precomputed subject-only crop; gt-pad=old padded GT square")
    ap.add_argument("--batch", type=int, default=1, help="GPU batch size (amortizes backbones + camera solve)")
    ap.add_argument("--iters", type=int, default=0, help="override camera-solve iterations (0 = default 100)")
    args = ap.parse_args()
    os.environ.setdefault("MINI_BATCHSIZE", "1")
    os.chdir(BLADE)
    import torch
    from api.BLADE_API import BLADE_API

    lab = np.load(os.path.join(SSP, "labels.npz"))
    fn, gt = lab["fnames"], lab["shapes"]
    ctr, whs = lab["bbox_centres"], lab["bbox_whs"]
    if args.all_subjects:
        reps = [m[0] for m in subject_groups(gt)]
    else:
        reps = select_rep_frames(gt, args.n_subjects, skinny=args.skinny, smpl_dir=SMPL_DIR)
    stem2row = {os.path.splitext(str(fn[i]))[0]: i for i in reps}
    tag = ("all-subjects" if args.all_subjects else ("skinny" if args.skinny else "diverse"))
    tag += "+crop" if args.crop else ""
    print(f"BLADE betas on {len(reps)} {tag} SSP-3D frames (one model load)")

    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)

    def rgb_for(i):
        src = os.path.join(SSP, "images", str(fn[i]))
        if not args.crop:
            return src
        pre = os.path.join(REPO, "runs", "ssp3d_target_crops", os.path.splitext(str(fn[i]))[0] + ".png")
        if args.crop_mode == "target" and os.path.exists(pre):
            return pre
        import cv2
        im = cv2.imread(src); H, W = im.shape[:2]; cx, cy = ctr[i]; wh = float(whs[i])
        if args.crop_mode == "gt-pad":
            wh *= 1.25
        x1, y1 = int(max(0, cx - wh / 2)), int(max(0, cy - wh / 2))
        x2, y2 = int(min(W, cx + wh / 2)), int(min(H, cy + wh / 2))
        cp = os.path.join(TMP, "crop_" + os.path.splitext(str(fn[i]))[0] + ".png")
        cv2.imwrite(cp, im[y1:y2, x1:x2]); return cp
    batch_list = {os.path.splitext(str(fn[i]))[0]: {"rgb_file": rgb_for(i)} for i in reps}
    import time as _t; t0 = _t.time()
    api = BLADE_API(batch_list=batch_list, render_and_save_imgs=False, temp_output_folder=TMP,
                    device="cuda:0", samples_per_gpu=args.batch, workers_per_gpu=0)   # ONE model load
    if args.iters > 0:
        try:
            api.model.module.n_optimization_iterations = args.iters
            print(f"camera-solve iters -> {args.iters}")
        except Exception as e:
            print(f"could not set iters: {e}")
    api.process()
    print(f"BLADE inference: {(_t.time()-t0):.0f}s for {len(reps)} frames "
          f"(batch={args.batch}, iters={args.iters or 100})")

    # MERGE with any existing betas (so a skinny run keeps the diverse ones, and vice-versa).
    # --crop writes a SEPARATE file so cropped vs uncropped can be scored side by side.
    sfx = "_crop" if args.crop else ""
    out_npz = os.path.expanduser(f"~/BLADE_ssp3d{sfx}.npz")
    betas = np.full((len(fn), 10), np.nan, np.float32)
    if os.path.exists(out_npz):
        prev = np.load(out_npz)["betas"]
        if prev.shape == betas.shape:
            betas = prev.copy()
    for stem, row in stem2row.items():
        pth = os.path.join(TMP, stem + ".pth")
        if not os.path.exists(pth):
            print(f"  !! no .pth for {stem}"); continue
        b = torch.load(pth, map_location="cpu").get("smplx_betas")
        if b is not None:
            betas[row] = np.asarray(b).reshape(-1)[:10]
    np.savez(os.path.join(REPO, "runs", f"BLADE_ssp3d{sfx}.npz"), betas=betas, fnames=fn)
    np.savez(out_npz, betas=betas, fnames=fn)   # durable
    print(f"saved runs/BLADE_ssp3d{sfx}.npz ({int(np.sum(~np.isnan(betas[:,0])))} total subjects covered)")


if __name__ == "__main__":
    main()
