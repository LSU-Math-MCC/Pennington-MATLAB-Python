"""Run SHAPY's image->shape regressor on SSP-3D and save SMPL-X betas aligned to its 311 rows.

SHAPY's demo needs OpenPose BODY-25 keypoints; SSP-3D ships COCO-17 joints2D, so we remap them
and lay out the demo's expected {images/, openpose/} folders, then call SHAPY (shapy env) and
collect each image's predicted SMPL-X betas into runs/SHAPY_ssp3d.npz (used by make_shapy_teaser).

Run (shapy env):  cd ~/shapy/regressor && python <this> [--n-subjects 6]
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

import os, sys, json, glob, subprocess, argparse
import numpy as np

REPO = _repo
SSP = os.path.expanduser("~/SSP-3D/data_ext/ssp_3d")
SHAPY = os.path.expanduser("~/shapy/regressor")
# COCO-17 index -> OpenPose BODY-25 index
COCO2OP = {0: 0, 1: 16, 2: 15, 3: 18, 4: 17, 5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4,
           11: 12, 12: 9, 13: 13, 14: 10, 15: 14, 16: 11}


def coco17_to_body25(j):                       # j: (17,3) x,y,conf
    b = np.zeros((25, 3), np.float32)
    for c, o in COCO2OP.items():
        b[o] = j[c]
    b[1] = (j[5] + j[6]) / 2; b[1, 2] = min(j[5, 2], j[6, 2])     # neck = shoulder mid
    b[8] = (j[11] + j[12]) / 2; b[8, 2] = min(j[11, 2], j[12, 2]) # midhip = hip mid
    return b


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssp3d_subjects import select_diverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=6, help="0 = all 311 frames")
    ap.add_argument("--work", default=os.path.join(REPO, "runs", "shapy_ssp3d"))
    args = ap.parse_args()

    lab = np.load(os.path.join(SSP, "labels.npz"))
    fn, j2d, gt = lab["fnames"], lab["joints2D"], lab["shapes"]
    if args.n_subjects > 0:
        idx = sorted(i for mem in select_diverse(gt, args.n_subjects) for i in mem)
    else:
        idx = list(range(len(fn)))
    print(f"SHAPY on {len(idx)} SSP-3D frames")

    img_dir = os.path.join(args.work, "images"); kp_dir = os.path.join(args.work, "openpose")
    out_dir = os.path.join(args.work, "shapy_out")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(kp_dir, exist_ok=True)
    import shutil
    for i in idx:
        stem = os.path.splitext(str(fn[i]))[0]
        shutil.copy(os.path.join(SSP, "images", str(fn[i])), os.path.join(img_dir, str(fn[i])))
        json.dump({"version": 1.3, "people": [{"person_id": [-1],
                   "pose_keypoints_2d": coco17_to_body25(j2d[i]).reshape(-1).tolist()}]},
                  open(os.path.join(kp_dir, stem + ".json"), "w"))

    # call SHAPY demo
    cmd = ["python", "demo.py", "--save-params", "true", "--save-vis", "false", "--save-mesh", "false",
           "--split", "test", "--datasets", "openpose", "--output-folder", out_dir,
           "--exp-cfg", "configs/b2a_expose_hrnet_demo.yaml",
           "--exp-opts", "output_folder=../data/trained_models/shapy/SHAPY_A", "part_key=pose",
           "network.smplx.compute_measurements=False",  # skip CUDA mesh-mesh op; we only need betas
           f"datasets.pose.openpose.data_folder={args.work}",
           "datasets.pose.openpose.img_folder=images", "datasets.pose.openpose.keyp_folder=openpose",
           "datasets.batch_size=1", "datasets.pose_shape_ratio=1.0"]
    print("running SHAPY demo ...")
    subprocess.run(cmd, cwd=SHAPY, check=True)

    # collect betas per image -> align to all 311 rows
    betas = np.full((len(fn), 10), np.nan, np.float32)
    got = 0
    for i in idx:
        stem = os.path.splitext(str(fn[i]))[0]
        hits = glob.glob(os.path.join(out_dir, "**", stem + ".npz"), recursive=True)
        if not hits:
            continue
        d = np.load(hits[0], allow_pickle=True)
        b = d["betas"] if "betas" in d.files else None
        if b is None:
            continue
        betas[i] = np.asarray(b).reshape(-1)[:10]; got += 1
    np.savez(os.path.join(REPO, "runs", "SHAPY_ssp3d.npz"), betas=betas, fnames=fn, idx=np.array(idx))
    print(f"saved runs/SHAPY_ssp3d.npz  ({got}/{len(idx)} betas)")


if __name__ == "__main__":
    main()
