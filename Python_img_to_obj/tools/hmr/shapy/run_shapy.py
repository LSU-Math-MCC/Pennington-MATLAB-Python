"""Honest SHAPY baseline runner.

SHAPY (Choutas et al., CVPR'22) regresses metric SMPL-X body shape from a single image +
2D keypoints. It is our SHAPE BASELINE — an independent estimator we run for real and FUSE
(inverse-variance, sigma=0.5 = tightest) with Multi-HMR prior + silhouette-fit. We do NOT
fabricate its numbers; if the gated checkpoints are absent this prints the one download the
user must do (their MPI credentials) and exits.

Bridge: SHAPY's image->shape path (SHAPY_A) reads OpenPose BODY_25 JSON per image. Rather
than install OpenPose, we synthesize those keypoints from Multi-HMR's j2d via the standard
SMPL-X->OpenPose-25 joint map (the same map smplx/SMPLify-X use). Confidence = Multi-HMR
detection (1.0 for detected joints). This reuses keypoints we already trust.

Run (WSL):  python tools/hmr/shapy/run_shapy.py --subject "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png" --out runs/fit_ssp3d_bodybuilder [--gender female]
Writes:     <out>/shapy_betas.npy   (then `python tools/geometry/fuse_betas.py <out>` folds it in)
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
import subprocess

import numpy as np

REPO = _repo
SHAPY = os.path.expanduser("~/shapy")
CKPT = os.path.join(SHAPY, "data", "trained_models", "shapy", "SHAPY_A")

# OpenPose BODY_25 index -> SMPL-X joint index (standard smplx/SMPLify-X correspondence).
# 0 Nose,1 Neck,2 RSho,3 RElb,4 RWri,5 LSho,6 LElb,7 LWri,8 MidHip,9 RHip,10 RKnee,11 RAnk,
# 12 LHip,13 LKnee,14 LAnk,15 REye,16 LEye,17 REar,18 LEar,19 LBigToe,20 LSmallToe,21 LHeel,
# 22 RBigToe,23 RSmallToe,24 RHeel
OP25_FROM_SMPLX = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7,
                   57, 56, 59, 58, 60, 61, 62, 63, 64, 65]


def write_openpose_json(j2d, conf, out_path):
    body = np.zeros((25, 3), np.float32)
    n = j2d.shape[0]
    for op, sx in enumerate(OP25_FROM_SMPLX):
        if sx < n:
            body[op, :2] = j2d[sx]
            body[op, 2] = conf[sx] if conf is not None else 1.0
    people = [{"pose_keypoints_2d": body.reshape(-1).tolist(),
               "hand_left_keypoints_2d": [0.0] * 63,
               "hand_right_keypoints_2d": [0.0] * 63,
               "face_keypoints_2d": [0.0] * 210}]
    json.dump({"people": people}, open(out_path, "w"))


def stage_inputs(subject, workdir):
    """Detect each image with Multi-HMR, write img + OpenPose-25 JSON into SHAPY layout."""
    sys.path.insert(0, REPO + "/tools")
    import lhm_anthropometry as A
    import texture_uv_bake as TB
    from PIL import Image
    img_dir = os.path.join(workdir, "images"); kp_dir = os.path.join(workdir, "openpose")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(kp_dir, exist_ok=True)
    if any(ch in subject for ch in "*?[]"):
        imgs = sorted(glob.glob(subject))
    else:
        imgs = (sorted(sum([glob.glob(os.path.join(subject, "**", e), recursive=True)
                            for e in ("*.jpg", "*.jpeg", "*.png", "*.webp")], []))
                if os.path.isdir(subject) else [subject])
    est = A._estimator(); n = 0
    for ip in imgs:
        img, h, j2d = TB.posed_view(est, ip)
        if h is None:
            continue
        fn = f"img{n:03d}"
        Image.fromarray(img).save(os.path.join(img_dir, fn + ".png"))
        conf = h.get("j2d_conf")
        write_openpose_json(np.asarray(j2d), conf, os.path.join(kp_dir, fn + "_keypoints.json"))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gender", default="female")
    args = ap.parse_args()

    if not os.path.isdir(CKPT):
        print("SHAPY_GATED: checkpoints missing at", CKPT)
        print("  ONE credentialed download (MPI account, like the SMPL file):")
        print("    wsl bash -lc 'cd ~/shapy/data && bash download_data.sh'")
        print("  enter your shapy.is.tue.mpg.de username/password, then re-run this.")
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "shapy_in")
    n = stage_inputs(args.subject, work)
    if n == 0:
        print("SHAPY_FAIL: no detected views"); sys.exit(1)
    print(f"staged {n} views -> {work}")

    demo_out = os.path.join(args.out, "shapy_out")
    cmd = ["python", "demo.py", "--save-params", "true", "--save-vis", "false",
           "--save-mesh", "false", "--split", "test", "--datasets", "openpose",
           "--output-folder", demo_out, "--exp-cfg", "configs/b2a_expose_hrnet_demo.yaml",
           "--exp-opts",
           f"output_folder={SHAPY}/data/trained_models/shapy/SHAPY_A", "part_key=pose",
           f"datasets.pose.openpose.data_folder={work}",
           "datasets.pose.openpose.img_folder=images",
           "datasets.pose.openpose.keyp_folder=openpose",
           "datasets.batch_size=1", "datasets.pose_shape_ratio=1.0"]
    subprocess.run(cmd, cwd=os.path.join(SHAPY, "regressor"), check=True)

    # collect per-view betas, average (each view = one noisy SHAPY observation)
    npzs = glob.glob(os.path.join(demo_out, "**", "*.npz"), recursive=True)
    betas = []
    for f in npzs:
        d = np.load(f)
        if "betas" in d:
            betas.append(np.asarray(d["betas"]).reshape(-1))
    if not betas:
        print("SHAPY_FAIL: no betas in", demo_out); sys.exit(1)
    nb = min(len(b) for b in betas)
    fused = np.mean([b[:nb] for b in betas], 0)
    np.save(os.path.join(args.out, "shapy_betas.npy"), fused)
    print(f"SHAPY_OK views={len(betas)} betas[:5]={np.round(fused[:5],3)} -> {args.out}/shapy_betas.npy")


if __name__ == "__main__":
    main()
