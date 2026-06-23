"""Run SHAPY on all frames of one subject and save the fused SMPL-X shape betas.

SHAPY needs OpenPose BODY-25 keypoints; in-the-wild subject frames have none, so we generate them
with MediaPipe Pose (33 landmarks -> BODY-25). Then SHAPY's openpose demo runs as usual.

Run (shapy env):  cd ~/shapy/regressor && python <this> --images-dir <dir> --out runs/shapy_<subj>_betas.npy
"""
import os, sys, json, glob, shutil, subprocess, argparse
import numpy as np

SHAPY = os.path.expanduser("~/shapy/regressor")
# MediaPipe Pose landmark index -> OpenPose BODY-25 index
MP2OP = {0: 0, 5: 15, 2: 16, 8: 17, 7: 18, 12: 2, 14: 3, 16: 4, 11: 5, 13: 6, 15: 7,
         24: 9, 26: 10, 28: 11, 23: 12, 25: 13, 27: 14, 31: 19, 29: 21, 32: 22, 30: 24}


def mp_to_body25(lms, w, h):
    b = np.zeros((25, 3), np.float32)
    P = {i: (lm.x * w, lm.y * h, lm.visibility) for i, lm in enumerate(lms)}
    for mp_i, op_i in MP2OP.items():
        if mp_i in P:
            b[op_i] = P[mp_i]
    b[1] = ((P[11][0] + P[12][0]) / 2, (P[11][1] + P[12][1]) / 2, min(P[11][2], P[12][2]))   # neck
    b[8] = ((P[23][0] + P[24][0]) / 2, (P[23][1] + P[24][1]) / 2, min(P[23][2], P[24][2]))   # mid-hip
    return b


def robust_mean(b, iters=5):
    b = np.asarray(b); mu = b.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(b - mu, axis=1); s = np.median(d) + 1e-6
        w = np.clip(1 - (d / (2.5 * s)) ** 2, 0, 1) ** 2
        if w.sum() < 1e-9:
            w = np.ones(len(b))
        mu = (b * (w / w.sum())[:, None]).sum(0)
    return mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=os.path.expanduser("~/shapy_subj_work"))
    args = ap.parse_args()
    import cv2, mediapipe as mp

    imgs = sorted(sum([glob.glob(os.path.join(args.images_dir, e))
                       for e in ("*.jpg", "*.jpeg", "*.png")], []))
    img_dir = os.path.join(args.work, "images"); kp_dir = os.path.join(args.work, "openpose")
    out_dir = os.path.join(args.work, "shapy_out")
    for d in (img_dir, kp_dir):
        os.makedirs(d, exist_ok=True)
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    used = 0
    for p in imgs:
        im = cv2.imread(p); h, w = im.shape[:2]
        res = pose.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            print(f"  no pose in {os.path.basename(p)}"); continue
        stem = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(img_dir, stem + ".png"), im)
        json.dump({"version": 1.3, "people": [{"person_id": [-1],
                   "pose_keypoints_2d": mp_to_body25(res.pose_landmarks.landmark, w, h).reshape(-1).tolist()}]},
                  open(os.path.join(kp_dir, stem + ".json"), "w"))
        used += 1
    print(f"generated keypoints for {used}/{len(imgs)} frames")

    subprocess.run(["python", "demo.py", "--save-params", "true", "--save-vis", "false", "--save-mesh", "false",
                    "--split", "test", "--datasets", "openpose", "--output-folder", out_dir,
                    "--exp-cfg", "configs/b2a_expose_hrnet_demo.yaml",
                    "--exp-opts", "output_folder=../data/trained_models/shapy/SHAPY_A", "part_key=pose",
                    "network.smplx.compute_measurements=False",
                    f"datasets.pose.openpose.data_folder={args.work}",
                    "datasets.pose.openpose.img_folder=images", "datasets.pose.openpose.keyp_folder=openpose",
                    "datasets.batch_size=1", "datasets.pose_shape_ratio=1.0"], cwd=SHAPY, check=True)

    betas = []
    for npz in glob.glob(os.path.join(out_dir, "**", "*.npz"), recursive=True):
        d = np.load(npz, allow_pickle=True)
        if "betas" in d.files:
            betas.append(np.asarray(d["betas"]).reshape(-1)[:10])
    if not betas:
        sys.exit("no SHAPY betas produced")
    np.save(args.out, robust_mean(betas).astype(np.float32))
    print(f"saved {args.out} (fused over {len(betas)} frames)")


if __name__ == "__main__":
    main()
