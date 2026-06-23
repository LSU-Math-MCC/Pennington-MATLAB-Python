"""Run BLADE on all frames of one subject (one model load) and save the fused SMPL-X shape betas.

Used by the pairwise inter-method matrix. Robust-mean over frames (down-weight outliers).
Run (blade_env):  cd ~/blade && python <this> --images-dir <dir> --out runs/blade_<subj>_betas.npy
"""
import os, sys, glob, shutil, argparse
import numpy as np

BLADE = os.path.expanduser("~/blade")
sys.path.insert(0, BLADE)


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
    args = ap.parse_args()
    os.environ.setdefault("MINI_BATCHSIZE", "1")
    os.chdir(BLADE)
    import torch
    from api.BLADE_API import BLADE_API

    imgs = sorted(sum([glob.glob(os.path.join(args.images_dir, e))
                       for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")], []))
    print(f"BLADE on {len(imgs)} frames of {args.images_dir}")
    tmp = os.path.join(os.path.dirname(args.out), "_blade_subj_tmp")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    batch_list = {f"{i}": {"rgb_file": os.path.abspath(p)} for i, p in enumerate(imgs)}
    BLADE_API(batch_list=batch_list, render_and_save_imgs=False, temp_output_folder=tmp,
              device="cuda:0", samples_per_gpu=1, workers_per_gpu=0).process()   # ONE model load

    betas = []
    for i in range(len(imgs)):
        pth = os.path.join(tmp, f"{i}.pth")
        if not os.path.exists(pth):
            continue
        b = torch.load(pth, map_location="cpu").get("smplx_betas")
        if b is not None:
            betas.append(np.asarray(b).reshape(-1)[:10])
    if not betas:
        sys.exit("no BLADE betas produced")
    np.save(args.out, robust_mean(betas).astype(np.float32))
    print(f"saved {args.out} (fused over {len(betas)} frames)")


if __name__ == "__main__":
    main()
