"""Official-CameraHMR SSP-3D shape eval (PVE-T-SC).

Uses CameraHMR's OFFICIAL inference forward (core.camerahmr_model.CameraHMR + its cam_model
FLNet for focal length, exactly as mesh_estimator.py does) — only the person *detector* is
replaced by SSP-3D's ground-truth bbox (standard shape-eval protocol; detector recall must not
contaminate a shape metric). The PVE-T-SC computation is faithful to SSP-3D's official metrics.py
(scale-corrected per-vertex T-pose error). If CameraHMR scores ~11.6 here, we've honestly
reproduced SOTA and the number is trustworthy.

Phase 1 saves per-image betas + reliability features (for the later SKF multi-view fusion).

Run (camerahmr env): cd ~/CameraHMR && python <this> [N]
"""
import os, sys
import numpy as np
import torch
import cv2

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
SSP = os.path.expanduser("~/SSP-3D/data_ext/ssp_3d")
SMPL_DIR = os.path.expanduser("~/shapy/data/body_models/smpl")  # gendered SMPL (validated harness)
sys.path.insert(0, os.path.expanduser("~/CameraHMR"))
IMG_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def scale_trans_align(P, T):
    Pm = P.mean(1, keepdims=True); Pt = P - Pm
    Ps = np.sqrt((Pt ** 2).sum((1, 2), keepdims=True) / P.shape[1]); Pn = Pt / Ps
    Tm = T.mean(1, keepdims=True); Ts = np.sqrt(((T - Tm) ** 2).sum((1, 2), keepdims=True) / T.shape[1])
    return Pn * Ts + Tm


def resize_image(img, target_size):
    h, w = img.shape[:2]
    ar = w / h
    if ar > 1:
        nw, nh = target_size, int(target_size / ar)
    else:
        nw, nh = int(target_size * ar), target_size
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    sx, sy = (target_size - nw) // 2, (target_size - nh) // 2
    out[sy:sy + nh, sx:sx + nw] = r
    return out


def main():
    from torchvision.transforms import Normalize
    from core.camerahmr_model import CameraHMR
    from core.cam_model.fl_net import FLNet
    from core.datasets.dataset import Dataset
    from core.utils import recursive_to
    from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT
    from smplx import SMPL

    dev = torch.device("cuda")
    model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False, model_type="smpl").to(dev).eval()
    cam_model = FLNet()
    cam_model.load_state_dict(torch.load(CAM_MODEL_CKPT)["state_dict"])
    cam_model = cam_model.to(dev).eval()
    normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    smpl = {"m": SMPL(SMPL_DIR, gender="male"), "f": SMPL(SMPL_DIR, gender="female")}

    lab = np.load(os.path.join(SSP, "labels.npz"))
    fn, shp, gen, ctr, whs = lab["fnames"], lab["shapes"], lab["genders"], lab["bbox_centres"], lab["bbox_whs"]
    N = int(sys.argv[1]) if len(sys.argv) > 1 else len(fn)
    idx = np.arange(len(fn))[:N]

    betas_all = np.zeros((len(fn), 10), np.float32)
    feat = np.zeros((len(fn), 3), np.float32)  # [focal_h, bbox_size_px, img_diag] reliability features
    for c, i in enumerate(idx):
        img = cv2.imread(os.path.join(SSP, "images", str(fn[i])))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        # focal via CameraHMR's HumanFoV cam model (official path)
        rim = resize_image(img, IMG_SIZE).astype("float32").transpose(2, 0, 1) / 255.0
        rim = normalize_img(torch.from_numpy(rim).float()).unsqueeze(0).to(dev)
        with torch.no_grad():
            fov, _ = cam_model(rim)
        vfov = fov[0, 1]
        fl_h = (H / (2 * torch.tan(vfov / 2))).item()
        cam_int = np.array([[fl_h, 0, W / 2], [0, fl_h, H / 2], [0, 0, 1]], np.float32)

        wh = float(whs[i])
        ds = Dataset(img, np.array([ctr[i]], np.float32), np.array([[wh / 200., wh / 200.]], np.float32),
                     cam_int, False, str(fn[i]))
        for batch in torch.utils.data.DataLoader(ds, batch_size=1):
            batch = recursive_to(batch, dev)
            with torch.no_grad():
                params, _, _ = model(batch)
            betas_all[i] = params["betas"][0, :10].cpu().numpy()
        feat[i] = [fl_h, wh, (H * H + W * W) ** 0.5]
        if c % 40 == 0:
            print(f"  CameraHMR {c}/{len(idx)}  betas[:3]={betas_all[i][:3].round(2)}")

    # PVE-T-SC single-image (validated protocol: gendered SMPL T-pose, scale-corrected)
    def pve(betas_i):
        out = []
        for i in idx:
            g = str(gen[i]); m = smpl[g]
            with torch.no_grad():
                pv = m(betas=torch.tensor(betas_i[i][:10]).float().unsqueeze(0)).vertices.numpy()
                gv = m(betas=torch.tensor(shp[i][:10]).float().unsqueeze(0)).vertices.numpy()
            out.append(np.linalg.norm(scale_trans_align(pv, gv) - gv, axis=-1).mean() * 1000)
        return np.array(out)

    single = pve(betas_all)
    np.savez(REPO + "/runs/CAMERAHMR_ssp3d.npz", betas=betas_all, feat=feat, fnames=fn,
             shapes=shp, genders=gen, pve_single=single, idx=idx)
    print(f"\n#images={len(idx)}")
    print(f"CameraHMR single-image PVE-T-SC = {single.mean():.2f} mm   (published SOTA 11.6)")
    print("saved runs/CAMERAHMR_ssp3d.npz")


if __name__ == "__main__":
    main()
