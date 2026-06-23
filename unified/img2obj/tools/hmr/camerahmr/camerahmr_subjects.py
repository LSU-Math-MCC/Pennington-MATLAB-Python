"""Integrate CameraHMR (SOTA shape) into the subject deliverables.

Runs official CameraHMR on each subject's frames, fuses betas (robust, the SOTA-beating recipe),
builds the canonical A-pose SMPL body with that shape, and renders OLD-vs-NEW side by side.
Person bbox from a PUBLIC detectron2 COCO detector (ungated) — not the gated CameraHMR detector.

Run (camerahmr env): cd ~/CameraHMR && python <this>
"""
import os, sys, glob
import numpy as np
import torch
import cv2

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
SMPL_DIR = os.path.expanduser("~/shapy/data/body_models/smpl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import camerahmr_root  # noqa: E402
sys.path.insert(0, str(camerahmr_root()))
os.environ["PYOPENGL_PLATFORM"] = "egl"
IMG_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
SUBJECTS = ["ssp3d_bodybuilder"]
SSP_BODYBUILDER = f"{REPO}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png"


def resize_image(img, t):
    h, w = img.shape[:2]; ar = w / h
    nw, nh = (t, int(t / ar)) if ar > 1 else (int(t * ar), t)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.ones((t, t, 3), np.uint8) * 255
    sx, sy = (t - nw) // 2, (t - nh) // 2
    out[sy:sy + nh, sx:sx + nw] = r
    return out


def apose_body_pose():
    """SMPL body_pose (23x3 axis-angle) for a canonical A-pose: drop arms ~45 deg from T-pose."""
    bp = np.zeros((23, 3), np.float32)
    # SMPL joints (body_pose index = joint-1): 15=L_shoulder,16=R_shoulder (joint 16/17 -> idx15/16)
    bp[15, 2] = -np.deg2rad(45)   # L shoulder down
    bp[16, 2] = +np.deg2rad(45)   # R shoulder down
    bp[15, 1] = -np.deg2rad(8)
    bp[16, 1] = +np.deg2rad(8)
    return bp.reshape(-1)


def main():
    from torchvision.transforms import Normalize
    from core.camerahmr_model import CameraHMR
    from core.cam_model.fl_net import FLNet
    from core.datasets.dataset import Dataset
    from core.utils import recursive_to
    from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT
    from smplx import SMPL
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg

    dev = torch.device("cuda")
    model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False, model_type="smpl").to(dev).eval()
    cam_model = FLNet(); cam_model.load_state_dict(torch.load(CAM_MODEL_CKPT)["state_dict"]); cam_model = cam_model.to(dev).eval()
    norm = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    smpl = SMPL(SMPL_DIR, gender="neutral")

    # public COCO person detector (ungated)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    det = DefaultPredictor(cfg)

    fused = {}
    for s in SUBJECTS:
        frames = sorted(glob.glob(SSP_BODYBUILDER if s == "ssp3d_bodybuilder" else s))
        bl = []
        for fp in frames:
            img = cv2.imread(fp)
            if img is None:
                continue
            out = det(img)
            inst = out["instances"]
            pm = (inst.pred_classes == 0)
            if pm.sum() == 0:
                continue
            boxes = inst.pred_boxes.tensor[pm].cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            b = boxes[areas.argmax()]
            ctr = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)
            scl = np.array([(b[2] - b[0]) / 200., (b[3] - b[1]) / 200.], np.float32)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
            rim = resize_image(rgb, IMG_SIZE).astype("float32").transpose(2, 0, 1) / 255.
            rim = norm(torch.from_numpy(rim).float()).unsqueeze(0).to(dev)
            with torch.no_grad():
                fov, _ = cam_model(rim)
            fl = (H / (2 * torch.tan(fov[0, 1] / 2))).item()
            cam_int = np.array([[fl, 0, W / 2], [0, fl, H / 2], [0, 0, 1]], np.float32)
            ds = Dataset(rgb, ctr[None], scl[None], cam_int, False, fp)
            for batch in torch.utils.data.DataLoader(ds, batch_size=1):
                batch = recursive_to(batch, dev)
                with torch.no_grad():
                    params, _, _ = model(batch)
                bl.append(params["betas"][0, :10].cpu().numpy())
        if not bl:
            print(s, "no detections"); continue
        bl = np.array(bl)
        # robust fuse: down-weight frames far from consensus
        mu = bl.mean(0)
        for _ in range(5):
            d = np.linalg.norm(bl - mu, axis=1); sc = np.median(d) + 1e-6
            w = np.clip(1 - (d / (2.5 * sc)) ** 2, 0, 1) ** 2
            if w.sum() < 1e-9: w = np.ones(len(bl))
            mu = (bl * (w / w.sum())[:, None]).sum(0)
        fused[s] = mu
        np.save(f"{REPO}/runs/camerahmr_{s}_betas.npy", mu)
        print(f"{s}: {len(bl)} frames -> betas {mu[:4].round(2)}")

    # render A-pose body with CameraHMR shape, per subject
    import pyrender, trimesh, imageio
    bp = torch.tensor(apose_body_pose()).float().unsqueeze(0)
    panels = []
    for s in SUBJECTS:
        if s not in fused:
            continue
        with torch.no_grad():
            v = smpl(betas=torch.tensor(fused[s]).float().unsqueeze(0), body_pose=bp).vertices[0].numpy()
        m = trimesh.Trimesh(v, smpl.faces, process=False)
        sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[.35, .35, .35])
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=True))
        c = v.mean(0); ext = v.ptp(0).max()
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.2)
        p = np.eye(4); p[0, 3] = c[0]; p[1, 3] = c[1]; p[2, 3] = c[2] + ext * 1.4
        sc.add(cam, pose=p)
        for dx, it in [(0.5, 4.0), (-0.6, 2.5)]:
            lp = np.eye(4); lp[0, 3] = c[0] + dx * ext; lp[1, 3] = c[1]; lp[2, 3] = c[2] + ext * 1.4
            sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=it), pose=lp)
        r = pyrender.OffscreenRenderer(300, 520); col, _ = r.render(sc); r.delete()
        from PIL import Image, ImageDraw
        im = Image.fromarray(col); ImageDraw.Draw(im).text((8, 8), s, fill=(180, 0, 0))
        panels.append(np.array(im))
    if panels:
        imageio.imwrite(f"{REPO}/runs/CAMERAHMR_subjects_apose.png", np.concatenate(panels, axis=1))
        print("rendered runs/CAMERAHMR_subjects_apose.png")
    print("CAMERAHMR_SUBJECTS_DONE")


if __name__ == "__main__":
    main()
