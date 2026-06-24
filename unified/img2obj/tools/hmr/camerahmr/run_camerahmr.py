"""CameraHMR backend runner (conda env: camerahmr, cwd: ~/CameraHMR).

Official CameraHMR forward (HumanFoV focal + CameraHMR regressor + the exact full-image camera
conversion from mesh_estimator.py), but with the PUBLIC detectron2 COCO person detector instead
of CameraHMR's gated one (same choice as tools/hmr/camerahmr/camerahmr_subjects.py -- no gated download needed).
Reduces every detected person to the normalized schema: camera-space verts, faces, focal, size.
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

import os, sys, argparse, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import camerahmr_root  # noqa: E402
sys.path.insert(0, str(camerahmr_root()))
import schema  # noqa: E402

IMG_SIZE, IMAGE_MEAN, IMAGE_STD = 256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def resize_image(img, t):
    h, w = img.shape[:2]; ar = w / h
    nw, nh = (t, int(t / ar)) if ar > 1 else (int(t * ar), t)
    r = __import__("cv2").resize(img, (nw, nh), interpolation=1)
    out = np.ones((t, t, 3), np.uint8) * 255
    sx, sy = (t - nw) // 2, (t - nh) // 2
    out[sy:sy + nh, sx:sx + nw] = r
    return out


def full_img_cam(pare_cam, bbox_h, bbox_c, W, H, fl):
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    tz = 2. * fl / (bbox_h * s)
    cx = 2. * (bbox_c[:, 0] - W / 2.) / (s * bbox_h)
    cy = 2. * (bbox_c[:, 1] - H / 2.) / (s * bbox_h)
    return torch.stack([tx + cx, ty + cy, tz], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    from torchvision.transforms import Normalize
    from core.camerahmr_model import CameraHMR
    from core.cam_model.fl_net import FLNet
    from core.datasets.dataset import Dataset
    from core.utils import recursive_to
    from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, NUM_BETAS, SMPL_MODEL_PATH
    import smplx
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg

    print("[camerahmr] loading models", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    model = CameraHMR.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location="cpu",
        strict=False,
        model_type="smpl",
    ).to(dev).eval()
    cam_model = FLNet()
    cam_model.load_state_dict(torch.load(CAM_MODEL_CKPT, map_location="cpu")["state_dict"])
    cam_model = cam_model.to(dev).eval()
    body = smplx.SMPLLayer(model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS).to(dev)
    faces = body.faces
    norm = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    print("[camerahmr] loading detector", flush=True)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    det = DefaultPredictor(cfg)

    for img_path in args.images:
        print(f"[camerahmr] processing {img_path}", flush=True)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  !! unreadable {img_path}"); continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        inst = det(bgr)["instances"]
        keep = (inst.pred_classes == 0) & (inst.scores > 0.5)
        boxes = inst.pred_boxes.tensor[keep].cpu().numpy()
        if len(boxes) == 0:
            schema.save(os.path.join(args.out, stem + ".npz"), img_path, W, H, 1000.0, [], faces)
            print(f"  {stem}: 0 people"); continue
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        # focal via HumanFoV
        rim = resize_image(rgb, IMG_SIZE).astype("float32").transpose(2, 0, 1) / 255.
        rim = norm(torch.from_numpy(rim).float()).unsqueeze(0).to(dev)
        with torch.no_grad():
            fov, _ = cam_model(rim)
        fl = (H / (2 * torch.tan(fov[0, 1] / 2))).item()
        cam_int = np.array([[fl, 0, W / 2], [0, fl, H / 2], [0, 0, 1]], np.float32)
        ds = Dataset(rgb, centers, scales, cam_int, False, img_path)
        people = []
        for batch in torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False):
            print(f"[camerahmr] inference batch {len(people) + 1}", flush=True)
            batch = recursive_to(batch, dev)
            with torch.no_grad():
                params, cam, _ = model(batch)
                verts = body(**{k: v.float() for k, v in params.items()}).vertices
            cam_t = full_img_cam(cam, batch["box_size"], batch["box_center"], W, H, batch["cam_int"][:, 0, 0])
            people.extend(list((verts + cam_t.unsqueeze(1)).cpu().numpy()))
        schema.save(os.path.join(args.out, stem + ".npz"), img_path, W, H, fl, people, faces)
        print(f"  {stem}: {len(people)} people, focal {fl:.0f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
