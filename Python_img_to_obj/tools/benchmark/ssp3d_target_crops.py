"""Create target-aware SSP-3D crops for single-subject methods.

Uses SSP-3D's GT target box to choose the matching detected person, then writes the largest crop
that contains that target box while excluding other detected people.

Run (camerahmr env, has detectron2):
  python tools/benchmark/ssp3d_target_crops.py --indices 0 1 2 --out runs/ssp3d_target_crops
  python tools/benchmark/ssp3d_target_crops.py --all-subjects --out runs/ssp3d_target_crops
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

import argparse
import os

import numpy as np

from person_crop import largest_box_excluding_others
from ssp3d_subjects import select_rep_frames, subject_groups


REPO = _repo
SSP = os.path.expanduser("~/SSP-3D/data_ext/ssp_3d")
SMPL_DIR = os.path.expanduser("~/shapy/data/body_models/smpl")


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + ba - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", nargs="*", type=int, default=None,
                    help="explicit SSP-3D row indices to crop")
    ap.add_argument("--all-subjects", action="store_true",
                    help="crop one representative frame for every unique SSP-3D subject")
    ap.add_argument("--n-subjects", type=int, default=8,
                    help="representative diverse/skinniest subjects when --indices is omitted")
    ap.add_argument("--skinny", action="store_true", help="use skinniest reps instead of diverse reps")
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "ssp3d_target_crops"))
    ap.add_argument("--target-union-gt", action="store_true",
                    help="expand the chosen detector target box to include the SSP-3D GT box")
    args = ap.parse_args()

    import cv2
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    lab = np.load(os.path.join(SSP, "labels.npz"))
    fn, ctr, whs, gt = lab["fnames"], lab["bbox_centres"], lab["bbox_whs"], lab["shapes"]
    if args.indices:
        indices = args.indices
        tag = f"{len(indices)} explicit indices"
    elif args.all_subjects:
        indices = [m[0] for m in subject_groups(gt)]
        tag = f"all {len(indices)} unique subjects"
    else:
        indices = select_rep_frames(gt, args.n_subjects, skinny=args.skinny, smpl_dir=SMPL_DIR)
        tag = f"{len(indices)} {'skinny' if args.skinny else 'diverse'} representative subjects"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    det = DefaultPredictor(cfg)

    os.makedirs(args.out, exist_ok=True)
    print(f"target-aware crops for {tag}")
    for i in indices:
        src = os.path.join(SSP, "images", str(fn[i]))
        im = cv2.imread(src)
        if im is None:
            print(f"  !! unreadable {src}")
            continue
        H, W = im.shape[:2]
        cx, cy = ctr[i]
        wh = float(whs[i])
        gt_box = np.array([cx - wh / 2, cy - wh / 2, cx + wh / 2, cy + wh / 2], dtype=float)
        inst = det(im)["instances"]
        keep = (inst.pred_classes == 0) & (inst.scores > 0.5)
        boxes = inst.pred_boxes.tensor[keep].cpu().numpy()
        if len(boxes) == 0:
            boxes = gt_box[None]
        ious = np.array([_iou(b, gt_box) for b in boxes])
        if ious.max() < 0.05:
            boxes = np.vstack([boxes, gt_box])
            target = len(boxes) - 1
        else:
            target = int(np.argmax(ious))
            if args.target_union_gt:
                x1, y1, x2, y2 = boxes[target]
                gx1, gy1, gx2, gy2 = gt_box
                boxes[target] = [min(x1, gx1), min(y1, gy1), max(x2, gx2), max(y2, gy2)]

        L, T, R, B = largest_box_excluding_others(boxes, W, H, target)
        stem = os.path.splitext(str(fn[i]))[0]
        out = os.path.join(args.out, stem + ".png")
        cv2.imwrite(out, im[T:B, L:R])
        frac = ((R - L) * (B - T)) / (W * H)
        print(f"  {stem}: target det {target}/{len(boxes)} -> crop [{L},{T},{R},{B}] = {frac*100:.0f}%")


if __name__ == "__main__":
    main()
