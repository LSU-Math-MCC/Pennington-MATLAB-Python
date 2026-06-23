"""Crop images by extending the full target box outward until each side hits another person.

BLADE is a close-range single-subject method; feeding it frames with OTHER people confuses its
single-subject pipeline. The crop is intentionally simple: start with the full target bbox, then
each side goes outward to the first other-person box it would hit in that direction, or to the image
edge if it hits no one. The target box is always fully included.

Run (camerahmr env, has detectron2):
  python tools/render/person_crop.py --images-dir <in> --out-dir <out>
"""
import os, glob, argparse
import numpy as np
import math


def _overlap_1d(a1, a2, b1, b2):
    return a1 < b2 and a2 > b1


def largest_box_excluding_others(boxes, W, H, target):
    """Return [left, top, right, bottom] while always including the full target bbox."""
    boxes = np.asarray(boxes, dtype=float)
    tx1, ty1, tx2, ty2 = boxes[target]

    L, T, R, B = 0.0, 0.0, float(W), float(H)
    for k, b in enumerate(boxes):
        if k == target:
            continue
        ox1, oy1, ox2, oy2 = tuple(b)
        if _overlap_1d(ox1, ox2, tx1, tx2) and _overlap_1d(oy1, oy2, ty1, ty2):
            continue

        if _overlap_1d(oy1, oy2, ty1, ty2):
            if ox1 <= tx1 <= ox2:
                L = max(L, tx1)
            elif ox2 <= tx1:
                L = max(L, ox2)
            if ox1 <= tx2 <= ox2:
                R = min(R, tx2)
            elif ox1 >= tx2:
                R = min(R, ox1)
        if _overlap_1d(ox1, ox2, tx1, tx2):
            if oy1 <= ty1 <= oy2:
                T = max(T, ty1)
            elif oy2 <= ty1:
                T = max(T, oy2)
            if oy1 <= ty2 <= oy2:
                B = min(B, ty2)
            elif oy1 >= ty2:
                B = min(B, oy1)

    L = max(0.0, min(L, tx1))
    T = max(0.0, min(T, ty1))
    R = min(float(W), max(R, tx2))
    B = min(float(H), max(B, ty2))
    return (
        max(0, min(int(math.floor(L)), int(W))),
        max(0, min(int(math.floor(T)), int(H))),
        max(0, min(int(math.ceil(R)), int(W))),
        max(0, min(int(math.ceil(B)), int(H))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import cv2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    det = DefaultPredictor(cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    imgs = sorted(sum([glob.glob(os.path.join(args.images_dir, e))
                       for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")], []))
    for p in imgs:
        im = cv2.imread(p); H, W = im.shape[:2]
        inst = det(im)["instances"]
        keep = (inst.pred_classes == 0) & (inst.scores > 0.5)
        boxes = inst.pred_boxes.tensor[keep].cpu().numpy()
        stem = os.path.splitext(os.path.basename(p))[0]
        if len(boxes) == 0:
            cv2.imwrite(os.path.join(args.out_dir, stem + ".png"), im); print(f"  {stem}: no person, kept full"); continue
        t = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        L, T, R, B = largest_box_excluding_others(boxes, W, H, t)
        cv2.imwrite(os.path.join(args.out_dir, stem + ".png"), im[T:B, L:R])
        frac = ((R - L) * (B - T)) / (W * H)
        print(f"  {stem}: {len(boxes)} persons -> crop [{L},{T},{R},{B}] = {frac*100:.0f}% of {W}x{H}")


if __name__ == "__main__":
    main()
