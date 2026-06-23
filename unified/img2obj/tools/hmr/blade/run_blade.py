"""BLADE backend runner (conda env: blade_env, cwd: ~/blade).

LOADS BLADE ONCE and processes all images in a single process (model load = Sapiens-1B +
Depth-Anything + AiOS is ~2 min, so per-image reload was the bottleneck). Constructing BLADE_API
twice in a process double-registers mmcv modules, so we never do that -- one construction, one run.

Output per image (`<stem>.overlay.png`): BLADE's SMPL-X mesh composited onto the FULL original
(padded) input image with real background, plus the subject marked in blue. Default marker is a box;
use `--mark outline` for the older segmentation contour.
"""
import os, sys, glob, argparse, shutil
import numpy as np
from PIL import Image

BLADE = os.path.expanduser("~/blade")
sys.path.insert(0, BLADE)


def _compose(side_jpg, out_path, style="box"):
    """BLADE side-by-side jpg = [ full original image (top) ; gray mesh on white (bottom) ].
    Extract the gray mesh from the bottom (low-saturation, non-white) and composite it onto the
    original. Mark the subject in blue as either a `box` (bounding box, default) or an `outline`
    (segmentation contour)."""
    import cv2
    side = np.array(Image.open(side_jpg).convert("RGB"))
    H = side.shape[0] // 2
    orig_u8 = side[:H].copy()                                # full padded input image, real background
    orig = orig_u8.astype(np.float32)
    bot = side[H:]                                           # gray SMPL-X mesh on white (+ faint ghost)
    mx = bot.max(2).astype(np.int16); mn = bot.min(2).astype(np.int16)
    mesh = ((mx - mn) < 28) & (mx < 240)                    # gray + not-white -> the rendered mesh
    seg = (mx < 246)                                         # non-white -> subject region
    a = 0.85
    comp = np.where(mesh[..., None], bot.astype(np.float32) * a + orig * (1 - a), orig).astype(np.uint8)
    seg_u = cv2.morphologyEx((seg.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(seg_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 400]
    if style == "outline":
        cv2.drawContours(comp, cnts, -1, (0, 90, 255), 3)               # blue segmentation outline
    else:
        # BLADE pads non-square crops before rendering. The default blue box marks the usable
        # subject-only input crop, clipped to real image pixels so it never includes black padding.
        nonpad = orig_u8.max(2) > 8
        rows = np.where(nonpad.mean(1) > 0.02)[0]
        cols = np.where(nonpad.mean(0) > 0.02)[0]
        if len(rows) and len(cols):
            x1, x2 = int(cols[0]), int(cols[-1])
            y1, y2 = int(rows[0]), int(rows[-1])
            cv2.rectangle(comp, (x1, y1), (x2, y2), (0, 90, 255), 3)
        elif cnts:
            allpts = np.vstack(cnts); x, y, w, h = cv2.boundingRect(allpts)
            cv2.rectangle(comp, (x, y), (x + w, y + h), (0, 90, 255), 3)
    Image.fromarray(comp).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1, help="reserved; overlays run robustly at batch=1")
    ap.add_argument("--mark", choices=("box", "outline"), default="box", help="blue subject marker style")
    args = ap.parse_args()
    os.environ.setdefault("MINI_BATCHSIZE", "1")
    os.chdir(BLADE)
    from api.BLADE_API import BLADE_API

    images = [os.path.abspath(os.path.expanduser(p)) for p in args.images]
    batch_list, stems = {}, []
    for p in images:
        s = os.path.splitext(os.path.basename(p))[0]
        batch_list[s] = {"rgb_file": p}; stems.append(s)     # key = stem -> cur_id/.pth = stem
    tmp = os.path.join(args.out, "_tmp")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    # samples_per_gpu=1: one model load, images processed sequentially. BLADE inference is ~60s/img
    # so each render_overlay gets a distinct (second-resolution) timestamp -> sort by mtime to recover
    # the dataset/id_list order and map to stems.
    BLADE_API(batch_list=batch_list, render_and_save_imgs=True, temp_output_folder=tmp,
              device="cuda:0", samples_per_gpu=1, workers_per_gpu=0).process()   # ONE model load
    jpgs = [j for j in sorted(glob.glob(os.path.join(tmp, "*_0.jpg")), key=os.path.getmtime)]
    n_done = 0
    for gi, jpg in enumerate(jpgs):
        if gi >= len(stems):
            break
        _compose(jpg, os.path.join(args.out, stems[gi] + ".overlay.png"), style=args.mark)
        n_done += 1
    print(f"  BLADE: {n_done}/{len(images)} overlays (mesh-on-image + blue {args.mark}) in one load")


if __name__ == "__main__":
    main()
