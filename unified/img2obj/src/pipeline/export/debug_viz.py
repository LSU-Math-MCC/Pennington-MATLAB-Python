"""Visual debug artifacts at every major stage."""
from __future__ import annotations

import numpy as np

from ..io import save_image


def _to_rgb(img):
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a[:, :, :3]


def overlay_mask(image, mask, color=(255, 60, 60), alpha=0.5):
    img = _to_rgb(image).astype(np.float32).copy()
    m = np.asarray(mask, bool)
    col = np.array(color, np.float32)
    img[m] = (1 - alpha) * img[m] + alpha * col
    return img.astype(np.uint8)


def depth_preview(depth):
    d = np.asarray(depth, np.float64)
    finite = np.isfinite(d) & (d > 0)
    out = np.zeros_like(d)
    if finite.any():
        lo, hi = np.percentile(d[finite], [2, 98])
        out = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    # turbo-ish colormap
    out = np.nan_to_num(out, nan=0.0)
    r = np.clip(1.5 - np.abs(4 * out - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * out - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * out - 1), 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    rgb[~finite] = 0
    return rgb


def draw_points(image, uv, color=(60, 255, 60), radius=1):
    img = _to_rgb(image).astype(np.uint8).copy()
    H, W = img.shape[:2]
    uv = np.asarray(uv)
    for u, v in uv:
        ui, vi = int(round(u)), int(round(v))
        y0, y1 = max(0, vi - radius), min(H, vi + radius + 1)
        x0, x1 = max(0, ui - radius), min(W, ui + radius + 1)
        img[y0:y1, x0:x1] = color
    return img


def draw_pose(image, keypoints: dict, edges=None, color=(60, 200, 255), conf_thresh=0.3):
    img = _to_rgb(image).astype(np.uint8).copy()
    H, W = img.shape[:2]
    # scale line/joint size with image size so overlays read clearly at any res
    r = max(2, int(round(min(H, W) / 180)))
    lw = max(2, int(round(min(H, W) / 320)))
    try:
        import cv2
        if edges:
            for a, b in edges:
                if a in keypoints and b in keypoints and \
                        keypoints[a][2] > conf_thresh and keypoints[b][2] > conf_thresh:
                    pa = tuple(int(v) for v in keypoints[a][:2])
                    pb = tuple(int(v) for v in keypoints[b][:2])
                    cv2.line(img, pa, pb, (50, 220, 90), lw, cv2.LINE_AA)
        for name, (x, y, c) in keypoints.items():
            if c <= conf_thresh:
                continue
            cv2.circle(img, (int(x), int(y)), r + 1, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), r, (255, 90, 70), -1, cv2.LINE_AA)
    except Exception:
        uv = np.array([[v[0], v[1]] for v in keypoints.values() if v[2] > conf_thresh])
        if uv.size:
            img = draw_points(img, uv, color, 2)
    return img


def save(path, img):
    save_image(path, img)
