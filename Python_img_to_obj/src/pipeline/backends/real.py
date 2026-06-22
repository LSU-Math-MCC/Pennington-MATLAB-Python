"""Real model backends (CPU-capable):

  segmentation + boxes : Ultralytics YOLOv8-seg (multi-person instance masks)
  2D pose (17 kpts)    : Ultralytics YOLOv8-pose
  dense face landmarks : MediaPipe FaceMesh (468 landmarks) + landmark-hull mask
  monocular depth      : Depth-Anything-V2-Small (transformers), metric-proxy z
  3DGS scene           : colored point splats lifted from monocular depth
  association embedding: color histogram (body appearance)

Everything stays behind the backend interface; if a model can't load, get_backends()
falls back to dummy. Heavy results are memoized per image within a process.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..types import (SplatCloud, SegmentationResult, Pose2DResult, Face2DResult,
                     DepthResult, Camera)
from ..geometry import camera as camlib
from . import base

COCO_KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
           "left_wrist", "right_wrist", "left_hip", "right_hip",
           "left_knee", "right_knee", "left_ankle", "right_ankle"]


def _img_key(image: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(image)).hexdigest()[:16]


# --------------------------------------------------------------- model cache ---
import threading


class _Models:
    """Lazily-loaded, process-wide singletons for the heavy models.

    `_load_lock` serializes first-time construction (concurrent first imports of
    e.g. transformers.pipeline are not safe). `_infer_lock` serializes inference for
    models that are not thread-safe (MediaPipe Tasks, the depth pipeline).
    """
    _yolo_seg = None
    _yolo_pose = None
    _facemesh = None
    _posemark = None
    _depth = None
    _depth_cache: dict = {}
    _load_lock = threading.RLock()
    _infer_lock = threading.Lock()

    @classmethod
    def warmup(cls):
        """Load every singleton up front (call from the main thread before parallel
        jobs) so worker threads never race on first import/construction."""
        with cls._load_lock:
            cls.yolo_seg(); cls.yolo_pose(); cls.facemesh()
            cls.posemark(); cls.depth()

    @classmethod
    def yolo_seg(cls):
        if cls._yolo_seg is None:
            with cls._load_lock:
                if cls._yolo_seg is None:
                    from ultralytics import YOLO
                    cls._yolo_seg = YOLO("yolov8n-seg.pt")
        return cls._yolo_seg

    @classmethod
    def yolo_pose(cls):
        if cls._yolo_pose is None:
            with cls._load_lock:
                if cls._yolo_pose is None:
                    from ultralytics import YOLO
                    cls._yolo_pose = YOLO("yolov8n-pose.pt")
        return cls._yolo_pose

    @classmethod
    def posemark(cls):
        if cls._posemark is None:
            with cls._load_lock:
                if cls._posemark is None:
                    from mediapipe.tasks.python import vision, BaseOptions
                    from mediapipe.tasks.python.vision import RunningMode
                    model_path = str(Path(__file__).resolve().parents[3] / "models" / "pose_landmarker_heavy.task")
                    opts = vision.PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        running_mode=RunningMode.IMAGE, num_poses=1,
                        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
                    cls._posemark = vision.PoseLandmarker.create_from_options(opts)
        return cls._posemark

    @classmethod
    def facemesh(cls):
        if cls._facemesh is None:
            with cls._load_lock:
                if cls._facemesh is None:
                    from mediapipe.tasks.python import vision, BaseOptions
                    from mediapipe.tasks.python.vision import RunningMode
                    model_path = str(Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task")
                    opts = vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        running_mode=RunningMode.IMAGE, num_faces=1,
                        min_face_detection_confidence=0.3)
                    cls._facemesh = vision.FaceLandmarker.create_from_options(opts)
        return cls._facemesh

    _depth_models: dict = {}
    DEPTH_HF = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    @classmethod
    def depth(cls, size="small"):
        if size not in cls._depth_models:
            with cls._load_lock:
                if size not in cls._depth_models:
                    from transformers import pipeline
                    import torch
                    dev = 0 if torch.cuda.is_available() else -1
                    cls._depth_models[size] = pipeline(
                        "depth-estimation", model=cls.DEPTH_HF.get(size, cls.DEPTH_HF["small"]),
                        device=dev)
        return cls._depth_models[size]

    @classmethod
    def depth_map(cls, image: np.ndarray, size="small") -> np.ndarray:
        """Return a metric-proxy z map (HxW float32, smaller = closer)."""
        key = f"{size}:{_img_key(image)}"
        if key in cls._depth_cache:
            return cls._depth_cache[key]
        from PIL import Image
        with cls._infer_lock:
            out = cls.depth(size)(Image.fromarray(image))
        raw = out["predicted_depth"] if "predicted_depth" in out else out["depth"]
        pred = np.squeeze(np.asarray(raw, dtype=np.float32))     # (1,H,W)/(H,W,1) -> (H,W)
        Hh, Ww = int(image.shape[0]), int(image.shape[1])
        if pred.ndim != 2 or pred.shape != (Hh, Ww):
            import cv2
            pred = cv2.resize(np.asarray(pred, np.float32).reshape(pred.shape[-2], pred.shape[-1])
                              if pred.ndim >= 2 else pred,
                              (Ww, Hh), interpolation=cv2.INTER_LINEAR)
        # Depth-Anything predicts disparity-like values (larger = nearer).
        disp = pred - pred.min()
        disp = disp / (disp.max() + 1e-6)               # 0..1, 1=nearest
        z = 1.5 + (1.0 - disp) * 2.5                     # ~1.5 (near) .. 4.0 (far)
        z = z.astype(np.float32)
        if len(cls._depth_cache) > 16:
            cls._depth_cache.clear()
        cls._depth_cache[key] = z
        return z


# ------------------------------------------------------------------- depth -----
class RealDepth(base.DepthBackend):
    name = "depth-anything-v2"
    version = "2"

    def __init__(self, size="small"):
        self.size = size
        self.name = f"depth-anything-v2-{size}"

    def estimate_or_render_depth(self, image, splats, camera, out_dir):
        z = _Models.depth_map(image, size=self.size)
        conf = np.ones_like(z, dtype=np.float32)
        return DepthResult(depth=z, confidence=conf)


# ----------------------------------------------------------------- 3dgs scene --
class RealGS(base.GSReconstructionBackend):
    name = "depth-lift-splats"
    version = "1"

    def __init__(self, max_points=120000, depth_size="small"):
        self.max_points = max_points
        self.depth_size = depth_size

    def reconstruct(self, image: np.ndarray, out_dir: Path):
        H, W = image.shape[:2]
        cam = camlib.default_camera(W, H)
        z = _Models.depth_map(image, size=self.depth_size)
        ys, xs = np.mgrid[0:H, 0:W]
        uv = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
        d = z.ravel().astype(np.float64)
        n = uv.shape[0]
        if n > self.max_points:
            idx = np.linspace(0, n - 1, self.max_points).astype(int)
            uv, d = uv[idx], d[idx]
        centers = camlib.backproject_to_world(cam, uv, d)
        cols = image.reshape(-1, 3)[
            (uv[:, 1].astype(int) * W + uv[:, 0].astype(int))].astype(np.float64) / 255.0
        m = centers.shape[0]
        splats = SplatCloud(centers=centers, scales=np.full((m, 3), 0.01),
                            rotations=np.tile([1.0, 0, 0, 0], (m, 1)),
                            opacities=np.full(m, 0.9), colors=cols, extras={})
        return splats, cam


# ----------------------------------------------------------- segmentation ------
class RealSeg(base.SegmentationBackend):
    name = "yolov8n-seg"
    version = "1"
    _cache: dict = {}

    def segment_people(self, image: np.ndarray, out_dir: Path):
        key = _img_key(image)
        if key in self._cache:
            return self._cache[key]
        H, W = image.shape[:2]
        with _Models._infer_lock:
            res = _Models.yolo_seg()(image[:, :, ::-1], verbose=False)[0]  # BGR in
        out = []
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()             # n x h x w (model res)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            import cv2
            from scipy.ndimage import binary_fill_holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            for i in range(masks.shape[0]):
                if cls[i] != 0 or confs[i] < 0.3:           # person class only
                    continue
                m = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
                # close small gaps, then fill interior holes (YOLO masks are often holey)
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
                m = binary_fill_holes(m).astype(bool)
                if m.sum() < 50:
                    continue
                x0, y0, x1, y1 = boxes[i]
                out.append(SegmentationResult(person_mask=m,
                                              confidence=m.astype(np.float32) * float(confs[i]),
                                              bbox=(float(x0), float(y0), float(x1), float(y1))))
        out.sort(key=lambda s: -s.person_mask.sum())
        self._cache[key] = out
        return out


# ------------------------------------------------------------------ pose -------
# MediaPipe PoseLandmarker 33-landmark index -> name (subject-anatomical L/R).
_MP_POSE = {
    0: "nose", 2: "left_eye", 5: "right_eye", 7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee", 27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel", 31: "left_foot_index", 32: "right_foot_index",
}


class RealPose(base.PoseBackend):
    name = "mediapipe-pose-heavy"
    version = "2"

    def estimate_pose(self, image: np.ndarray, out_dir: Path, bbox=None):
        import cv2
        import mediapipe as mp
        H, W = image.shape[:2]
        if bbox is not None:
            bx0, by0, bx1, by1 = [int(v) for v in bbox]
            mw = int(0.12 * (bx1 - bx0)); mh = int(0.12 * (by1 - by0))
            x0 = max(0, bx0 - mw); y0 = max(0, by0 - mh)
            x1 = min(W, bx1 + mw); y1 = min(H, by1 + mh)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        if crop.size == 0:
            return Pose2DResult(keypoints={}, skeleton_edges=_SKEL)
        # upscale tight crops so distant/small people get accurate landmarks
        long_side = max(crop.shape[:2])
        scale = max(1.0, 512.0 / long_side)
        if scale > 1.0:
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                              interpolation=cv2.INTER_LINEAR)
        ch, cw = crop.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        with _Models._infer_lock:
            res = _Models.posemark().detect(mp_img)
        if not res.pose_landmarks:
            return self._yolo_fallback(image, bbox)
        lms = res.pose_landmarks[0]
        kp = {}
        for idx, name in _MP_POSE.items():
            if idx < len(lms):
                lm = lms[idx]
                conf = float(getattr(lm, "visibility", 0.0)) or float(getattr(lm, "presence", 0.0))
                kp[name] = (x0 + lm.x * cw / scale, y0 + lm.y * ch / scale, conf)
        if sum(1 for v in kp.values() if v[2] > 0.3) < 4:
            return self._yolo_fallback(image, bbox)
        return Pose2DResult(keypoints=kp, skeleton_edges=_SKEL)

    def _yolo_fallback(self, image, bbox):
        try:
            with _Models._infer_lock:
                res = _Models.yolo_pose()(image[:, :, ::-1], verbose=False)[0]
            if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
                return Pose2DResult(keypoints={}, skeleton_edges=_SKEL)
            kxy = res.keypoints.xy.cpu().numpy()
            kcf = res.keypoints.conf
            kcf = kcf.cpu().numpy() if kcf is not None else np.ones(kxy.shape[:2])
            boxes = res.boxes.xyxy.cpu().numpy()
            i = 0 if bbox is None else int(np.argmax([_iou(b, bbox) for b in boxes]))
            kp = {COCO_KP[j]: (float(kxy[i, j, 0]), float(kxy[i, j, 1]), float(kcf[i, j]))
                  for j in range(min(17, kxy.shape[1]))}
            return Pose2DResult(keypoints=kp, skeleton_edges=_SKEL)
        except Exception:
            return Pose2DResult(keypoints={}, skeleton_edges=_SKEL)


_SKEL = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
         ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
         ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"),
         ("right_eye", "right_ear"), ("left_ankle", "left_heel"),
         ("left_heel", "left_foot_index"), ("right_ankle", "right_heel"),
         ("right_heel", "right_foot_index")]


def _iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    iw = max(0, x1 - x0); ih = max(0, y1 - y0)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


# ------------------------------------------------------------------ face -------
# MediaPipe FaceMesh landmark indices for the named anchors we need.
_FM = {
    "right_eye_outer": 33, "right_eye_inner": 133,
    "left_eye_inner": 362, "left_eye_outer": 263,
    "right_eye": 159, "left_eye": 386,
    "nose_tip": 1, "nose_bridge": 168,
    "mouth_left": 61, "mouth_right": 291, "upper_lip": 13, "lower_lip": 14,
    "chin": 152, "left_face_contour": 234, "right_face_contour": 454,
    "forehead_center": 10,
}


class RealFace(base.FaceBackend):
    name = "mediapipe-facemesh"
    version = "1"
    _cache: dict = {}

    def _detect(self, image, roi):
        """Run FaceLandmarker on a ROI of `image`, upscaling small crops. Returns
        (lms, (x0,y0), scale) in full-image coords, or None."""
        import cv2
        import mediapipe as mp
        H, W = image.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in roi]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        # upscale so the longest side is >= 384 (FaceLandmarker likes larger faces)
        long_side = max(crop.shape[:2])
        scale = max(1.0, 384.0 / long_side)
        if scale > 1.0:
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                              interpolation=cv2.INTER_LINEAR)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        with _Models._infer_lock:
            res = _Models.facemesh().detect(mp_img)
        if not res.face_landmarks:
            return None
        return res.face_landmarks[0], (x0, y0), crop.shape[1], crop.shape[0]

    def estimate_face(self, image: np.ndarray, out_dir: Path, bbox=None):
        H, W = image.shape[:2]
        if bbox is not None:
            bx0, by0, bx1, by1 = [int(v) for v in bbox]
        else:
            bx0, by0, bx1, by1 = 0, 0, W, H
        bw, bh = bx1 - bx0, by1 - by0
        # attempt 1: padded person bbox; attempt 2: upper head region (top 45%)
        rois = [
            (bx0 - 0.15 * bw, by0 - 0.15 * bh, bx1 + 0.15 * bw, by1 + 0.15 * bh),
            (bx0 - 0.05 * bw, by0 - 0.05 * bh, bx1 + 0.05 * bw, by0 + 0.45 * bh),
        ]
        det = None
        for roi in rois:
            det = self._detect(image, roi)
            if det is not None:
                break
        if det is None:
            return Face2DResult(face_mask=None, landmarks={}, bbox=None, confidence=0.0)
        lms, (x0, y0), cw, ch = det
        landmarks = {}
        for name, idx in _FM.items():
            if idx < len(lms):
                lm = lms[idx]
                landmarks[name] = (x0 + lm.x * cw, y0 + lm.y * ch, 0.9)
        pts = np.array([[x0 + lm.x * cw, y0 + lm.y * ch] for lm in lms], np.int32)
        face_mask = None
        try:
            import cv2
            from scipy.spatial import ConvexHull
            hull = pts[ConvexHull(pts).vertices]
            canvas = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(canvas, hull, 1)
            face_mask = canvas.astype(bool)
            fx0, fy0 = pts.min(0); fx1, fy1 = pts.max(0)
            fbbox = (float(fx0), float(fy0), float(fx1), float(fy1))
        except Exception:
            fbbox = (float(x0), float(y0), float(x1), float(y1))
        return Face2DResult(face_mask=face_mask, landmarks=landmarks,
                            bbox=fbbox, confidence=0.85)


# ----------------------------------------------------------- association -------
class RealAssoc(base.SubjectAssociationBackend):
    name = "color-hist"
    version = "1"

    def embed(self, image, mask):
        m = np.asarray(mask, bool)
        if m.sum() < 10:
            return None
        px = image[m].astype(np.float32) / 255.0
        hist = []
        for c in range(3):
            h, _ = np.histogram(px[:, c], bins=8, range=(0, 1), density=True)
            hist.append(h)
        v = np.concatenate(hist)
        return v / (np.linalg.norm(v) + 1e-9)


def warmup_models():
    """Pre-load all heavy singletons in the calling (main) thread."""
    _Models.warmup()


def make_backend_set(config):
    from . import BackendSet
    # touch lightweight construction; heavy models load lazily on first use.
    bs = BackendSet(
        gs=RealGS(), segment=RealSeg(), pose=RealPose(), face=RealFace(),
        depth=RealDepth(), association=RealAssoc(),
        versions={"gs": "depth-lift-2", "seg": "yolov8n-seg-2", "pose": "mp-pose-heavy-2",
                  "face": "facemesh-2", "depth": "depth-anything-v2-small-1",
                  "assoc": "color-hist-1"},
    )
    return bs
