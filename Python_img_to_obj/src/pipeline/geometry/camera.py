"""Camera model: intrinsics, world<->camera transforms, project/backproject.

Convention: a world point X maps to camera coords by  x_cam = R @ X + t.
Pixel coords use the pinhole model  [u, v, 1]^T ~ K @ x_cam (x_cam.z > 0 in front).
"""
from __future__ import annotations

import numpy as np

from ..types import Camera


def default_camera(width: int, height: int, fov_deg: float = 55.0) -> Camera:
    """An identity-extrinsic camera with a sensible focal length for a given image."""
    f = 0.5 * max(width, height) / np.tan(np.deg2rad(fov_deg) * 0.5)
    K = np.array([[f, 0, width / 2.0],
                  [0, f, height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    return Camera(K=K, R=np.eye(3), t=np.zeros(3), width=width, height=height)


def world_to_cam(cam: Camera, X_world: np.ndarray) -> np.ndarray:
    """X_world: (...,3) -> camera coords (...,3)."""
    X = np.asarray(X_world, dtype=np.float64)
    return X @ cam.R.T + cam.t


def cam_to_world(cam: Camera, X_cam: np.ndarray) -> np.ndarray:
    X = np.asarray(X_cam, dtype=np.float64)
    return (X - cam.t) @ cam.R


def project(cam: Camera, X_world: np.ndarray):
    """Project world points to pixels.

    Returns (uv: Nx2 float, valid: N bool). valid is True when the point is in front
    of the camera and lands within image bounds.
    """
    X_world = np.atleast_2d(np.asarray(X_world, dtype=np.float64))
    x_cam = world_to_cam(cam, X_world)
    z = x_cam[:, 2]
    in_front = z > 1e-9
    z_safe = np.where(in_front, z, 1.0)
    proj = (cam.K @ (x_cam / z_safe[:, None]).T).T
    uv = proj[:, :2]
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] <= cam.width - 1) &
        (uv[:, 1] >= 0) & (uv[:, 1] <= cam.height - 1)
    )
    valid = in_front & in_bounds
    return uv, valid


def backproject(cam: Camera, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Backproject pixels + per-pixel depth (camera-space z) to camera-space 3D points.

    uv: Nx2, depth: N -> Nx3 camera-space points.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    ones = np.ones((uv.shape[0], 1))
    pix_h = np.concatenate([uv, ones], axis=1)         # Nx3
    Kinv = np.linalg.inv(cam.K)
    rays = pix_h @ Kinv.T                               # Nx3, z-component == 1
    return rays * depth[:, None]


def backproject_to_world(cam: Camera, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return cam_to_world(cam, backproject(cam, uv, depth))
