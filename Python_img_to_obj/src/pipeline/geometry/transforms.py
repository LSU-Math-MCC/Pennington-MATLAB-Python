"""Rigid transform / rotation utilities (quaternions, 4x4 homogeneous, Procrustes)."""
from __future__ import annotations

import numpy as np


def make_T(R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = scale * np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    return pts @ T[:3, :3].T + T[:3, 3]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix. Accepts batches (...,4)."""
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R[0] if single else R


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


def quat_average(quats: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Markley quaternion averaging via the largest eigenvector of sum(w q q^T)."""
    quats = np.atleast_2d(np.asarray(quats, dtype=np.float64))
    if weights is None:
        weights = np.ones(quats.shape[0])
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    # align signs to first quaternion to avoid cancellation
    ref = quats[0]
    signs = np.sign(quats @ ref)
    signs[signs == 0] = 1
    q = quats * signs[:, None]
    M = (q * weights[:, None]).T @ q
    w, v = np.linalg.eigh(M)
    avg = v[:, -1]
    if avg[0] < 0:
        avg = -avg
    return avg / (np.linalg.norm(avg) + 1e-12)


def right_handed_frame(x_axis: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Build an orthonormal right-handed rotation matrix whose columns are X, Y, Z.

    X follows x_axis; Y is y_hint orthogonalized against X; Z = X cross Y.
    Returns R with R[:,0]=X etc. (a canonical-from-world basis when rows are axes).
    """
    x = np.asarray(x_axis, dtype=np.float64)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.asarray(y_hint, dtype=np.float64)
    y = y - np.dot(y, x) * x
    ny = np.linalg.norm(y)
    if ny < 1e-9:
        # pick any orthogonal vector
        tmp = np.array([1.0, 0, 0]) if abs(x[0]) < 0.9 else np.array([0, 1.0, 0])
        y = tmp - np.dot(tmp, x) * x
        ny = np.linalg.norm(y)
    y = y / (ny + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.stack([x, y, z], axis=1)  # columns are axes


def procrustes(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Best rigid (optionally similarity) transform mapping src -> dst.

    Returns (R, t, s) minimizing || s R src + t - dst ||. Kabsch / Umeyama.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d
    H = S.T @ D / src.shape[0]
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    if with_scale:
        var_s = (S ** 2).sum() / src.shape[0]
        s = (sig * np.array([1, 1, d])).sum() / (var_s + 1e-12)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s
