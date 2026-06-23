"""SMPL human-body backend (ROMP) producing a real 3D body reposed to canonical A-pose.

This is the corrected geometry path (see docs/PROJECT.md §13, Postmortem): instead of lifting a
non-metric monocular depth billboard, we regress an SMPL body per person with ROMP,
then re-pose it to the A-pose using the SMPL kinematic tree. Output is a real 3D,
metric-scaled, A-pose colored point cloud + mesh — which is the project's actual goal.

Requires the (license-gated) SMPL neutral model at ~/.romp/SMPL_NEUTRAL.pth. See
`smpl_setup_instructions()`. Without it, `available()` is False and the pipeline falls
back to the depth-lift backend.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..types import SplatCloud
from ..geometry.repose import smpl_apose_thetas


def smpl_model_path() -> Path:
    return Path(os.path.expanduser("~")) / ".romp" / "SMPL_NEUTRAL.pth"


def available() -> bool:
    try:
        import romp  # noqa: F401
    except Exception:
        return False
    return smpl_model_path().exists()


def smpl_setup_instructions() -> str:
    return (
        "SMPL backend needs the license-gated SMPL neutral model.\n"
        "  1. Register (free) at https://smpl.is.tue.mpg.de and download SMPL v1.0.0\n"
        "     (file: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl).\n"
        "  2. Put the male/female/neutral .pkl files in a folder, then run:\n"
        "       romp.prepare_smpl -source_dir <that folder>\n"
        "     This writes ~/.romp/SMPL_NEUTRAL.pth (and SMPL_MALE/FEMALE).\n"
        f"  3. Confirm {smpl_model_path()} exists, then rerun with --backend smpl."
    )


class _SMPLRuntime:
    _romp = None
    _smpl = None

    @classmethod
    def romp(cls):
        if cls._romp is None:
            import romp
            settings = romp.romp_settings(["--mode", "image", "-i", "x", "-o", "y"])
            settings.calc_smpl = True
            settings.render_mesh = False
            cls._romp = romp.ROMP(settings)
        return cls._romp

    @classmethod
    def smpl(cls):
        """ROMP's SMPL forward module, used to re-pose with custom thetas."""
        if cls._smpl is None:
            from romp.smpl import SMPL
            cls._smpl = SMPL(str(smpl_model_path()))
        return cls._smpl


def _weak_persp_project(verts, cam, H, W):
    """ROMP weak-perspective projection: cam=(s,tx,ty) in a normalized square."""
    s, tx, ty = float(cam[0]), float(cam[1]), float(cam[2])
    xy = verts[:, :2] * s + np.array([tx, ty])
    px = (xy[:, 0] + 1.0) * 0.5 * W
    py = (xy[:, 1] + 1.0) * 0.5 * H
    return np.stack([px, py], axis=1)


def reconstruct_apose(image: np.ndarray):
    """Return a list of per-person dicts:
        { 'splats': SplatCloud (canonical A-pose), 'joints': {name: xyz},
          'faces': (F,3) int, 'confidence': float }
    Coordinates are canonical: pelvis at origin, +Y up, scaled by body height.
    """
    H, W = image.shape[:2]
    out = []
    res = _SMPLRuntime.romp()(image[:, :, ::-1])
    if not res:
        return out
    betas = np.atleast_2d(np.asarray(res["smpl_betas"]))
    thetas = np.atleast_2d(np.asarray(res["smpl_thetas"]))   # (N,72) detected pose
    verts_det = np.asarray(res["verts"])                     # (N,6890,3) detected
    cams = np.atleast_2d(np.asarray(res["cam"]))
    smpl = _SMPLRuntime.smpl()

    apose_body = smpl_apose_thetas().reshape(-1)             # (72,) with zero global
    for i in range(betas.shape[0]):
        # re-pose: keep shape (betas), set body pose to A-pose, global orient upright
        pose = np.zeros(72, dtype=np.float32)
        pose[3:] = apose_body[3:]                            # body joints -> A-pose
        verts_a, joints_a = _smpl_forward(smpl, betas[i], pose)

        # color A-pose verts by sampling the image at the DETECTED projection
        col = _sample_colors(image, _weak_persp_project(verts_det[i], cams[i], H, W))

        # canonicalize: pelvis (joint 0) to origin, normalize by height, +Y up
        pelvis = joints_a[0]
        centers = verts_a - pelvis
        height = float(np.percentile(centers[:, 1], 97) - np.percentile(centers[:, 1], 3)) or 1.0
        centers = centers / height
        # SMPL +Y is up but -Y in its own axis convention is down; flip so head is +Y
        if centers[:, 1].mean() < 0:
            centers[:, 1] *= -1
        n = centers.shape[0]
        splats = SplatCloud(centers=centers, scales=np.full((n, 3), 0.01),
                            rotations=np.tile([1.0, 0, 0, 0], (n, 1)),
                            opacities=np.full(n, 0.95), colors=col,
                            extras={"region": np.zeros(n, int),
                                    "confidence": np.full(n, 0.9)})
        jn = _joint_names(joints_a - pelvis, height)
        faces = np.asarray(smpl.faces) if hasattr(smpl, "faces") else None
        out.append({"splats": splats, "joints": jn, "faces": faces, "confidence": 0.9})
    return out


def _smpl_forward(smpl, beta, pose):
    """Call ROMP's SMPL forward; return (verts (6890,3), joints (J,3)) as numpy."""
    import torch
    beta_t = torch.tensor(np.atleast_2d(beta).astype(np.float32))
    pose_t = torch.tensor(np.atleast_2d(pose).astype(np.float32))
    with torch.no_grad():
        res = smpl(betas=beta_t, poses=pose_t)
    if isinstance(res, dict):
        verts = res.get("verts"); joints = res.get("joints", res.get("j3d"))
    else:
        verts, joints = res[0], res[1]
    verts = np.asarray(verts).reshape(-1, 3)
    joints = np.asarray(joints).reshape(-1, 3)
    return verts, joints


def _sample_colors(image, px):
    H, W = image.shape[:2]
    u = np.clip(np.round(px[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(px[:, 1]).astype(int), 0, H - 1)
    return image[v, u].astype(np.float64) / 255.0


_SMPL_JOINT_NAMES = {0: "pelvis", 1: "left_hip", 2: "right_hip", 4: "left_knee",
                     5: "right_knee", 7: "left_ankle", 8: "right_ankle",
                     12: "neck", 15: "head", 16: "left_shoulder", 17: "right_shoulder",
                     18: "left_elbow", 19: "right_elbow", 20: "left_wrist", 21: "right_wrist"}


def _joint_names(joints, height):
    out = {}
    for idx, name in _SMPL_JOINT_NAMES.items():
        if idx < joints.shape[0]:
            p = joints[idx] / height
            if p[1] < 0 and joints[:, 1].mean() < 0:
                p = p.copy(); p[1] *= -1
            out[name] = [float(p[0]), float(p[1]), float(p[2])]
    return out
