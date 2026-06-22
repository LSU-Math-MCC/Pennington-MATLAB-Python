"""Normalized single-image HMR result shared by every backend (CameraHMR, BLADE, ...).

The whole point of the polymorphic layer: each backend, in its OWN conda env, reduces its
native output to this one contract, so a single env-agnostic compositor can render every method
identically and tile them into a comparison figure.

Per image we store, in camera space EXACTLY as fed to a pinhole camera at the origin
(i.e. SMPL/-X vertices with the predicted full-image translation already added; the renderer
applies the 180-deg x-flip itself, matching CameraHMR's renderer_pyrd):
  people  : list of (Vi,3) float32 vertex arrays (one per detected person)
  faces   : (F,3) int32 topology (SMPL=13776, SMPL-X=20908 faces) -- renderer is topology-agnostic
  focal   : float, pixels (fx=fy)
  img_w,h : ints
A backend may ALSO drop a native overlay PNG beside the npz (overlay fallback) if it cannot
cleanly expose geometry; the compositor prefers geometry but will use the PNG if that's all there is.
"""
import os
import re
import numpy as np


def native_path(p):
    """Translate a stored path to the OS we're running on, so a Windows kernel can read an
    image_path written by a WSL runner (and vice-versa). /mnt/c/a <-> C:\\a."""
    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", p)
        return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\") if m else p
    m = re.match(r"^([a-zA-Z]):[\\/](.*)$", p)
    return "/mnt/" + m.group(1).lower() + "/" + m.group(2).replace("\\", "/") if m else p


def save(path, image_path, img_w, img_h, focal, people, faces):
    """people: list of (Vi,3) arrays. Stored as an object array so per-person counts can differ."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = np.empty(len(people), dtype=object)
    for i, v in enumerate(people):
        obj[i] = np.asarray(v, np.float32)
    np.savez(path,
             image_path=str(image_path),
             img_w=int(img_w), img_h=int(img_h), focal=float(focal),
             n_people=len(people), people=obj, faces=np.asarray(faces, np.int32))


def load(path):
    d = np.load(path, allow_pickle=True)
    return dict(image_path=native_path(str(d["image_path"])),
                img_w=int(d["img_w"]), img_h=int(d["img_h"]), focal=float(d["focal"]),
                people=list(d["people"]), faces=d["faces"])
