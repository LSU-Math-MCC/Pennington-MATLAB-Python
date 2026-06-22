"""Project a source photo directly onto the SMPL-X face carrier.

This is the face-texture path for cases where we want to keep the CameraHMR/LHM
mesh shape and avoid replacing/cutting the head. It builds a tiny faceplate mesh
from the same SMPL-X A-pose vertices and gives those vertices UVs in the source
photo, derived from the fitted posed SMPL-X camera.

Run in the lhm env:
  python tools/face/project_smplx_faceplate.py datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png \
    runs/singleton_texture_ssp3d_bodybuilder_facehybrid_2048/apose_textured_uv.glb \
    runs/singleton_texture_ssp3d_bodybuilder_facehybrid_2048/photo_faceplate
"""

from __future__ import annotations
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


import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import lhm_anthropometry as A  # noqa: E402
import texture_uv_bake as TB  # noqa: E402


def _faces_from_mask(faces: np.ndarray, mask: np.ndarray, min_count: int = 2):
    keep = mask[faces].sum(1) >= min_count
    return faces[keep]


def _seam_split(vertices, faces, uv):
    out_v = []
    out_uv = []
    out_f = []
    index = {}
    for tri in faces:
        ft = []
        for vi in tri:
            key = int(vi)
            if key not in index:
                index[key] = len(out_v)
                out_v.append(vertices[vi])
                out_uv.append(uv[vi])
            ft.append(index[key])
        out_f.append(ft)
    return np.asarray(out_v), np.asarray(out_f, np.int64), np.asarray(out_uv)


def main():
    import trimesh
    import pyrender
    import imageio

    os.environ["PYOPENGL_PLATFORM"] = "egl"

    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: project_smplx_faceplate.py <image> <body_glb> <out_prefix> [betas.npy] "
            "[--warp-mode none|similarity|rbf]"
        )
    image_path = Path(sys.argv[1])
    body_glb = Path(sys.argv[2])
    outp = Path(sys.argv[3])
    betas_path = Path(sys.argv[4]) if len(sys.argv) > 4 else body_glb.parent / "smplx_betas.npy"
    warp_mode = "similarity"
    if "--no-landmark-warp" in sys.argv[4:]:
        warp_mode = "none"
    if "--warp-mode" in sys.argv[4:]:
        i = sys.argv.index("--warp-mode")
        warp_mode = sys.argv[i + 1]
    if warp_mode not in {"none", "similarity", "rbf"}:
        raise ValueError(f"bad warp mode: {warp_mode}")
    outp.parent.mkdir(parents=True, exist_ok=True)

    betas = np.load(betas_path)[:10] if betas_path.exists() else np.zeros(10, np.float32)
    gender, _ = A.estimate_gender([str(image_path)])
    av, _aj, af, _named = A.smplx_apose(betas, gender=gender, arm_deg=45.0)
    av = np.asarray(av)
    af = np.asarray(af, np.int64)

    est = A._estimator()
    img, h, j2d = TB.posed_view(est, str(image_path))
    if h is None:
        raise RuntimeError(f"no person found: {image_path}")
    fx, fy, cx, cy = TB.fit_pinhole(h["j3d"], j2d)
    v3d = h["v3d"]
    z = np.clip(v3d[:, 2], 1e-5, None)
    xy = np.stack([fx * v3d[:, 0] / z + cx, fy * v3d[:, 1] / z + cy], 1)
    H, W = img.shape[:2]
    tex_img = img
    if warp_mode == "rbf":
        warped = TB.face_align_warp(img, j2d)
        if warped is not None:
            tex_img = warped
            imageio.imwrite(str(outp) + "_warp_rbf.png", tex_img)
            print("landmark warp: rbf")
        else:
            print("landmark warp: unavailable; using raw projection")
    elif warp_mode == "similarity":
        warped = similarity_face_warp(img, j2d)
        if warped is not None:
            tex_img = warped
            imageio.imwrite(str(outp) + "_warp_similarity.png", tex_img)
            print("landmark warp: similarity")
        else:
            print("landmark warp: unavailable; using raw projection")
    else:
        print("landmark warp: none")
    uv = np.zeros((len(av), 2), np.float32)
    uv[:, 0] = np.clip(xy[:, 0] / max(W - 1, 1), 0, 1)
    uv[:, 1] = np.clip(1.0 - xy[:, 1] / max(H - 1, 1), 0, 1)

    face_mask = TB._face_core_vertex_mask(betas)
    head_mask = TB._head_vertex_mask(betas)
    face_faces = _faces_from_mask(af, face_mask, min_count=2)
    # Keep the frontal/central head band too, so cheek/forehead seams are less abrupt.
    centers = av[face_faces].mean(1)
    z_cut = np.percentile(av[head_mask, 2], 52)
    face_faces = face_faces[centers[:, 2] >= z_cut]
    vv, ff, vu = _seam_split(av, face_faces, uv)
    mesh = trimesh.Trimesh(vv, ff, process=False)
    mesh.vertices = np.asarray(mesh.vertices) + np.asarray(mesh.vertex_normals) * 0.0025
    tex = Image.fromarray(tex_img)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex, roughnessFactor=0.62, metallicFactor=0.0)
    mesh.visual = trimesh.visual.TextureVisuals(uv=vu, material=mat)
    mesh.export(str(outp) + "_faceplate.glb")

    body = trimesh.load(body_glb, process=False, force="mesh")
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="textured_body", geom_name="textured_body")
    scene.add_geometry(mesh, node_name="photo_projected_faceplate", geom_name="photo_projected_faceplate")
    scene.export(str(outp) + "_body_faceplate.glb")

    def render(closeup):
        sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1, 1, 1])
        sc.add(pyrender.Mesh.from_trimesh(body, smooth=True))
        sc.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        fv = np.asarray(mesh.vertices)
        if closeup:
            c = fv.mean(0); e = max(fv[:, 0].ptp(), fv[:, 1].ptp())
            zc = fv[:, 2].max() + e * 1.2; Wout, Hout = 360, 400
        else:
            bv = np.asarray(body.vertices)
            c = bv.mean(0); e = bv.ptp(0).max(); zc = c[2] + e * 1.45; Wout, Hout = 360, 620
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.2)
        p = np.eye(4); p[0, 3] = c[0]; p[1, 3] = c[1]; p[2, 3] = zc; sc.add(cam, pose=p)
        if not closeup:
            sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.0), pose=p)
        r = pyrender.OffscreenRenderer(Wout, Hout)
        col, _ = r.render(sc, flags=pyrender.RenderFlags.FLAT if closeup else 0); r.delete()
        return col

    imageio.imwrite(str(outp) + "_face.png", render(True))
    imageio.imwrite(str(outp) + "_body.png", render(False))
    print(f"FACEPLATE_OK faces={len(ff)} -> {outp}_body_faceplate.glb")


def similarity_face_warp(img, j2d):
    """Conservative face placement warp.

    Full TPS/RBF warping can make eyes/nose too thin because every landmark is
    exactly forced to the SMPL-X projection. This estimates one global similarity
    transform from robust feature groups, preserving feature proportions.
    """
    import cv2
    lm = TB._fan_landmarks(img)
    if lm is None or lm.shape[0] < 68 or j2d is None or j2d.shape[0] < 127:
        return None

    def mean(ids):
        p = lm[list(ids)]
        return p[np.isfinite(p).all(1)].mean(0)

    def sx_mean(dlib_ids):
        idx = [76 + (k - 17) for k in dlib_ids]
        p = j2d[idx]
        return p[np.isfinite(p).all(1)].mean(0)

    src = np.stack([
        mean(range(36, 42)),       # right eye
        mean(range(42, 48)),       # left eye
        mean(range(31, 36)),       # nose base
        mean(range(48, 60)),       # mouth outer
        mean(range(17, 22)),       # right brow
        mean(range(22, 27)),       # left brow
    ]).astype(np.float32)
    dst = np.stack([
        sx_mean(range(36, 42)),
        sx_mean(range(42, 48)),
        sx_mean(range(31, 36)),
        sx_mean(range(48, 60)),
        sx_mean(range(17, 22)),
        sx_mean(range(22, 27)),
    ]).astype(np.float32)
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    if ok.sum() < 3:
        return None
    M, _inliers = cv2.estimateAffinePartial2D(src[ok], dst[ok], method=cv2.LMEDS)
    if M is None:
        return None
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


if __name__ == "__main__":
    main()
