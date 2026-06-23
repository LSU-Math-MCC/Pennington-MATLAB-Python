"""Place a DECA FLAME face (recreated from the photo) onto the textured SMPL-X body using the
EXACT SMPL-X__FLAME_vertex_ids correspondence: Procrustes-fit the DECA FLAME verts to the body's
A-pose head verts (the 5023 FLAME-corresponding SMPL-X verts) -> the face lands exactly on the
head (fully determined, no orientation guessing). Render body (UV texture) + face (photo vcols).

Run (lhm env): python tools/face/integrate_deca_face.py <subject> [body_glb] [out_prefix]
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

import sys, os
import numpy as np
for a, t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
             ("object", object), ("str", str), ("nan", float("nan")), ("inf", float("inf"))]:
    if not hasattr(np, a):
        setattr(np, a, t)
import trimesh, pyrender, imageio
os.environ["PYOPENGL_PLATFORM"] = "egl"

REPO = _repo
sys.path.insert(0, REPO + "/tools")
S = sys.argv[1] if len(sys.argv) > 1 else "ssp3d_bodybuilder"
BODY_GLB = sys.argv[2] if len(sys.argv) > 2 else f"{REPO}/runs/uv_{S}_1v/apose_textured_uv.glb"
OUTP = sys.argv[3] if len(sys.argv) > 3 else f"{REPO}/runs/final_{S}"
DECA_PREFIX = sys.argv[4] if len(sys.argv) > 4 else f"{REPO}/runs/decafull_{S}"
GENDER_IMAGE = sys.argv[5] if len(sys.argv) > 5 else f"{REPO}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png"
MODE = sys.argv[6] if len(sys.argv) > 6 else "smplx_faceplate"
if not os.path.isabs(BODY_GLB):
    BODY_GLB = os.path.join(REPO, BODY_GLB)
if not os.path.isabs(OUTP):
    OUTP = os.path.join(REPO, OUTP)
if not os.path.isabs(DECA_PREFIX):
    DECA_PREFIX = os.path.join(REPO, DECA_PREFIX)
if not os.path.isabs(GENDER_IMAGE):
    GENDER_IMAGE = os.path.join(REPO, GENDER_IMAGE)
os.makedirs(os.path.dirname(OUTP), exist_ok=True)
FID = os.path.expanduser("~/LHM/pretrained_models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy")

import lhm_anthropometry as A
body_dir = os.path.dirname(BODY_GLB)
betas_candidates = [
    os.path.join(body_dir, "smplx_betas.npy"),
    f"{REPO}/runs/camerahmr_{S}_smplx_betas.npy",
]
betas_path = next((p for p in betas_candidates if os.path.exists(p)), None)
if betas_path:
    betas = np.load(betas_path)[:10]
    print(f"{S}: using SMPL-X betas from {betas_path}")
else:
    betas = np.zeros(10, dtype=np.float32)
    print(f"{S}: missing fitted SMPL-X betas; using neutral SMPL-X betas for DECA placement")
gender_images = [GENDER_IMAGE] if os.path.exists(GENDER_IMAGE) else []
gender, _ = A.estimate_gender(gender_images)
av, aj, af, named = A.smplx_apose(betas, gender=gender, arm_deg=45.0)
av = np.asarray(av)
ids = np.load(FID).astype(int)
sx_head = av[ids]                                          # SMPL-X head verts (FLAME order)

fverts = np.load(f"{DECA_PREFIX}_verts.npy")              # DECA FLAME (neutral, y-up)
ffaces = np.load(f"{DECA_PREFIX}_faces.npy")
fvcols = np.load(f"{DECA_PREFIX}_vcols.npy")
uvmesh_faces_path = f"{DECA_PREFIX}_uvmesh_faces.npy"
uvmesh_uv_path = f"{DECA_PREFIX}_uvmesh_uv.npy"
uvmesh_ids_path = f"{DECA_PREFIX}_uvmesh_vertex_ids.npy"
texture_path = f"{DECA_PREFIX}_texture.png"


def procrustes(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S0, D0 = src - ms, dst - md
    U, _, Vt = np.linalg.svd(S0.T @ D0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    sc = np.linalg.norm(D0) / (np.linalg.norm(S0) + 1e-9)
    return lambda P: (sc * (R @ (P - ms).T).T) + md


if MODE == "smplx_faceplate":
    # Keep the CameraHMR/SMPL-X head shape. DECA contributes texture only.
    # SMPL-X__FLAME_vertex_ids is ordered in FLAME vertex order, so sx_head[i]
    # is the SMPL-X carrier vertex corresponding to FLAME/DECA vertex i.
    face_on_head = sx_head.copy()
else:
    face_on_head = procrustes(fverts, sx_head)(fverts)     # DECA face shape -> body head frame
neck_y = float(aj[12, 1])
deca_cut_y = max(neck_y + 0.065, float(np.percentile(face_on_head[:, 1], 5)))


def _sample_uv_mask(uv, mask_path):
    from PIL import Image
    if not os.path.exists(mask_path):
        return np.ones(len(uv), bool)
    mask = np.asarray(Image.open(mask_path).convert("L"))
    h, w = mask.shape[:2]
    x = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
    y = np.clip(((1.0 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
    return mask[y, x] > 127


def deca_overlay_mesh(verts, faces, colors, cut_y):
    """Build the DECA FLAME overlay.

    Keep the face/head, but drop the low FLAME neck flap. The body already owns the neck,
    and leaving both surfaces there creates the torn/doubled throat seen in closeups.
    """
    centers = verts[faces].mean(1)
    keep = centers[:, 1] >= cut_y
    mesh = trimesh.Trimesh(verts.copy(), faces[keep].copy(), vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def deca_uv_overlay_mesh():
    """Build a textured FLAME UV faceplate/full-head mesh if DECA UV files exist."""
    from PIL import Image
    if not (os.path.exists(uvmesh_faces_path) and os.path.exists(uvmesh_uv_path) and
            os.path.exists(uvmesh_ids_path) and os.path.exists(texture_path)):
        return None
    uf = np.load(uvmesh_faces_path)
    uuv = np.load(uvmesh_uv_path).astype(np.float32)
    uids = np.load(uvmesh_ids_path).astype(int)
    uv_verts = face_on_head[uids]
    centers = uv_verts[uf].mean(1)
    keep = centers[:, 1] >= deca_cut_y
    if MODE in ("faceplate", "smplx_faceplate"):
        uv_centers = uuv[uf].mean(1)
        keep &= _sample_uv_mask(uv_centers, os.path.expanduser("~/DECA/data/uv_face_eye_mask.png"))
    tex = Image.open(texture_path).convert("RGB")
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex, roughnessFactor=0.62, metallicFactor=0.0)
    visual = trimesh.visual.TextureVisuals(uv=uuv, material=mat)
    mesh = trimesh.Trimesh(uv_verts.copy(), uf[keep].copy(), visual=visual, process=False)
    mesh.remove_unreferenced_vertices()
    if MODE == "smplx_faceplate":
        mesh.vertices = np.asarray(mesh.vertices) + np.asarray(mesh.vertex_normals) * 0.0025
    return mesh


face = deca_uv_overlay_mesh()
if face is None:
    face = deca_overlay_mesh(face_on_head, ffaces, fvcols, deca_cut_y)
face.export(f"{OUTP}_deca_face.glb")

body = trimesh.load(BODY_GLB, process=False, force="mesh")
bv = np.asarray(body.vertices)
faces = np.asarray(body.faces)

if MODE == "smplx_faceplate":
    # Overlay the DECA-derived face texture directly on the SMPL-X/CameraHMR carrier.
    # No head slicing: this is a face-texture layer for the existing mesh shape.
    print(f"{S}: SMPL-X faceplate mode; preserving body/head mesh and overlaying DECA face texture")
    def render_smplx_faceplate(closeup):
        sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1, 1, 1])
        sc.add(pyrender.Mesh.from_trimesh(body, smooth=True))
        sc.add(pyrender.Mesh.from_trimesh(face, smooth=True))
        fv = np.asarray(face.vertices)
        if closeup:
            c = fv.mean(0); e = max(fv[:, 0].ptp(), fv[:, 1].ptp())
            z = fv[:, 2].max() + e * 1.2; W, Hh = 360, 400
        else:
            c = bv.mean(0); e = bv.ptp(0).max(); z = c[2] + e * 1.45; W, Hh = 360, 620
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.2)
        p = np.eye(4); p[0, 3] = c[0]; p[1, 3] = c[1]; p[2, 3] = z; sc.add(cam, pose=p)
        if not closeup:
            sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.0), pose=p)
        r = pyrender.OffscreenRenderer(W, Hh)
        col, _ = r.render(sc, flags=pyrender.RenderFlags.FLAT if closeup else 0); r.delete()
        return col
    imageio.imwrite(f"{OUTP}_face.png", render_smplx_faceplate(True))
    imageio.imwrite(f"{OUTP}_body.png", render_smplx_faceplate(False))
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="textured_body", geom_name="textured_body")
    scene.add_geometry(face, node_name="deca_face_texture_overlay", geom_name="deca_face_texture_overlay")
    scene.export(f"{OUTP}_body_deca_face.glb")
    print(f"{S}: DECA face texture overlay -> {OUTP}_body_deca_face.glb")
    sys.exit(0)

# Cut out the old SMPL-X head under the DECA/FLAME overlay. Receding the old face is not
# enough: textured scalp/eyes/mouth/neck fragments can z-fight or bleed through the hybrid
# head render. Keep a low neck/socket band so the FLAME head has body geometry to meet.
fv_face = np.asarray(face.vertices)
flame_min = fv_face.min(0)
flame_max = fv_face.max(0)
face_centers = bv[faces].mean(1)
in_flame_box = (
    (face_centers[:, 0] > flame_min[0] - 0.035) &
    (face_centers[:, 0] < flame_max[0] + 0.035) &
    (face_centers[:, 1] > deca_cut_y - 0.015) &
    (face_centers[:, 1] < flame_max[1] + 0.045) &
    (face_centers[:, 2] > flame_min[2] - 0.050) &
    (face_centers[:, 2] < flame_max[2] + 0.050)
)
high_head = face_centers[:, 1] > deca_cut_y - 0.010
front_cut = 45 if MODE == "smplx_faceplate" else (35 if MODE == "faceplate" else 8)
front_or_side = face_centers[:, 2] > np.percentile(fv_face[:, 2], front_cut)
keep_faces = ~(in_flame_box & high_head & front_or_side)
removed = int((~keep_faces).sum())
body.update_faces(keep_faces)
body.remove_unreferenced_vertices()
bv2 = np.asarray(body.vertices)
print(f"{S}: removed {removed} old SMPL-X head faces before DECA overlay; mode={MODE}; deca_cut_y={deca_cut_y:.4f}")

# render full body + face, and a head closeup
def render(closeup):
    sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1, 1, 1])
    sc.add(pyrender.Mesh.from_trimesh(body, smooth=True))
    sc.add(pyrender.Mesh.from_trimesh(face, smooth=True))
    fv = np.asarray(face.vertices)
    if closeup:
        c = fv.mean(0); e = max(fv[:, 0].ptp(), fv[:, 1].ptp())
        z = fv[:, 2].max() + e * 1.2; W, Hh = 360, 400
    else:
        c = bv2.mean(0); e = bv2.ptp(0).max(); z = c[2] + e * 1.45; W, Hh = 360, 620
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.2)
    p = np.eye(4); p[0, 3] = c[0]; p[1, 3] = c[1]; p[2, 3] = z; sc.add(cam, pose=p)
    if not closeup:
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.0), pose=p)
    r = pyrender.OffscreenRenderer(W, Hh)
    col, _ = r.render(sc, flags=pyrender.RenderFlags.FLAT if closeup else 0); r.delete()
    return col

imageio.imwrite(f"{OUTP}_face.png", render(True))
imageio.imwrite(f"{OUTP}_body.png", render(False))
scene = trimesh.Scene()
scene.add_geometry(body, node_name="textured_body", geom_name="textured_body")
scene.add_geometry(face, node_name="deca_face", geom_name="deca_face")
scene.export(f"{OUTP}_body_deca_face.glb")
print(f"{S}: DECA face integrated (Procrustes on {len(ids)} exact correspondences) -> {OUTP}_body_deca_face.glb")
