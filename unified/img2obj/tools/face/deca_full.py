"""DECA full: fit FLAME to the photo, then BACK-PROJECT the photo onto the fitted FLAME verts
(photo -> per-vertex colour via the fitted camera). Rendering the result recreates the photo.
Saves FLAME verts + per-vertex colours + a frontal render for vision QC.

Run (camerahmr env): python tools/face/deca_full.py <image> <out_prefix> [cuda|cpu]
"""
import sys, os
import numpy as np
import torch
import cv2
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/DECA"))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util

img_path, outp = sys.argv[1], sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"DECA_START image={img_path} out={outp} device={device}", flush=True)
deca_cfg.model.use_tex = False
deca_cfg.model.flame_model_path = os.path.expanduser("~/DECA/data/generic_model.pkl")
deca_cfg.rasterizer_type = "pytorch3d"
deca = DECA(config=deca_cfg, device=device)
print("DECA_MODEL_READY", flush=True)

td = datasets.TestData(img_path, iscrop=True, face_detector="fan")
data = td[0]
images = data["image"].to(device)[None]
print("DECA_INPUT_READY", flush=True)
with torch.no_grad():
    codedict = deca.encode(images)
    opdict, visdict = deca.decode(codedict)
print("DECA_DECODE_READY", flush=True)

tv = opdict["trans_verts"][0].cpu().numpy()               # image-aligned (x,y in [-1,1], z depth)
# NEUTRAL head pose (zero global rotation, keep jaw/expression) so the face is frontal & upright
# -> clean portrait + correct shape to weld onto the A-pose SMPL-X head.
with torch.no_grad():
    pose0 = codedict["pose"].clone(); pose0[:, :3] = 0.0
    verts = deca.flame(shape_params=codedict["shape"], expression_params=codedict["exp"],
                       pose_params=pose0)[0][0].cpu().numpy()
# input crop (denormalised, 224) for sampling
inp = visdict["inputs"][0].permute(1, 2, 0).cpu().numpy()
inp = np.clip(inp, 0, 1)
H = W = inp.shape[0]
px = np.clip(((tv[:, 0] + 1) * 0.5 * W).astype(int), 0, W - 1)
py = np.clip(((tv[:, 1] + 1) * 0.5 * H).astype(int), 0, H - 1)   # DECA trans_verts y already image-down
vcols = (inp[py, px] * 255).astype(np.uint8)              # photo colour per FLAME vertex

def build_uv_mesh(verts, faces, uvfaces, uvcoords):
    """Split FLAME vertices at UV seams so a real texture image can be used."""
    out_v = []
    out_uv = []
    out_f = []
    out_orig = []
    index = {}
    for f, tf in zip(faces, uvfaces):
        tri = []
        for vi, ti in zip(f, tf):
            key = (int(vi), int(ti))
            if key not in index:
                index[key] = len(out_v)
                out_v.append(verts[vi])
                out_uv.append(uvcoords[ti])
                out_orig.append(int(vi))
            tri.append(index[key])
        out_f.append(tri)
    return (np.asarray(out_v, np.float32), np.asarray(out_f, np.int64),
            np.asarray(out_uv, np.float32), np.asarray(out_orig, np.int64))

np.save(outp + "_verts.npy", verts)
np.save(outp + "_vcols.npy", vcols)
cv2.imwrite(outp + "_input.png", (inp[:, :, ::-1] * 255).astype(np.uint8))

# frontal render of the photo-coloured FLAME mesh (should look like the input)
import trimesh, pyrender, imageio
os.environ["PYOPENGL_PLATFORM"] = "egl"
import decalib.utils.config as _c
faces = deca.flame.faces_tensor.cpu().numpy()
np.save(outp + "_faces.npy", faces)
uv_texture = opdict["uv_texture_gt"][0].permute(1, 2, 0).detach().cpu().numpy()
uv_texture = np.clip(uv_texture * 255.0, 0, 255).astype(np.uint8)
imageio.imwrite(outp + "_texture.png", uv_texture)
uvcoords = deca.render.raw_uvcoords[0].detach().cpu().numpy()
uvfaces = deca.render.uvfaces[0].detach().cpu().numpy()
np.save(outp + "_uvcoords.npy", uvcoords)
np.save(outp + "_uvfaces.npy", uvfaces)
uv_v, uv_f, uv_uv, uv_orig = build_uv_mesh(verts, faces, uvfaces, uvcoords)
np.save(outp + "_uvmesh_faces.npy", uv_f)
np.save(outp + "_uvmesh_uv.npy", uv_uv)
np.save(outp + "_uvmesh_vertex_ids.npy", uv_orig)
uv_visual = trimesh.visual.TextureVisuals(
    uv=uv_uv,
    material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(uv_texture),
        roughnessFactor=0.65,
        metallicFactor=0.0,
    ),
)
trimesh.Trimesh(uv_v, uv_f, visual=uv_visual, process=False).export(outp + "_mesh_uv.glb")
vr = verts.copy()                                         # neutral FLAME-forward verts are y-up already
m = trimesh.Trimesh(vr, faces, vertex_colors=vcols, process=False)
m.export(outp + "_mesh.glb")
# zoom to the FACE (front, upper-mid of the head), not the whole head extent
fc = vr[(vr[:, 2] > np.percentile(vr[:, 2], 55))]         # forward-facing (face) verts
c = fc.mean(0); e = max(fc[:, 0].ptp(), fc[:, 1].ptp())
sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1, 1, 1])
sc.add(pyrender.Mesh.from_trimesh(m, smooth=True))
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3)
p = np.eye(4); p[0, 3] = c[0]; p[1, 3] = c[1]; p[2, 3] = vr[:, 2].max() + e * 1.1
sc.add(cam, pose=p)
r = pyrender.OffscreenRenderer(400, 440); col, _ = r.render(sc, flags=pyrender.RenderFlags.FLAT); r.delete()
imageio.imwrite(outp + "_render.png", col)
print("DECA done:", outp, "verts", verts.shape)
