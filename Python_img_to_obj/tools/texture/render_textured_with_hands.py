"""Render the textured (UV-baked) SOTA body WITH HaMeR finger-level hands grafted at the wrists,
preserving the body's UV texture. Body + hands are separate pyrender meshes in one scene (NOT
concatenated -- concatenation collapses TextureVisuals). The baked body already carries the
coherent face from the frontal view; the hands add finger geometry.

Run (lhm env): python tools/texture/render_textured_with_hands.py <subject>
"""
import sys, os
import numpy as np
for a, t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
             ("object", object), ("str", str), ("nan", float("nan")), ("inf", float("inf"))]:
    if not hasattr(np, a):
        setattr(np, a, t)
import trimesh, pyrender, imageio
os.environ["PYOPENGL_PLATFORM"] = "egl"
from scipy.spatial import cKDTree
from trimesh.registration import icp

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
S = sys.argv[1] if len(sys.argv) > 1 else "s1"
body_glb = f"{REPO}/runs/uv_{S}_1v/apose_textured_uv.glb"
hamer_dir = f"{REPO}/runs/hamer_out" if S == "s1" else f"{REPO}/runs/hamer_{S}"

body = trimesh.load(body_glb, process=False, force="mesh")
bv = np.asarray(body.vertices)
xmax = np.abs(bv[:, 0]).max()

hand_meshes = []
for tag, sgn in [("L", -1), ("R", 1)]:
    hp = f"{hamer_dir}/hand_{tag}_flat.obj"
    if not os.path.exists(hp):
        continue
    sel = (np.sign(bv[:, 0]) == sgn) & (np.abs(bv[:, 0]) > 0.72 * xmax) & (bv[:, 1] < np.percentile(bv[:, 1], 45))
    tgt = bv[sel]
    if len(tgt) < 30:
        continue
    hand = trimesh.load(hp, process=False)
    hv = np.asarray(hand.vertices)

    # CONSTRAINED placement (no ICP reflection/curl): align the HaMeR hand's PCA frame to the
    # body native-hand region's PCA frame, force the long axis to point AWAY from the body
    # (fingers outward), and keep a proper rotation (det=+1).
    def pca_frame(P):
        c = P.mean(0); X = P - c
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return c, Vt
    bc = bv.mean(0)
    nc, nA = pca_frame(tgt)        # native hand frame
    hc, hA = pca_frame(hv)         # HaMeR hand frame
    # orient each long axis (row0) to point from body center outward to the hand
    if np.dot(nA[0], nc - bc) < 0: nA[0] = -nA[0]
    if np.dot(hA[0], hc - bc) < 0: hA[0] = -hA[0]
    # keep frames right-handed
    nA[2] = np.cross(nA[0], nA[1]); hA[2] = np.cross(hA[0], hA[1])
    R = nA.T @ hA
    if np.linalg.det(R) < 0:
        hA[1] = -hA[1]; hA[2] = np.cross(hA[0], hA[1]); R = nA.T @ hA
    scale = tgt.ptp(0).max() / (hv.ptp(0).max() + 1e-9)
    new = (scale * (R @ (hv - hc).T).T) + nc
    h2 = trimesh.Trimesh(new, hand.faces, process=False)
    h2.visual = trimesh.visual.ColorVisuals(h2, vertex_colors=[225, 170, 150, 255])
    hand_meshes.append(h2)

sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[.7, .7, .7])
sc.add(pyrender.Mesh.from_trimesh(body, smooth=True))        # textured body (UV preserved; baked face)
for hm in hand_meshes:
    sc.add(pyrender.Mesh.from_trimesh(hm, smooth=True))
c = bv.mean(0); e = bv.ptp(0).max()
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.2)
p = np.eye(4); p[0, 3] = c[0]; p[1, 3] = c[1]; p[2, 3] = c[2] + e * 1.45
sc.add(cam, pose=p)
sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.5), pose=p)
r = pyrender.OffscreenRenderer(380, 640); col, _ = r.render(sc); r.delete()
out = f"{REPO}/runs/uv_{S}_final.png"
imageio.imwrite(out, col)
print("rendered", out, "| hands:", len(hand_meshes))
