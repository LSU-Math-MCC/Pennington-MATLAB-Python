"""Graft flat HaMeR A-pose hands onto any A-pose body glb. Proves hand integration
generalizes across subjects. Usage: python tools/hands/fuse_hands_generic.py <body_glb> <hamer_dir> <out_prefix>
"""
import os, sys, numpy as np, trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
from trimesh.registration import icp

BODY = os.path.abspath(sys.argv[1])
HO = os.path.abspath(sys.argv[2])
OUTP = os.path.abspath(sys.argv[3])

body = trimesh.load(BODY, process=False, force="mesh")
bv = np.asarray(body.vertices)
xmax = np.abs(bv[:, 0]).max()
side = {}
for tag, sgn in [("L", -1), ("R", 1)]:
    m = (np.sign(bv[:, 0]) == sgn) & (np.abs(bv[:, 0]) > 0.72 * xmax) & (bv[:, 1] < np.percentile(bv[:, 1], 45))
    side[tag] = bv[m]

meshes = [body]
for tag in ("L", "R"):
    hp = f"{HO}/hand_{tag}_flat.obj"
    if not os.path.exists(hp) or len(side[tag]) < 30:
        print("skip", tag); continue
    hand = trimesh.load(hp, process=False)
    tgt = side[tag]
    hv = np.asarray(hand.vertices)
    hv = (hv - hv.mean(0)) * (tgt.ptp(0).max() / (hv.ptp(0).max() + 1e-9)) + tgt.mean(0)
    h2 = trimesh.Trimesh(hv, hand.faces, process=False)
    try:
        T, _, _ = icp(h2.vertices, tgt, max_iterations=40, scale=True); h2.apply_transform(T)
    except Exception as e:
        print("icp", tag, e)
    h2.visual = trimesh.visual.ColorVisuals(h2, vertex_colors=[225, 170, 150, 255])
    meshes.append(h2)
    print(f"placed {tag} at {tgt.mean(0).round(3)}")

scene = trimesh.util.concatenate(meshes)
scene.export(OUTP + "_body_hands.glb")

import pyrender, imageio
v = scene.vertices; c = v.mean(0); ext = v.ptp(0).max()
sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[.35, .35, .35])
sc.add(pyrender.Mesh.from_trimesh(scene, smooth=True))
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.5)
p = np.eye(4); p[0, 3] = c[0]; p[1, 3] = c[1]; p[2, 3] = c[2] + ext * 1.45
sc.add(cam, pose=p)
for dx, dy, it in [(0.6, 0.3, 4.0), (-0.7, 0.2, 2.5)]:
    lp = np.eye(4); lp[0, 3] = c[0] + dx * ext; lp[1, 3] = c[1] + dy * ext; lp[2, 3] = c[2] + ext * 1.4
    sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=it), pose=lp)
r = pyrender.OffscreenRenderer(420, 700); col, _ = r.render(sc); r.delete()
imageio.imwrite(OUTP + "_front.png", col)
print("rendered", OUTP + "_front.png")
