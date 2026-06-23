"""Render quick front and face previews for a textured GLB.

Run (lhm env): python tools/texture/render_texture_preview.py <mesh.glb> <out_dir>
Writes: render_front.png, render_face.png
"""
import os
import sys

import imageio
import numpy as np
import pyrender
import trimesh


os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _meshes_and_bounds(asset):
    if isinstance(asset, trimesh.Scene):
        meshes = [g for g in asset.geometry.values() if hasattr(g, "vertices") and len(g.vertices)]
        bounds = asset.bounds
    else:
        meshes = [asset]
        bounds = asset.bounds
    if not meshes or bounds is None:
        raise ValueError("no renderable mesh geometry found")
    return meshes, np.asarray(bounds)


def _vertices(meshes):
    return np.concatenate([np.asarray(m.vertices) for m in meshes if len(m.vertices)], axis=0)


def render(asset, out_path, face=False):
    meshes, bounds = _meshes_and_bounds(asset)
    verts = _vertices(meshes)
    if face:
        y_min, y_max = float(bounds[0, 1]), float(bounds[1, 1])
        head = verts[verts[:, 1] > y_min + 0.70 * (y_max - y_min)]
        if len(head) < 50:
            head = verts
        c = head.mean(0)
        ext = max(np.ptp(head[:, 0]), np.ptp(head[:, 1]), 1e-3)
    else:
        c = bounds.mean(0)
        ext = np.ptp(bounds, axis=0).max()
    width, height = (700, 700) if face else (520, 760)

    sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.45, 0.45, 0.45])
    for mesh in meshes:
        sc.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    cam = pyrender.PerspectiveCamera(yfov=np.pi / (5.0 if face else 3.5))
    pose = np.eye(4)
    pose[0, 3] = c[0]
    pose[1, 3] = c[1]
    pose[2, 3] = (verts[:, 2].max() if face else c[2]) + ext * (1.25 if face else 1.45)
    sc.add(cam, pose=pose)

    for dx, dy, intensity in [(0.6, 0.3, 4.0), (-0.7, 0.2, 2.5)]:
        lp = np.eye(4)
        lp[0, 3] = c[0] + dx * ext
        lp[1, 3] = c[1] + dy * ext
        lp[2, 3] = c[2] + ext * 1.4
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=intensity), pose=lp)

    renderer = pyrender.OffscreenRenderer(width, height)
    flags = pyrender.RenderFlags.FLAT if face else 0
    color, _ = renderer.render(sc, flags=flags)
    renderer.delete()
    imageio.imwrite(out_path, color)


def main():
    if len(sys.argv) < 3:
        print("usage: python tools/texture/render_texture_preview.py <mesh.glb> <out_dir>")
        raise SystemExit(2)
    glb, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    asset = trimesh.load(glb, process=False)
    render(asset, os.path.join(out_dir, "render_front.png"), face=False)
    render(asset, os.path.join(out_dir, "render_face.png"), face=True)
    print("PREVIEW_OK", out_dir)


if __name__ == "__main__":
    main()
