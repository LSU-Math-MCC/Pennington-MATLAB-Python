"""Render a close orthographic face preview from a GLB/scene."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import trimesh

os.environ["PYOPENGL_PLATFORM"] = "egl"


def main():
    import imageio
    import pyrender

    if len(sys.argv) < 3:
        raise SystemExit("usage: render_face_ortho.py <mesh_or_scene.glb> <out.png>")
    mesh_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    loaded = trimesh.load(mesh_path, process=False)
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1, 1, 1])
    verts = []
    if isinstance(loaded, trimesh.Scene):
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                scene.add(pyrender.Mesh.from_trimesh(geom, smooth=True))
                verts.append(np.asarray(geom.vertices))
    else:
        scene.add(pyrender.Mesh.from_trimesh(loaded, smooth=True))
        verts.append(np.asarray(loaded.vertices))
    v = np.concatenate(verts, axis=0)
    # Focus on upper/front vertices; robust enough for the generated A-pose scenes.
    upper = v[v[:, 1] > np.percentile(v[:, 1], 82)]
    front = upper[upper[:, 2] > np.percentile(upper[:, 2], 35)] if len(upper) else v
    if len(front) < 20:
        front = upper if len(upper) else v
    c = front.mean(0)
    span = max(float(front[:, 0].ptp()), float(front[:, 1].ptp()), 0.25)
    cam = pyrender.OrthographicCamera(xmag=span * 0.72, ymag=span * 0.80)
    p = np.eye(4)
    p[0, 3] = c[0]
    p[1, 3] = c[1]
    p[2, 3] = front[:, 2].max() + span * 4.0
    scene.add(cam, pose=p)
    r = pyrender.OffscreenRenderer(420, 460)
    col, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
    r.delete()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, col)
    print(out_path)


if __name__ == "__main__":
    main()
