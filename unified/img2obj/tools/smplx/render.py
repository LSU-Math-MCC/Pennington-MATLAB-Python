"""Topology-agnostic multi-person overlay renderer (one renderer for every backend's row).

Mirrors CameraHMR's renderer_pyrd: pinhole camera at the origin with the predicted focal,
each person flipped 180deg about x, distinct hue per person, composited over the photo via the
depth buffer. Because every backend's verts are normalized into the same camera convention
(schema.py), CameraHMR / BLADE / SHAPY rows look identical in style -- only the geometry differs.
"""
import os
import sys
import shlex
import colorsys
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def render_overlay(verts_list, faces, focal, img_w, img_h, bg_rgb=None, same_color=False):
    """verts_list: list of (Vi,3) in camera space (translation already applied). Returns HxWx3 uint8.

    On Linux this renders in-process (pyrender/EGL). On Windows there is no GL stack, so it bounces
    to a worker in the WSL `camerahmr` env (same bridge as backends.run) and reads the PNG back."""
    if os.name == "nt":
        return _render_via_wsl(verts_list, faces, focal, img_w, img_h, bg_rgb, same_color)
    return _render_impl(verts_list, faces, focal, img_w, img_h, bg_rgb, same_color)


def _render_via_wsl(verts_list, faces, focal, img_w, img_h, bg_rgb, same_color):
    import uuid
    from PIL import Image
    import backends  # sibling, stdlib-only: gives REPO / CONDA_SH / _sh / _bash
    tmp = os.path.join(backends.REPO, "runs", ".render_tmp")
    os.makedirs(tmp, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    in_npz = os.path.join(tmp, f"in_{tag}.npz")
    out_png = os.path.join(tmp, f"out_{tag}.png")
    # Plain per-person arrays (NO object dtype) so the npz is loadable across numpy versions
    # (Windows numpy 2.x writes it; the WSL env may be numpy 1.x).
    kw = {f"p{i}": np.asarray(v, np.float32) for i, v in enumerate(verts_list)}
    kw.update(n=int(len(verts_list)), faces=np.asarray(faces), focal=float(focal),
              img_w=int(img_w), img_h=int(img_h), same_color=bool(same_color))
    if bg_rgb is not None:
        kw["bg"] = np.asarray(bg_rgb)
    np.savez(in_npz, **kw)
    worker = os.path.join(backends.REPO, "tools", "smplx", "render.py")
    inner = (f"source {backends.CONDA_SH} && conda activate camerahmr && "
             f"python {shlex.quote(backends._sh(worker))} __worker__ "
             f"{shlex.quote(backends._sh(in_npz))} {shlex.quote(backends._sh(out_png))}")
    backends._bash(inner)
    img = np.array(Image.open(out_png).convert("RGB"))
    for f in (in_npz, out_png):
        try: os.remove(f)
        except OSError: pass
    return img


def _render_impl(verts_list, faces, focal, img_w, img_h, bg_rgb=None, same_color=False):
    import trimesh
    import pyrender
    n = len(verts_list)
    scene = pyrender.Scene(bg_color=(1., 1., 1., 1.), ambient_light=np.zeros(3))
    cam = pyrender.camera.IntrinsicsCamera(fx=focal, fy=focal, cx=img_w / 2., cy=img_h / 2.)
    scene.add(cam, pose=np.eye(4))
    for ang, ax in [(-45, [1, 0, 0]), (45, [0, 1, 0])]:
        scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0),
                  pose=trimesh.transformations.rotation_matrix(np.radians(ang), ax))
    flip = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    for i, v in enumerate(verts_list):
        m = trimesh.Trimesh(np.asarray(v), faces, process=False)
        m.apply_transform(flip)
        hue = 1.0 if same_color else (float(i) / max(n, 1))
        mat = pyrender.MetallicRoughnessMaterial(metallicFactor=0.2, alphaMode="OPAQUE",
                                                 baseColorFactor=(*colorsys.hsv_to_rgb(hue, 0.6, 1.0), 1.0))
        scene.add(pyrender.Mesh.from_trimesh(m, material=mat, wireframe=False), "mesh")
    r = pyrender.OffscreenRenderer(img_w, img_h, point_size=1.0)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    rgb = color[:, :, :3]
    if bg_rgb is None:
        return rgb
    mask = (depth > 0)[..., None]
    return (rgb * mask + np.asarray(bg_rgb)[:, :, :3] * (~mask)).astype(np.uint8)


if __name__ == "__main__" and len(sys.argv) >= 4 and sys.argv[1] == "__worker__":
    # invoked inside WSL by _render_via_wsl: in.npz -> out.png
    from PIL import Image
    d = np.load(sys.argv[2], allow_pickle=False)
    people = [d[f"p{i}"] for i in range(int(d["n"]))]
    bg = d["bg"] if "bg" in d.files else None
    out = _render_impl(people, d["faces"], float(d["focal"]),
                       int(d["img_w"]), int(d["img_h"]), bg, bool(d["same_color"]))
    Image.fromarray(out).save(sys.argv[3])
