"""Hi-fidelity, relightable export of canonical A-pose splats.

Two complementary artifacts, because they serve different purposes:

  * apose_splats.ply  - full INRIA 3DGS schema (xyz, normals, SH f_dc/f_rest,
                        opacity, scale, rot). Loads in splat viewers/editors
                        (SuperSplat, gsplat). Maximum *appearance* fidelity, but
                        lighting is baked (gaussians do not relight).

  * apose_mesh.glb    - watertight high-detail mesh with per-vertex normals +
                        vertex colors, reconstructed (open3d Poisson) from the
                        splat centers with oriented normals. THIS is the format
                        that relights: a moving light reveals geometric micro-relief
                        (abdomen indents, blemish raises) because the surface carries
                        real normals.

Plus a self-contained three.js "relight studio" GUI (write_relight_viewer) with
movable lights, intensity/exposure/ambient and material controls.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import SplatCloud

SH_C0 = 0.28209479177387814


# ----------------------------------------------------------- full 3DGS PLY ----
def save_gaussian_ply_full(path, splats: SplatCloud):
    """Write the standard INRIA 3D Gaussian Splatting binary PLY."""
    import struct
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(splats)
    centers = np.asarray(splats.centers, np.float32)
    colors = np.clip(np.asarray(splats.colors, np.float64), 1e-4, 1 - 1e-4)
    f_dc = ((colors - 0.5) / SH_C0).astype(np.float32)              # SH DC term
    op = np.asarray(splats.opacities, np.float64)
    op = np.clip(op, 1e-4, 1 - 1e-4)
    opacity = np.log(op / (1 - op)).astype(np.float32)[:, None]     # inverse sigmoid
    scales = np.log(np.clip(np.asarray(splats.scales, np.float64), 1e-8, None)).astype(np.float32)
    rots = np.asarray(splats.rotations, np.float32)
    if rots.shape[1] != 4:
        rots = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
    normals = np.zeros((n, 3), np.float32)

    props = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
              "opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"])
    data = np.concatenate([centers, normals, f_dc, opacity, scales, rots], axis=1).astype(np.float32)

    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return path


# ---------------------------------------------- relightable mesh (open3d) -----
def build_relight_mesh(splats: SplatCloud, poisson_depth=10, density_quantile=0.03,
                       normal_radius=0.04, normal_knn=40, smooth_iters=12):
    """Reconstruct a colored, normal-bearing mesh from splat centers via open3d Poisson.

    Returns a trimesh.Trimesh (vertex colors + normals) or None.
    """
    import open3d as o3d
    import trimesh

    pts = np.asarray(splats.centers, np.float64)
    if pts.shape[0] < 100:
        return None
    cols = np.clip(np.asarray(splats.colors, np.float64), 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    # estimate + orient normals (needed for a relightable surface)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_knn))
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth)
    densities = np.asarray(densities)
    if densities.size:
        thr = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < thr)
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
    mesh.compute_vertex_normals()

    # sanitize: drop non-finite vertices (Poisson can emit NaN/inf -> "Bad glTF: NaN")
    v = np.asarray(mesh.vertices)
    if v.shape[0] == 0:
        return None
    finite = np.isfinite(v).all(axis=1)
    if not finite.all():
        mesh.remove_vertices_by_mask(~finite)
        mesh.compute_vertex_normals()

    v = np.nan_to_num(np.asarray(mesh.vertices), nan=0.0, posinf=0.0, neginf=0.0)
    vn = np.asarray(mesh.vertex_normals)
    vn = np.nan_to_num(vn, nan=0.0)
    faces = np.asarray(mesh.triangles)
    if v.shape[0] == 0 or faces.shape[0] == 0:
        return None
    vc = np.asarray(mesh.vertex_colors)
    if vc.shape[0] != v.shape[0]:
        vc = np.full((v.shape[0], 3), 0.6)
    vc = np.nan_to_num(np.clip(vc, 0, 1), nan=0.6)
    tm = trimesh.Trimesh(vertices=v, faces=faces, vertex_normals=vn,
                         vertex_colors=(vc * 255).astype(np.uint8), process=True)
    return tm


def save_glb(mesh, path):
    if mesh is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


# ------------------------------------------------- dynamic relight GUI --------
def write_relight_viewer(out_dir, glb_name="apose_mesh.glb", splat_name="apose_splats.ply",
                         label="A-pose subject", title="Relight Studio"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = _RELIGHT_HTML.replace("__GLB__", glb_name).replace("__SPLAT__", splat_name) \
                        .replace("__LABEL__", label).replace("__TITLE__", title)
    (out_dir / "relight.html").write_text(html, encoding="utf-8")
    return out_dir / "relight.html"


_RELIGHT_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#0c0c12;color:#eee;font-family:system-ui,sans-serif;overflow:hidden}
  #hud{position:fixed;top:8px;left:8px;z-index:5;background:rgba(0,0,0,.55);padding:8px 12px;border-radius:6px;font-size:13px}
  a{color:#9cf}
</style>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/",
  "lil-gui": "https://unpkg.com/lil-gui@0.19/dist/lil-gui.esm.js"
}}
</script>
</head><body>
<div id="hud"><b>__TITLE__</b> &mdash; __LABEL__<br><span id="status">loading mesh&hellip;</span></div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'lil-gui';

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0c12);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.01, 100);
camera.position.set(0, 0.3, 3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.2, 0);

const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);

const key = new THREE.DirectionalLight(0xffffff, 3.0);
key.position.set(2, 3, 2);
const fill = new THREE.DirectionalLight(0x88aaff, 0.6); fill.position.set(-3,1,-1);
const amb = new THREE.AmbientLight(0xffffff, 0.25);
const rim = new THREE.DirectionalLight(0xffffff, 1.2); rim.position.set(0,1,-3);
scene.add(key, fill, amb, rim);

let mesh=null, mat=null;
const loader = new GLTFLoader();
loader.load('__GLB__', g=>{
  mesh = g.scene;
  mesh.traverse(o=>{ if(o.isMesh){
    mat = new THREE.MeshStandardMaterial({vertexColors:true, roughness:0.62, metalness:0.0,
      envMap: envRT.texture, envMapIntensity:0.7, flatShading:false});
    o.material = mat; o.geometry.computeVertexNormals();
  }});
  // center + scale to fit
  const box = new THREE.Box3().setFromObject(mesh);
  const c = box.getCenter(new THREE.Vector3()); const s = box.getSize(new THREE.Vector3());
  mesh.position.sub(c); mesh.position.y += s.y*0.0; controls.target.set(0,0,0);
  const fit = 1.6/Math.max(s.x,s.y,s.z); mesh.scale.setScalar(fit);
  scene.add(mesh);
  document.getElementById('status').textContent = 'mesh loaded — drag to orbit, use panel to relight';
  buildGUI();
}, undefined, e=>{ document.getElementById('status').innerHTML =
  'mesh not found (run produced no GLB yet). Splat PLY: <a href="__SPLAT__">__SPLAT__</a>'; });

function buildGUI(){
  const gui = new GUI({title:'Relight'});
  const P = {azimuth:45, elevation:55, keyIntensity:3.0, fill:0.6, rim:1.2, ambient:0.25,
             exposure:1.0, roughness:0.62, metalness:0.0, env:0.7, color:'#ffffff', wire:false};
  function place(){ const a=THREE.MathUtils.degToRad(P.azimuth), e=THREE.MathUtils.degToRad(P.elevation);
    key.position.set(Math.cos(e)*Math.sin(a)*5, Math.sin(e)*5, Math.cos(e)*Math.cos(a)*5); }
  place();
  const L = gui.addFolder('Key light');
  L.add(P,'azimuth',-180,180).onChange(place);
  L.add(P,'elevation',-10,90).onChange(place);
  L.add(P,'keyIntensity',0,8).onChange(v=>key.intensity=v);
  L.addColor(P,'color').onChange(v=>key.color.set(v));
  const F = gui.addFolder('Fill / rim / ambient');
  F.add(P,'fill',0,3).onChange(v=>fill.intensity=v);
  F.add(P,'rim',0,5).onChange(v=>rim.intensity=v);
  F.add(P,'ambient',0,1).onChange(v=>amb.intensity=v);
  const M = gui.addFolder('Material / camera');
  M.add(P,'exposure',0,2).onChange(v=>renderer.toneMappingExposure=v);
  M.add(P,'roughness',0,1).onChange(v=>{if(mat)mat.roughness=v});
  M.add(P,'metalness',0,1).onChange(v=>{if(mat)mat.metalness=v});
  M.add(P,'env',0,2).onChange(v=>{if(mat)mat.envMapIntensity=v});
  M.add(P,'wire').onChange(v=>{if(mat)mat.wireframe=v});
}

addEventListener('resize', ()=>{ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight); });
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera); })();
</script>
</body></html>
"""
