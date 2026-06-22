"""Headless Blender render of an A-pose GLB mesh in two lighting modes.

Run:
  blender --background --python tools/render/blender_render.py -- --glb <mesh.glb> --out <dir>

Mode A "studio"  : key + fill + rim, neutral world  -> clean relit look
Mode B "raking"  : single grazing side light, dark world -> exposes micro-relief
                   (abdomen indents, blemish raises read strongly under raking light)
"""
import bpy
import sys
import os
import math
from mathutils import Vector


def argval(name, default=None):
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return a[a.index(name) + 1] if name in a else default


GLB = argval("--glb")
OUT = argval("--out", os.path.dirname(GLB) if GLB else ".")
os.makedirs(OUT, exist_ok=True)


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in objs:                                  # smooth shading (no facets)
        for p in o.data.polygons:
            p.use_smooth = True
    return objs


def bounds(objs):
    mn = Vector((1e9, 1e9, 1e9)); mx = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mn = Vector((min(mn[i], w[i]) for i in range(3)))
            mx = Vector((max(mx[i], w[i]) for i in range(3)))
    return mn, mx


def setup_camera(center, radius):
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = center + Vector((0, -radius * 3.0, radius * 0.3))
    # point at center
    d = center - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = 60
    return cam


def add_sun(name, rot_euler, energy, color=(1, 1, 1)):
    l = bpy.data.lights.new(name, "SUN")
    l.energy = energy
    l.color = color
    ob = bpy.data.objects.new(name, l)
    ob.rotation_euler = rot_euler
    bpy.context.scene.collection.objects.link(ob)
    return ob


def world_color(rgb, strength):
    w = bpy.context.scene.world or bpy.data.worlds.new("W")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (*rgb, 1)
    bg.inputs[1].default_value = strength


def make_material(objs):
    for o in objs:
        # preserve an imported UV/texture material (GLB baseColor + normal maps)
        if o.data.uv_layers and o.data.materials and o.data.materials[0] is not None:
            m0 = o.data.materials[0]
            if m0.use_nodes and any(n.type == "TEX_IMAGE" for n in m0.node_tree.nodes):
                continue
        mat = bpy.data.materials.new("relit")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Roughness"].default_value = 0.6
            # use vertex colors if present
            if o.data.color_attributes:
                vc = mat.node_tree.nodes.new("ShaderNodeVertexColor")
                vc.layer_name = o.data.color_attributes[0].name
                mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
        o.data.materials.clear()
        o.data.materials.append(mat)


def render(path, res=1200):
    sc = bpy.context.scene
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def clear_lights():
    for o in list(bpy.context.scene.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)


def main():
    if not GLB or not os.path.exists(GLB):
        print("RENDER_ERROR no glb:", GLB)
        return
    clear()
    objs = import_glb(GLB)
    if not objs:
        print("RENDER_ERROR no mesh in glb")
        return
    make_material(objs)
    mn, mx = bounds(objs)
    center = (mn + mx) * 0.5
    radius = max((mx - mn).length * 0.5, 0.1)
    setup_camera(center, radius)

    # Mode A: studio (key + fill + rim, soft world)
    clear_lights()
    world_color((0.05, 0.05, 0.06), 0.4)
    add_sun("key", (math.radians(55), 0, math.radians(40)), 4.0)
    add_sun("fill", (math.radians(70), 0, math.radians(-120)), 1.2, (0.7, 0.8, 1.0))
    add_sun("rim", (math.radians(120), 0, math.radians(180)), 2.5)
    render(os.path.join(OUT, "blender_studio.png"))
    print("RENDER_OK studio")

    # Mode B: raking side light (reveals micro-relief), dark world
    clear_lights()
    world_color((0.0, 0.0, 0.0), 0.0)
    add_sun("rake", (math.radians(88), 0, math.radians(95)), 6.0)
    render(os.path.join(OUT, "blender_raking.png"))
    print("RENDER_OK raking")

    # Mode C: FACE close-up (judge face fidelity vs source photo)
    clear_lights()
    world_color((0.06, 0.06, 0.07), 0.6)
    add_sun("key", (math.radians(62), 0, math.radians(25)), 3.5)
    add_sun("fill", (math.radians(72), 0, math.radians(-120)), 1.6, (0.8, 0.85, 1.0))
    head_c = Vector((center.x, center.y, mx.z - radius * 0.18))
    setup_camera(head_c, radius * 0.17)
    render(os.path.join(OUT, "blender_face.png"), res=900)
    print("RENDER_OK face")
    print("RENDER_DONE")


main()
