"""
Atomistic render in the house style (Cycles): matte spheres, ambient-
occlusion contact shading, transparent background for compositing on
white, orthographic camera, no bonds unless a bonds table is given.

Run inside Blender, headless:

    Blender -b -P renders/common/render_atoms.py -- \
        --colours atoms.csv --out raw.png \
        [--elev 20 --azim 30] [--samples 128] [--res 1600] [--tilt 0]

The colours table has columns x, y, z (Å), r, g, b (sRGB, 0-1) and an
optional rad (Å) per atom; renders/common/atoms_csv.py writes it from an
ase.Atoms with the element law of mdlab.viz.

Views: 'hero' looks down the direction at elevation --elev and azimuth
--azim (degrees); '100', '110', '111' look down cubic zone axes. The
camera target and orthographic width follow the structure unless
--ortho-scale fixes a common width across a series.

All atoms live in one mesh; per-atom colour is a point-domain attribute,
so continuous colour scales render exactly.
"""

import os
import sys
import csv
import math
import argparse

import bpy
import bmesh
from mathutils import Matrix, Vector

# Cubic zone axes: view direction (camera sits along +axis looking back)
ZONE_AXES = {
    "100": (1.0, 0.0, 0.0),
    "110": (1.0, 1.0, 0.0),
    "111": (1.0, 1.0, 1.0),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--colours", required=True)
    p.add_argument("--bonds", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--view", default="hero",
                   choices=["hero", "ortho_x", "100", "110", "111"])
    p.add_argument("--ortho-scale", type=float, default=None,
                   help="orthographic width in Angstrom (default: fit)")
    p.add_argument("--cell", type=float, default=65.0,
                   help="cell edge for the registered ortho_x view")
    p.add_argument("--elev", type=float, default=45.0)
    p.add_argument("--azim", type=float, default=45.0)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--res", type=int, default=2400)
    p.add_argument("--radius", type=float, default=1.52)
    p.add_argument("--tilt", type=float, default=0.0,
                   help="degrees off the exact zone axis (bcc channels "
                        "are open along <100>/<110>/<111>, so a small "
                        "tilt lets the atoms behind fill them)")
    p.add_argument("--rough", type=float, default=0.62)
    p.add_argument("--spec", type=float, default=0.15)
    p.add_argument("--subdiv", type=int, default=4)
    p.add_argument("--light3", action="store_true")
    p.add_argument("--ao_dist", type=float, default=3.0)
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu", "hybrid"],
                   help="Cycles device: the Metal GPU alone (default), the "
                        "CPU alone, or both together")
    p.add_argument("--ao-floor", type=float, default=0.42,
                   help="minimum ambient-occlusion multiplier; the default "
                        "reproduces the earlier fixed 0.42 + 0.55 AO law, "
                        "and a higher floor keeps more of a quantitative "
                        "colour in atom-atom contacts")
    return p.parse_args(argv)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def read_bonds(path):
    bonds = []
    with open(path) as f:
        for row in csv.DictReader(f):
            p1 = Vector((float(row["x1"]), float(row["y1"]), float(row["z1"])))
            p2 = Vector((float(row["x2"]), float(row["y2"]), float(row["z2"])))
            c1 = tuple(srgb_to_linear(float(row[k])) for k in ("r1", "g1", "b1"))
            c2 = tuple(srgb_to_linear(float(row[k])) for k in ("r2", "g2", "b2"))
            bonds.append((p1, p2, c1, c2, float(row["rad"])))
    return bonds


def read_atoms(path):
    """Per-atom colour table; an optional 'rad' column sets an absolute
    radius, an optional 'radius_scale' column scales the common one."""
    atoms = []
    with open(path) as f:
        for row in csv.DictReader(f):
            atoms.append((float(row["x"]), float(row["y"]), float(row["z"]),
                          tuple(srgb_to_linear(float(row[k]))
                                for k in ("r", "g", "b")),
                          float(row["rad"]) if row.get("rad") else None,
                          float(row["radius_scale"])
                          if row.get("radius_scale") else 1.0))
    return atoms


SPEC = [0.15]
SUBDIV = [4]


def sphere_template(subdivisions):
    """One unit icosphere as arrays, replicated for every atom below."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=1.0)
    bm.verts.ensure_lookup_table()
    verts = [tuple(v.co) for v in bm.verts]
    faces = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    return verts, faces


def build_atoms(atoms, radius, rough, ao_dist, bonds=(), ao_floor=0.42):
    mesh = bpy.data.meshes.new("atoms")
    if bonds:
        # bonded scenes keep the bmesh route, which knows how to cap cones
        bm = bmesh.new()
        vert_colour = []
        for x, y, z, rgb, rad, scale in atoms:
            ret = bmesh.ops.create_icosphere(
                bm, subdivisions=SUBDIV[0], radius=(rad if rad else radius) * scale)
            dv = Vector((x, y, z))
            for v in ret["verts"]:
                v.co += dv
            vert_colour.extend([rgb] * len(ret["verts"]))
        for p1, p2, c1, c2, brad in bonds:
            mid = (p1 + p2) / 2
            for a, b, col in ((p1, mid, c1), (mid, p2, c2)):
                d = b - a
                rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
                mat = Matrix.Translation((a + b) / 2) @ rot
                ret = bmesh.ops.create_cone(bm, cap_ends=True, segments=32,
                                            radius1=brad, radius2=brad,
                                            depth=d.length, matrix=mat)
                vert_colour.extend([col] * len(ret["verts"]))
        bm.to_mesh(mesh)
        bm.free()
        flat = []
        for rgb in vert_colour:
            flat.extend((rgb[0], rgb[1], rgb[2], 1.0))
    else:
        # one template sphere, replicated with numpy: thousands of atoms
        # become one mesh in a fraction of a second
        import numpy as np
        tverts, tfaces = sphere_template(SUBDIV[0])
        tverts = np.asarray(tverts, dtype=np.float64)
        tfaces = np.asarray(tfaces, dtype=np.int64)
        centres = np.asarray([[a[0], a[1], a[2]] for a in atoms])
        radii = np.asarray([(a[4] if a[4] else radius) * a[5] for a in atoms])
        colours = np.asarray([a[3] for a in atoms])
        nv, nf = len(tverts), len(tfaces)
        verts = (tverts[None, :, :] * radii[:, None, None]
                 + centres[:, None, :]).reshape(-1, 3)
        faces = (tfaces[None, :, :]
                 + (np.arange(len(atoms)) * nv)[:, None, None]).reshape(-1, 3)
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        rgba = np.ones((len(atoms), nv, 4))
        rgba[:, :, :3] = colours[:, None, :]
        flat = rgba.reshape(-1).tolist()

    attr = mesh.color_attributes.new(name="atom_rgb", type="FLOAT_COLOR",
                                     domain="POINT")
    attr.data.foreach_set("color", flat)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))

    mat = bpy.data.materials.new("atom_matte")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = rough
    try:
        bsdf.inputs["Specular IOR Level"].default_value = SPEC[0]
    except KeyError:
        pass

    attr_node = nt.nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = "atom_rgb"
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = ao_dist
    if not 0.0 <= ao_floor < 0.97:
        raise ValueError("--ao-floor must satisfy 0 <= value < 0.97")
    # ao_soft = floor + (0.97 - floor) * AO: crevices darken, never to black
    scale = nt.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = 0.97 - ao_floor
    lift = nt.nodes.new("ShaderNodeVectorMath")
    lift.operation = "ADD"
    lift.inputs[1].default_value = (ao_floor, ao_floor, ao_floor)
    mult = nt.nodes.new("ShaderNodeVectorMath")
    mult.operation = "MULTIPLY"
    nt.links.new(ao.outputs["Color"], scale.inputs[0])
    nt.links.new(scale.outputs["Vector"], lift.inputs[0])
    nt.links.new(attr_node.outputs["Color"], mult.inputs[0])
    nt.links.new(lift.outputs["Vector"], mult.inputs[1])
    nt.links.new(mult.outputs["Vector"], bsdf.inputs["Base Color"])

    mesh.materials.append(mat)
    obj = bpy.data.objects.new("atoms", mesh)
    bpy.context.collection.objects.link(obj)


def look_at(obj, target):
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (
        (Vector(target) - obj.location).to_track_quat("-Z", "Y"))


def add_area(name, location, energy, size, target):
    light = bpy.data.lights.new(name, type="AREA")
    light.energy = energy
    light.size = size
    obj = bpy.data.objects.new(name, light)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    look_at(obj, target)


def main():
    import time
    clock = time.time()
    args = parse_args()
    atoms = read_atoms(args.colours)

    coords = [a[:3] for a in atoms]
    centre = Vector((sum(c[0] for c in coords) / len(coords),
                     sum(c[1] for c in coords) / len(coords),
                     sum(c[2] for c in coords) / len(coords)))
    span = 2.0 * max((Vector(c) - centre).length for c in coords)
    fit_scale = span + 2.5 * args.radius + 1.5

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    bonds = read_bonds(args.bonds) if args.bonds else ()
    SPEC[0] = args.spec
    SUBDIV[0] = args.subdiv
    build_atoms(atoms, args.radius, args.rough, args.ao_dist, bonds,
                args.ao_floor)
    print(f"scene built in {time.time() - clock:.1f} s")
    clock = time.time()

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    cam_data.type = "ORTHO"
    cam_data.clip_end = 900.0

    if args.view == "ortho_x":
        vdir = Vector((1.0, 0.0, 0.0))
        if args.tilt:
            t = math.radians(args.tilt)
            vdir = Vector((math.cos(t), 0.42 * math.sin(t),
                           math.sin(t))).normalized()
        cam_data.ortho_scale = args.ortho_scale or args.cell
    else:
        if args.view == "hero":
            er, ar = math.radians(args.elev), math.radians(args.azim)
            vdir = Vector((math.cos(er) * math.cos(ar),
                           math.cos(er) * math.sin(ar),
                           math.sin(er)))
        else:
            vdir = Vector(ZONE_AXES[args.view]).normalized()
        if args.tilt:
            up = Vector((0.0, 0.0, 1.0))
            if abs(vdir.dot(up)) > 0.95:
                up = Vector((0.0, 1.0, 0.0))
            axis = vdir.cross(up).normalized()
            vdir = vdir.copy()
            vdir.rotate(Matrix.Rotation(math.radians(args.tilt), 3, axis))
            vdir.normalize()
        cam_data.ortho_scale = args.ortho_scale or fit_scale

    cam.location = centre + vdir * 250.0
    look_at(cam, centre)

    # Key light in the camera frame, matching the original hero setup:
    # 78 deg off the view axis, high and slightly to the side, with the
    # same irradiance (E/d^2 = 9.05) so shading matches earlier figures.
    up = Vector((0.0, 0.0, 1.0))
    if abs(vdir.dot(up)) > 0.95:
        up = Vector((0.0, 1.0, 0.0))
    side = vdir.cross(up).normalized()
    lift = side.cross(vdir).normalized()
    key_dir = (vdir * 0.205 + side * 0.093 + lift * 0.996).normalized()
    add_area("key", centre + key_dir * 180.0, 2.93e5, 130.0, centre)
    if args.light3:
        fill_dir = (vdir - 1.1 * side + 0.25 * lift).normalized()
        add_area("fill", centre + fill_dir * 180.0, 2.93e5 * 0.30, 200.0, centre)
        rim_dir = (-vdir + 0.9 * lift + 0.5 * side).normalized()
        add_area("rim", centre + rim_dir * 180.0, 2.93e5 * 0.55, 150.0, centre)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 0.97
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if args.device == "cpu":
        scene.cycles.device = "CPU"
    else:
        try:
            prefs.compute_device_type = "METAL"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = (d.type != "CPU") or args.device == "hybrid"
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"
    print("cycles device:", scene.cycles.device,
          [(d.name, d.use) for d in prefs.devices])

    scene.render.film_transparent = True
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    # Blender resolves a relative path against the .blend file, not the shell
    scene.render.filepath = os.path.abspath(args.out)

    bpy.ops.render.render(write_still=True)
    print(f"rendered {args.out} in {time.time() - clock:.1f} s")


if __name__ == "__main__":
    main()
