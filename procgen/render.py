import blenderproc as bproc  # BlenderProc requires this to be the first line
# isort: skip_file

# Procedural wrist-camera tabletop scenes with exact depth, for fine-tuning metric depth.
#
#     blenderproc run render.py --assets DIR --gripper DIR --out DIR --scenes N --seed S
#
# Every scene is a random table (surface, size, clutter, lighting, background) seen through a
# wrist camera that carries the SO-101 gripper. Distributions follow the community census
# (research_notes/01_census.md) and the teleoperation pose study (02_pose_distribution.md).
# Only CC0 assets (Poly Haven, ambientCG) and the Apache-2.0 SO-ARM100 CAD are used, so the
# renders can be released. Output per frame: rgb, z-depth (m, along the optical axis),
# category mask, pinhole K; lens distortion is applied later, in the loader.
import argparse
import colorsys
import json
from pathlib import Path

import bpy
import numpy as np

W, H = 640, 480
FAR = 4.0
FINGER, TABLE, OBJECT, BACKGROUND = 1, 2, 3, 4

# Distance along the optical axis to the table (m) and pitch from straight down (deg).
# The first four components follow SO-101 teleoperation (median ~18 cm); the wide log-uniform
# share covers other arms and mobile bases (DROID wrist views have a 34 cm median and reach
# past 1 m) and keeps the network from learning a preferred distance.
POSE_MIXTURE = [
    (0.20, ("uniform", 0.06, 0.15), ("normal", 28, 10)),
    (0.20, ("uniform", 0.15, 0.30), ("normal", 33, 13)),
    (0.12, ("uniform", 0.30, 0.60), ("normal", 45, 15)),
    (0.10, ("uniform", 0.50, 1.20), ("uniform", 40, 85)),
    (0.38, ("loguniform", 0.05, 1.50), ("uniform", 0, 80)),
]
# Census table surfaces: share, ambientCG name prefixes (None = plain painted colour).
TABLES = [
    (0.37, None),
    (0.33, ("Wood", "WoodFloor", "Planks", "Chipboard", "WoodSiding")),
    (0.20, ("Leather", "Fabric", "Carpet", "Rubber")),
    (0.06, "checker"),
    (0.04, ("Concrete", "Terrazzo", "Tiles", "Marble", "Plastic", "PaintedPlaster")),
]
# Census finger colours (stock printed jaws): share, base RGB range.
FINGER_COLOURS = [
    (0.33, ((0.75, 0.75, 0.72), (0.95, 0.95, 0.93))),  # white
    (0.29, ((0.01, 0.01, 0.01), (0.05, 0.05, 0.05))),  # black
    (0.06, ((0.25, 0.25, 0.25), (0.5, 0.5, 0.5))),  # grey
    (0.10, ((0.8, 0.2, 0.0), (1.0, 0.45, 0.05))),  # orange
    (0.08, ((0.25, 0.05, 0.45), (0.5, 0.15, 0.75))),  # purple
    (0.14, None),  # other saturated colours
]


def pick(rng, table):
    w = np.array([t[0] for t in table])
    return table[rng.choice(len(table), p=w / w.sum())][1]


def draw(rng, spec):
    kind, a, b = spec
    if kind == "uniform":
        return rng.uniform(a, b)
    if kind == "loguniform":
        return float(np.exp(rng.uniform(np.log(a), np.log(b))))
    return rng.normal(a, b)


def principled(name, rgb, roughness, metallic=0.0, transmission=0.0, ior=1.45):
    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", [*rgb, 1.0])
    mat.set_principled_shader_value("Roughness", float(roughness))
    mat.set_principled_shader_value("Metallic", float(metallic))
    if transmission:
        mat.set_principled_shader_value("Transmission Weight", float(transmission))
        mat.set_principled_shader_value("IOR", float(ior))
    return mat


def saturated(rng):
    return colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.6, 1.0), rng.uniform(0.4, 0.95))


def revolve(name, profile, segments=48):
    """Closed (r, z) polyline revolved around z: plates, bowls, cups with wall thickness."""
    prof = np.asarray(profile, float)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts = [(r * np.cos(a), r * np.sin(a), z) for a in ang for r, z in prof]
    n = len(prof)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        for k in range(n - 1):
            faces.append((i * n + k, j * n + k, j * n + k + 1, i * n + k + 1))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = True
    out = bproc.types.MeshObject(obj)
    bpy.ops.object.select_all(action="DESELECT")
    return out


def dishware(rng, kind):
    t = rng.uniform(0.003, 0.006)
    if kind == "plate":
        r, h = rng.uniform(0.08, 0.14), rng.uniform(0.012, 0.025)
        prof = [(0, 0), (r * 0.6, 0), (r, h), (r - t, h), (r * 0.6 - t, t), (0, t)]
    elif kind == "bowl":
        r, h = rng.uniform(0.05, 0.10), rng.uniform(0.04, 0.08)
        prof = [
            (0, 0),
            (r * 0.45, 0),
            (r * 0.85, h * 0.5),
            (r, h),
            (r - t, h),
            (r * 0.85 - t, h * 0.5),
            (r * 0.45 - t, t),
            (0, t),
        ]
    else:  # cup or glass
        r, h = rng.uniform(0.03, 0.05), rng.uniform(0.07, 0.13)
        rb = r * rng.uniform(0.7, 1.0)
        prof = [(0, 0), (rb, 0), (r, h), (r - t, h), (rb - t, t * 2), (0, t * 2)]
    return revolve(kind, prof)


def cc_material(rng, cc_dir: Path, names):
    # Some ambientCG sets lack a colour map and load as nothing; draw again.
    for _ in range(20):
        mats = bproc.loader.load_ccmaterials(
            str(cc_dir), used_assets=[names[rng.integers(len(names))]]
        )
        if mats:
            return mats[0]
    return principled("fallback", rng.uniform(0.2, 0.9, 3), 0.5)


def table_material(rng, cc_dir: Path, cc_names):
    kind = pick(rng, TABLES)
    if kind is None:
        v = rng.uniform(0.55, 0.92)
        tint = rng.uniform(0.95, 1.05, 3)
        return principled("table_paint", np.clip(v * tint, 0, 1), rng.uniform(0.2, 0.7))
    if kind == "checker":
        mat = principled("table_checker", (1, 1, 1), rng.uniform(0.3, 0.7))
        nodes, links = mat.nodes, mat.links
        checker = nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = float(rng.uniform(10, 60))
        checker.inputs["Color1"].default_value = (0.9, 0.9, 0.9, 1)
        checker.inputs["Color2"].default_value = (0.05, 0.05, 0.05, 1)
        links.new(
            checker.outputs["Color"],
            mat.get_the_one_node_with_type("BsdfPrincipled").inputs["Base Color"],
        )
        return mat
    return cc_material(rng, cc_dir, [n for n in cc_names if n.startswith(kind)])


def object_material(rng, cc_dir: Path, cc_names):
    u = rng.random()
    if u < 0.30:  # glazed ceramic, mostly white
        rgb = (rng.uniform(0.8, 0.95),) * 3 if rng.random() < 0.6 else saturated(rng)
        return principled("ceramic", rgb, rng.uniform(0.05, 0.3))
    if u < 0.42:  # clear or tinted glass
        rgb = (1, 1, 1) if rng.random() < 0.7 else saturated(rng)
        return principled("glass", rgb, rng.uniform(0.0, 0.05), transmission=1.0, ior=1.5)
    if u < 0.62:  # coloured plastic
        return principled("plastic", saturated(rng), rng.uniform(0.2, 0.6))
    if u < 0.72:  # metal
        return principled(
            "metal", (rng.uniform(0.6, 0.9),) * 3, rng.uniform(0.1, 0.4), metallic=1.0
        )
    return cc_material(rng, cc_dir, cc_names)


def flat_material(rng, cc_dir: Path, cc_names, kind=None):
    """Material for a flat sheet: plain paper, a printed photo, a marker board or a mat.

    Prints use only the colour map of a texture set, without its normal or height maps, so
    the sheet looks like ink on paper rather than relief.
    """
    u = rng.random()
    if kind == "mat":
        rgb = (
            (rng.uniform(0.02, 0.15),) * 3 if rng.random() < 0.6 else np.array(saturated(rng)) * 0.4
        )
        mat = principled("mat", rgb, rng.uniform(0.5, 0.9))
        if rng.random() < 0.5:
            add_checker(
                mat,
                rng,
                scale=rng.uniform(15, 40),
                dark=rgb,
                light=np.clip(np.array(rgb) + 0.15, 0, 1),
            )
        return mat
    if u < 0.3:
        return principled("paper", (rng.uniform(0.75, 0.97),) * 3, rng.uniform(0.5, 0.9))
    if u < 0.55:
        mat = principled("board", (1, 1, 1), rng.uniform(0.3, 0.8))
        add_checker(
            mat, rng, scale=rng.uniform(3, 25), dark=(0.02, 0.02, 0.02), light=(0.9, 0.9, 0.9)
        )
        return mat
    color_maps = list((cc_dir / cc_names[rng.integers(len(cc_names))]).glob("*_Color.jpg"))
    mat = principled("print", (1, 1, 1), rng.uniform(0.2, 0.8))
    if color_maps:
        tex = mat.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(color_maps[0]), check_existing=True)
        mat.links.new(
            tex.outputs["Color"],
            mat.get_the_one_node_with_type("BsdfPrincipled").inputs["Base Color"],
        )
    return mat


def add_checker(mat, rng, scale, dark, light):
    checker = mat.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = float(scale)
    checker.inputs["Color1"].default_value = (*light, 1)
    checker.inputs["Color2"].default_value = (*dark, 1)
    mat.links.new(
        checker.outputs["Color"],
        mat.get_the_one_node_with_type("BsdfPrincipled").inputs["Base Color"],
    )


def load_gripper(gdir: Path):
    spec = json.loads((gdir / "gripper.json").read_text())
    parts = []
    for m in spec["meshes"]:
        # OBJ import converts Y-up to Blender's Z-up unless told the file is already Z-up.
        obj = bproc.loader.load_obj(str(gdir / m["file"]), forward_axis="Y", up_axis="Z")[0]
        obj.set_cp("category_id", FINGER)
        parts.append((obj, m))
    return spec, parts


def gripper_poses(spec, parts, T_world_cam_cv, jaw_cmd, T_mount):
    jaw = spec["jaw_offset_rad"] + spec["jaw_gain"] * jaw_cmd
    k = int(np.argmin(np.abs(np.array(spec["jaw_grid_rad"]) - jaw)))
    out = []
    for obj, m in parts:
        T = np.array(m["T_cam_cv_from_mesh"][k] if m["moving"] else m["T_cam_cv_from_mesh"])
        out.append((obj, T_world_cam_cv @ T_mount @ T))
    return out


def mount_variant(rng):
    """Census mounts: 86% stock (fingers at the bottom), 10% upside down, 4% sideways."""
    u = rng.random()
    roll = 0.0 if u < 0.86 else np.pi if u < 0.96 else rng.choice([-np.pi / 2, np.pi / 2])
    roll += rng.normal(0, np.radians(3))
    tilt = rng.normal(0, np.radians(3), 2)

    def rot(axis, a):
        c, s = np.cos(a), np.sin(a)
        R = np.eye(3)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        return R

    T = np.eye(4)
    T[:3, :3] = rot(2, roll) @ rot(0, tilt[0]) @ rot(1, tilt[1])
    T[:3, 3] = rng.normal(0, 0.004, 3)
    return T


def camera_pose(rng, table_z, look_xy):
    """OpenCV camera-to-world with the optical axis hitting the table at look_xy."""
    w = np.array([m[0] for m in POSE_MIXTURE])
    comp = POSE_MIXTURE[rng.choice(len(POSE_MIXTURE), p=w / w.sum())]
    d = draw(rng, comp[1])
    pitch = np.radians(np.clip(draw(rng, comp[2]), 0, 85))
    yaw = rng.uniform(0, 2 * np.pi)
    z_axis = np.array([np.sin(pitch) * np.cos(yaw), np.sin(pitch) * np.sin(yaw), -np.cos(pitch)])
    target = np.array([look_xy[0], look_xy[1], table_z])
    pos = target - z_axis * d
    # Wrist roll is free, so the image "up" direction is uniform around the optical axis.
    ref = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    roll = rng.uniform(0, 2 * np.pi)
    xr = np.cos(roll) * x_axis + np.sin(roll) * y_axis
    yr = np.cross(z_axis, xr)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xr, yr, z_axis, pos
    return T, d


def lowest_point(obj, T):
    pts = np.array(obj.get_bound_box(local_coords=True))
    world = (T[:3, :3] @ pts.T).T + T[:3, 3]
    return world[:, 2].min()


def inside_box(obj, p, margin) -> bool:
    pts = np.array(obj.get_bound_box())
    return bool(((p >= pts.min(0) - margin) & (p <= pts.max(0) + margin)).all())


def build_scene(rng, args, cc_dir, cc_names, models, hdris):
    table_z = 0.0
    sx, sy = rng.uniform(0.6, 2.0), rng.uniform(0.5, 1.2)
    table = bproc.object.create_primitive(
        "CUBE", scale=[sx / 2, sy / 2, 0.015], location=[0, 0, table_z - 0.015]
    )
    table.replace_materials(table_material(rng, cc_dir, cc_names))
    table.set_cp("category_id", TABLE)
    floor = bproc.object.create_primitive(
        "PLANE", scale=[4, 4, 1], location=[0, 0, -rng.uniform(0.7, 0.8)]
    )
    floor.replace_materials(cc_material(rng, cc_dir, cc_names))
    floor.set_cp("category_id", BACKGROUND)
    for o in (table, floor):
        o.add_uv_mapping("smart", overwrite=True)
        uv_scale = rng.uniform(0.5, 2.0)
        mesh = o.get_mesh()
        for loop in mesh.uv_layers.active.data:
            loop.uv = loop.uv * uv_scale * (max(sx, sy) if o is table else 4)

    # A room: walls at random gaps beyond the table edges, so views past the table see
    # labelled surfaces at 0.5-3 m as in real labs, not only the environment map.
    for side in range(4):
        if rng.random() < 0.25:
            continue
        gap = rng.uniform(0.1, 1.5)
        along_x = side < 2
        sign = 1 if side % 2 == 0 else -1
        offset = (sy / 2 if along_x else sx / 2) + gap
        wall = bproc.object.create_primitive("PLANE", scale=[3, 1.5, 1])
        wall.set_location([0, sign * offset, 0.5] if along_x else [sign * offset, 0, 0.5])
        wall.set_rotation_euler([np.pi / 2, 0, 0] if along_x else [np.pi / 2, 0, np.pi / 2])
        wall.replace_materials(
            cc_material(rng, cc_dir, cc_names)
            if rng.random() < 0.5
            else principled("wall_paint", rng.uniform(0.5, 0.95, 3) * rng.uniform(0.8, 1), 0.8)
        )
        wall.set_cp("category_id", BACKGROUND)
        wall.add_uv_mapping("smart", overwrite=True)
    bproc.world.set_world_background_hdr_img(
        str(hdris[rng.integers(len(hdris))]),
        strength=float(rng.uniform(0.3, 1.5)),
        rotation_euler=[0, 0, float(rng.uniform(0, 2 * np.pi))],
    )
    for _ in range(rng.integers(0, 3)):
        light = bproc.types.Light()
        light.set_type(rng.choice(["POINT", "AREA", "SPOT"]))
        light.set_location([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.6, 2.0)])
        light.set_energy(float(rng.uniform(20, 300)))
        light.set_color(np.clip(rng.normal(1, 0.08, 3), 0.7, 1).tolist())
        if light.get_type() == "AREA":
            light.blender_obj.data.size = float(rng.uniform(0.05, 1.0))

    objects = []
    n = int(rng.integers(2, 14))
    for _ in range(n):
        u = rng.random()
        if u < 0.45 and models:
            m = models[rng.integers(len(models))]
            parts = [
                o
                for o in bproc.loader.load_blend(m["blend"])
                if isinstance(o, bproc.types.MeshObject)
            ]
            if not parts:
                continue
            obj = parts[0]
            if len(parts) > 1:
                obj.join_with_other_objects(parts[1:])
            obj.set_cp("haven", m["name"])
            if rng.random() < 0.3:
                obj.replace_materials(object_material(rng, cc_dir, cc_names))
        elif u < 0.75:
            obj = dishware(rng, rng.choice(["plate", "bowl", "cup", "cup"]))
            obj.replace_materials(object_material(rng, cc_dir, cc_names))
        else:
            kind = rng.choice(["CUBE", "CYLINDER", "SPHERE", "CONE"])
            obj = bproc.object.create_primitive(kind)
            obj.set_scale((rng.uniform(0.01, 0.07, 3)).tolist())
            obj.replace_materials(object_material(rng, cc_dir, cc_names))
        obj.set_cp("category_id", OBJECT)
        objects.append(obj)
    # Flat prints, markers, paper and mats lying on the table. Real desks are full of
    # high-contrast flat things (calibration boards, notes, mats); without them the network
    # learns that a sharp-edged patch of new texture is a raised object.
    flats = []
    if rng.random() < 0.3:
        flats.append(("mat", rng.uniform(0.15, 0.35), rng.uniform(0.12, 0.3)))
    for _ in range(rng.integers(0, 7)):
        flats.append((None, rng.uniform(0.03, 0.15), rng.uniform(0.03, 0.2)))
    for layer, (kind, hx, hy) in enumerate(flats):
        p = bproc.object.create_primitive("PLANE", scale=[hx, hy, 1])
        p.add_uv_mapping("smart", overwrite=True)
        p.replace_materials(flat_material(rng, cc_dir, cc_names, kind))
        p.set_cp("category_id", TABLE)
        p.set_location(
            [
                rng.uniform(-sx / 2, sx / 2) * 0.8,
                rng.uniform(-sy / 2, sy / 2) * 0.8,
                table_z + 0.0004 * (layer + 1),
            ]
        )
        p.set_rotation_euler([0, 0, rng.uniform(0, np.pi)])

    def sample(obj):
        obj.set_location(
            [
                rng.uniform(-sx / 2, sx / 2) * 0.85,
                rng.uniform(-sy / 2, sy / 2) * 0.85,
                table_z + 0.2,
            ]
        )
        obj.set_rotation_euler([0, 0, rng.uniform(0, 2 * np.pi)])

    placed = bproc.object.sample_poses_on_surface(
        objects,
        table,
        sample,
        min_distance=0.01,
        max_distance=10,
        check_all_bb_corners_over_surface=False,
    )
    for o in objects:
        if o not in placed:
            o.delete()
    return table_z, (sx, sy), placed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--gripper", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--frames-per-scene", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--denoiser", default="OPTIX", help="OPTIX, INTEL or none")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cc_dir = args.assets / "cc_textures"
    cc_names = sorted(p.name for p in cc_dir.iterdir() if p.is_dir())
    models = [
        m
        for m in json.loads((args.assets / "haven_models.json").read_text())
        if 0.02 <= max(m["size_m"]) <= 0.35
    ]
    for m in models:  # the packed manifest stores paths relative to the asset folder
        m["blend"] = m["blend"].replace("ASSETS", str(args.assets))
    hdris = sorted((args.assets / "haven" / "hdris").rglob("*.hdr"))

    bproc.init()
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.set_noise_threshold(0.02)
    bproc.renderer.set_denoiser(None if args.denoiser == "none" else args.denoiser)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id"], default_values={"category_id": BACKGROUND}
    )

    for scene in range(args.scenes):
        rng = np.random.default_rng([args.seed, scene])
        bproc.clean_up(clean_up_camera=True)
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        bpy.context.scene.world.use_nodes = True
        table_z, size, objects = build_scene(rng, args, cc_dir, cc_names, models, hdris)
        spec, parts = load_gripper(args.gripper)
        fmat = pick(rng, FINGER_COLOURS)
        rgb = saturated(rng) if fmat is None else rng.uniform(fmat[0], fmat[1])
        finger_mat = principled("finger", rgb, rng.uniform(0.35, 0.75))
        for obj, m in parts:
            obj.replace_materials(
                finger_mat
                if "jaw" in m["name"] or "follower" in m["name"]
                else principled("servo", (0.02, 0.02, 0.02), 0.5)
            )
        T_mount = mount_variant(rng)

        hfov = np.radians(rng.uniform(60, 100))
        f = (W / 2) / np.tan(hfov / 2)
        K = np.array(
            [[f, 0, W / 2 + rng.normal(0, 4)], [0, f, H / 2 + rng.normal(0, 4)], [0, 0, 1]]
        )
        bproc.camera.set_intrinsics_from_K_matrix(K, W, H, clip_start=0.005, clip_end=50)

        frames = []
        tries = 0
        while len(frames) < args.frames_per_scene and tries < args.frames_per_scene * 30:
            tries += 1
            if objects and rng.random() < 0.6:
                c = objects[rng.integers(len(objects))].get_location()[:2] + rng.normal(0, 0.03, 2)
            else:
                c = np.array(
                    [rng.uniform(-size[0] / 2, size[0] / 2), rng.uniform(-size[1] / 2, size[1] / 2)]
                )
            T_cv, d = camera_pose(rng, table_z, c)
            jaw_cmd = 1.48 * rng.beta(1.2, 4.0)
            poses = gripper_poses(spec, parts, T_cv, jaw_cmd, T_mount)
            # Fingers and camera must stay clear of the table surface.
            if (
                T_cv[2, 3] < table_z + 0.02
                or min(lowest_point(o, T) for o, T in poses) < table_z + 0.003
            ):
                continue
            if any(inside_box(o, T_cv[:3, 3], 0.01) for o in objects):
                continue
            frame = len(frames)
            T_gl = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
                T_cv, ["X", "-Y", "-Z"]
            )
            bproc.camera.add_camera_pose(T_gl, frame=frame)
            for obj, T in poses:
                obj.set_location(T[:3, 3], frame=frame)
                obj.set_rotation_mat(T[:3, :3], frame=frame)
            frames.append(
                {"jaw_cmd": float(jaw_cmd), "dist_m": float(d), "T_world_cam_cv": T_cv.tolist()}
            )

        if not frames:
            continue
        # BlenderProc numbers objects only when segmentation is enabled; objects of later
        # scenes need their own index or they all read as background.
        for index, obj in enumerate(bproc.object.get_all_mesh_objects()):
            obj.blender_obj.pass_index = index + 1
        data = bproc.renderer.render()
        for i, meta in enumerate(frames):
            depth = np.asarray(data["depth"][i], np.float32)
            seg = np.asarray(data["category_id_segmaps"][i], np.uint8)
            depth[(depth > FAR) | ~np.isfinite(depth)] = 0
            np.savez_compressed(
                args.out / f"s{args.seed:03d}_{scene:05d}_{i:02d}.npz",
                rgb=np.asarray(data["colors"][i], np.uint8),
                depth=depth,
                seg=seg,
                K=K.astype(np.float32),
                f_px=np.float32(K[0, 0]),
                meta=json.dumps({**meta, "mount": T_mount.tolist()}),
            )
        print(f"scene {scene}: {len(frames)} frames, {len(objects)} objects", flush=True)


if __name__ == "__main__":
    main()
