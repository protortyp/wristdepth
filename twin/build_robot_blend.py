"""Desk twin in the robot base frame, with the robot's real overhead and wrist cameras.

    blender -b --factory-startup --python build_robot_blend.py -- SPLAT_DIR OUT_DIR [SPLAT.ply]

Reads SPLAT_DIR/robot_in_twin.json (register_robot.py). The splat is moved from the board frame
into the robot base frame; cameras get the calibrated intrinsics, including the principal point.
The file opens looking through the wrist camera in Rendered shading. Each camera is rendered
to OUT_DIR/<name>.png.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1 :]
splat_dir, out_dir = Path(argv[0]), Path(argv[1])
ply = Path(argv[2]) if len(argv) > 2 else splat_dir / "gs_out/ply/point_cloud_29999.ply"
reg = json.loads((splat_dir / "robot_in_twin.json").read_text())
T_base_desk = np.linalg.inv(np.array(reg["T_desk_from_base"]))
# OpenCV camera (x right, y down, z forward) -> Blender camera (x right, y up, -z forward).
CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])

bpy.ops.wm.read_factory_settings(use_empty=False)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"

bpy.ops.wm.ply_import(filepath=str(ply))
splat = bpy.context.selected_objects[0]
splat.name = "desk_splat"
splat.matrix_world = Matrix(T_base_desk.tolist())

desk_z = float((T_base_desk @ np.array([0, 0, 0, 1.0]))[2])
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.25, 0.0, desk_z))
desk = bpy.context.active_object
desk.name = "desk_plane"
desk.display_type = "WIRE"
desk.hide_render = True

base = bpy.data.objects.new("robot_base", None)
base.empty_display_type = "ARROWS"
base.empty_display_size = 0.1
scene.collection.objects.link(base)


def add_camera(name: str, T_base_cam_cv: np.ndarray, K: np.ndarray, w: int, h: int):
    data = bpy.data.cameras.new(name)
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = K[0, 0] / w * data.sensor_width
    # Principal point offset, in units of the image width (Blender's shift convention).
    data.shift_x = -(K[0, 2] - w / 2) / w
    data.shift_y = (K[1, 2] - h / 2) / w
    data.clip_start = 0.005
    cam = bpy.data.objects.new(name, data)
    scene.collection.objects.link(cam)
    cam.matrix_world = Matrix((T_base_cam_cv @ CV_TO_BLENDER).tolist())
    cam["resolution"] = (w, h)
    return cam


cams = reg["cameras"]
T_desk_base = np.array(reg["T_desk_from_base"])
over_K = np.array(cams["overhead"]["K_640x480"])
over_K[:2] *= 2.25
over_K[0, 2] += 240.0
over_T = T_base_desk @ np.array(cams["overhead"]["T_desk_from_cam_cv"])
made = [add_camera("overhead_1920x1080", over_T, over_K, 1920, 1080)]
if "wrist_recorded" in cams:
    wrist_T = T_base_desk @ np.array(cams["wrist_recorded"]["T_desk_from_cam_cv"])
    made.append(
        add_camera(
            "wrist_recorded_640x480",
            wrist_T,
            np.array(cams["wrist_recorded"]["K_640x480"]),
            640,
            480,
        )
    )

for cam in made:
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = cam["resolution"]
    scene.render.filepath = str(out_dir / f"{cam.name}.png")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", scene.render.filepath)

# Open looking through the wrist camera, rendered.
scene.camera = made[-1]
scene.render.resolution_x, scene.render.resolution_y = made[-1]["resolution"]
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        space = area.spaces.active
        space.shading.type = "RENDERED"
        space.region_3d.view_perspective = "CAMERA"
        space.clip_start = 0.005
bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "desk_twin_robot.blend"))
print("SAVED", out_dir / "desk_twin_robot.blend")
