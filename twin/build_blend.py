"""Build the desk-twin .blend: splat in metric desk coordinates, desk plane, board, wrist cameras.

    blender -b --factory-startup --python build_blend.py -- SPLAT.ply OUT_DIR

Desk frame: metres, board corner 0 at the origin, desk plane z = 0, z up. Cameras use the real
wrist intrinsics (fx = 557.2 px at 640x480); each is rendered with EEVEE to OUT_DIR/wrist_<i>.png.
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ply, out_dir = sys.argv[sys.argv.index("--") + 1 :]
# Board corners in the final desk frame (after align_board.py's flip, y may run negative).
SIM3 = json.loads((Path(ply).parents[2] / "board_sim3.json").read_text())
WRIST_FX, WRIST_W, WRIST_H = 557.2, 640, 480
BOARD = (9 * 0.028, 6 * 0.028)
CENTER = Vector(SIM3["board_center"][:2] + [0.0])
# (height above desk m, offset from board centre in x/y m, tilt from straight down in degrees)
POSES = [
    (0.06, (0.00, 0.00), 0),
    (0.10, (0.06, -0.04), 20),
    (0.15, (-0.05, 0.03), 35),
    (0.20, (0.10, 0.10), 30),
    (0.25, (0.00, -0.15), 45),
    (0.30, (0.15, -0.05), 15),
]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = WRIST_W, WRIST_H

bpy.ops.wm.ply_import(filepath=ply)
splat = bpy.context.selected_objects[0]
splat.name = "desk_splat"

bpy.ops.mesh.primitive_plane_add(size=1.2, location=CENTER)
desk = bpy.context.active_object
desk.name = "desk_plane_z0"
desk.display_type = "WIRE"
desk.hide_render = True

bpy.ops.mesh.primitive_plane_add(size=1.0, location=CENTER + Vector((0, 0, 0.0002)))
board = bpy.context.active_object
board.name = "charuco_board_252x168mm"
board.scale = (BOARD[0], BOARD[1], 1.0)
board.display_type = "WIRE"
board.hide_render = True

collection = bpy.data.collections.new("wrist_cameras")
scene.collection.children.link(collection)
for i, (height, (dx, dy), tilt) in enumerate(POSES):
    data = bpy.data.cameras.new(f"wrist_{i}")
    data.lens_unit = "FOV"
    data.sensor_fit = "HORIZONTAL"
    data.angle = 2 * math.atan(WRIST_W / 2 / WRIST_FX)
    data.clip_start = 0.005
    cam = bpy.data.objects.new(f"wrist_{i}_{int(height * 100)}cm", data)
    collection.objects.link(cam)
    look = CENTER + Vector((dx, dy, 0.0))
    # Tilt: move the camera back along -y by height * tan(tilt) so it looks at `look` obliquely.
    back = height * math.tan(math.radians(tilt))
    cam.location = look + Vector((0.0, -back, height))
    cam.rotation_euler = (look - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    scene.render.filepath = f"{out_dir}/wrist_{i}_{int(height * 100)}cm.png"
    bpy.ops.render.render(write_still=True)
    print("RENDERED", scene.render.filepath)

scene.camera = collection.objects[0]
bpy.ops.wm.save_as_mainfile(filepath=f"{out_dir}/desk_twin.blend")
print("SAVED", f"{out_dir}/desk_twin.blend")
