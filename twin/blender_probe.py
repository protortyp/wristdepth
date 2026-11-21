"""Blender 5.3: import the splat PLY, report what it became, render one frame above the board."""

import sys
import time

import bpy
from mathutils import Vector

ply, out = sys.argv[sys.argv.index("--") + 1 :]
bpy.ops.wm.read_factory_settings(use_empty=True)
t = time.time()
bpy.ops.wm.ply_import(filepath=ply)
obj = bpy.context.selected_objects[0]
print("IMPORTED", obj.type, obj.name, f"{time.time() - t:.1f}s")
data = obj.data
attrs = [a.name for a in getattr(data, "attributes", [])]
print("ATTRS", len(attrs), attrs[:12])
print("POINTS", len(getattr(data, "points", getattr(data, "vertices", []))))

scene = bpy.context.scene
cam_data = bpy.data.cameras.new("wrist")
cam_data.lens_unit = "FOV"
cam_data.angle = 1.05
cam_data.clip_start = 0.01
cam = bpy.data.objects.new("wrist", cam_data)
scene.collection.objects.link(cam)
cam.location = Vector((0.20, -0.10, 0.25))
target = Vector((0.126, 0.084, 0.0))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
scene.render.resolution_x, scene.render.resolution_y = 640, 480
for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try:
        scene.render.engine = engine
        break
    except TypeError:
        continue
scene.render.filepath = out
t = time.time()
bpy.ops.render.render(write_still=True)
print("RENDERED", scene.render.engine, f"{time.time() - t:.1f}s", out)
