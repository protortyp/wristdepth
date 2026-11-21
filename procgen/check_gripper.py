import blenderproc as bproc  # BlenderProc requires this to be the first line
# isort: skip_file

# Render only the gripper with the real wrist intrinsics and the stock mount, one frame per
# real jaw shot, to check the Blender placement against the real finger silhouettes.
#
#     blenderproc run check_gripper.py GRIPPER_DIR REAL_META.json FINGERS_DIR OUT.npz
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from render import gripper_poses, load_gripper  # noqa: E402

gdir, meta, fingers, out = (
    Path(sys.argv[1]),
    json.load(open(sys.argv[2])),
    Path(sys.argv[3]),
    sys.argv[4],
)
bproc.init()
bproc.renderer.set_max_amount_of_samples(1)
bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
K = np.array(meta["K"])
w, h = 720, 540
bproc.camera.set_intrinsics_from_K_matrix(K, w, h, clip_start=0.005, clip_end=10)
spec, parts = load_gripper(gdir)
for index, (obj, _) in enumerate(parts):
    obj.blender_obj.pass_index = index + 1
index = json.loads((fingers / "index.json").read_text())
T_cv = np.eye(4)
for i, it in enumerate(index):
    bproc.camera.add_camera_pose(
        bproc.math.change_source_coordinate_frame_of_transformation_matrix(T_cv, ["X", "-Y", "-Z"]),
        frame=i,
    )
    for obj, T in gripper_poses(spec, parts, T_cv, it["gripper_rad"], np.eye(4)):
        obj.set_location(T[:3, 3], frame=i)
        obj.set_rotation_mat(T[:3, :3], frame=i)
data = bproc.renderer.render()
np.savez(out, seg=np.stack(data["category_id_segmaps"]))
