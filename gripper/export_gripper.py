"""Export the SO-101 gripper meshes posed in the wrist camera frame, for the Blender generator.

    MUJOCO_GL=egl python export_gripper.py MOUNT_FIT.json OUT_DIR   (so101 venv)

Writes one OBJ per mesh plus gripper.json: per mesh its 4x4 pose in the OpenCV camera frame,
with the moving jaw sampled over the joint range. Vertices come from the compiled model, not
the STL files: MuJoCo re-centres and rotates each mesh to its inertial frame, and the geom
poses refer to that frame. Mount pose and jaw mapping come from
fit_mount.py, so commanded openings render where the real fingers appear.
"""

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

XML = Path(os.environ.get("SO101_XML", "vendor/SO-ARM100/Simulation/SO101/so101_new_calib.xml"))
JAW_GRID = np.linspace(-0.17, 1.74, 40)


def geom_pose(model, data, g) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = data.geom_xmat[g].reshape(3, 3)
    T[:3, 3] = data.geom_xpos[g]
    return T


def write_obj(model, mid: int, path: Path) -> None:
    v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    verts = model.mesh_vert[v0 : v0 + nv]
    faces = model.mesh_face[f0 : f0 + nf] + 1
    with open(path, "w") as fh:
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts)
        fh.writelines(f"f {a} {b} {c}\n" for a, b, c in faces)


def main() -> None:
    fit = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    T_grip_cam = np.array(fit["T_gripper_from_cam_cv"])
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    grip, jaw = model.body("gripper").id, model.body("moving_jaw_so101_v1").id

    meshes = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2:
            continue
        if model.geom_bodyid[g] not in (grip, jaw):
            continue
        mid = model.geom_dataid[g]
        name = model.mesh(mid).name
        file = f"{name}.obj"
        write_obj(model, mid, out / file)
        poses = []
        for q in JAW_GRID:
            data.qpos[:] = 0
            data.qpos[5] = q
            mujoco.mj_kinematics(model, data)
            T_world_grip = np.eye(4)
            T_world_grip[:3, :3] = data.xmat[grip].reshape(3, 3)
            T_world_grip[:3, 3] = data.xpos[grip]
            T_cam = np.linalg.inv(T_world_grip @ T_grip_cam) @ geom_pose(model, data, g)
            poses.append(T_cam.tolist())
        moving = model.geom_bodyid[g] == jaw
        meshes.append(
            {
                "name": name,
                "file": file,
                "moving": bool(moving),
                "T_cam_cv_from_mesh": poses if moving else poses[0],
            }
        )
        print(
            name,
            "moving" if moving else "fixed",
            "origin in camera (m)",
            np.round(np.array(poses[0])[:3, 3], 3),
        )
    json.dump(
        {
            "jaw_grid_rad": JAW_GRID.tolist(),
            "jaw_offset_rad": fit["jaw_offset_rad"],
            "jaw_gain": fit["jaw_gain"],
            "meshes": meshes,
        },
        open(out / "gripper.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
