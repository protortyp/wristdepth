"""MuJoCo layers for synthetic wrist-camera frames: inserted objects and their desk shadows.

    python mj_layers.py SPLAT_DIR OUT_DIR --frames 2000 --seed 0     (so101 venv: mujoco, numpy)

For each frame: a random arm configuration whose wrist camera looks down at the work area from
4-35 cm, a few objects placed in view, and renders at the real wrist intrinsics (640x480):
RGB, metric z-depth, class (0 none, 1 desk, 3 object) and the desk shadow ratio (render with
objects over render without). The arm meshes are not drawn: near the camera the MuJoCo gripper
does not match the real one, so the compositor pastes real finger sprites instead (see
finger_sprites.py). Poses are in the robot base frame; the desk plane height comes from the
twin's robot registration so both layers share one desk.
"""

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

# Checkout of the so101 robot project (kinematics, vendored SO-ARM100 model).
SO101 = Path(os.environ.get("SO101_ROOT", "../so101"))
ARM_XML = SO101 / "vendor/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
W, H = 640, 480
N_SLOTS = 7
CV_TO_GL = np.diag([1.0, -1.0, -1.0])
ARM_GROUP, OBJ_GROUP, DESK_GROUP = 2, 4, 5


def build(desk_z: float, K: np.ndarray) -> tuple[mujoco.MjModel, list[str]]:
    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.group = ARM_GROUP  # visual meshes
        else:
            geom.group = 3  # collision meshes stay hidden
    world = spec.worldbody
    desk = world.add_geom()
    desk.type = mujoco.mjtGeom.mjGEOM_PLANE
    desk.size = [2.0, 2.0, 0.01]
    desk.pos = [0.0, 0.0, desk_z]
    desk.rgba = [0.8, 0.8, 0.8, 1.0]
    desk.group = DESK_GROUP
    desk.name = "desk"
    for i in range(N_SLOTS):
        body = world.add_body()
        body.name = f"obj{i}"
        body.mocap = True
        geom = body.add_geom()
        geom.name = f"obj{i}_geom"
        geom.type = [
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
        ][i]
        geom.size = [0.02, 0.02, 0.02]
        geom.group = OBJ_GROUP
        geom.contype = geom.conaffinity = 0
    light = world.add_light()
    light.name = "key"
    light.pos = [0.2, 0.0, 1.0]
    light.dir = [0.0, 0.0, -1.0]
    light.castshadow = True
    cam = world.add_camera()
    cam.name = "wrist"
    cam.resolution = [W, H]
    # MuJoCo needs a physical sensor size with pixel intrinsics; 10 um pixels, any value works.
    cam.sensor_size = [W * 1e-5, H * 1e-5]
    cam.focal_pixel = [float(K[0, 0]), float(K[1, 1])]
    cam.principal_pixel = [float(K[0, 2]), float(K[1, 2])]
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.map.znear = 0.002
    model.vis.quality.shadowsize = 4096
    return model, [f"obj{i}" for i in range(N_SLOTS)]


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


# Wrist poses seen in community SO-101 teleoperation (research_notes/02_pose_distribution.md):
# (weight, distance range m along the optical axis, pitch from straight down in degrees).
# 30% is a flat log-uniform component so the network cannot learn one preferred distance.
POSE_MIXTURE = [
    (0.30, ("uniform", 0.08, 0.15), ("normal", 28, 10)),
    (0.25, ("uniform", 0.15, 0.25), ("normal", 33, 13)),
    (0.10, ("uniform", 0.25, 0.40), ("normal", 45, 15)),
    (0.05, ("uniform", 0.40, 0.70), ("uniform", 50, 85)),
    (0.30, ("loguniform", 0.08, 0.60), ("uniform", 5, 70)),
]


def _draw(rng, spec):
    kind, a, b = spec
    if kind == "uniform":
        return rng.uniform(a, b)
    if kind == "loguniform":
        return float(np.exp(rng.uniform(np.log(a), np.log(b))))
    return rng.normal(a, b)


def on_desk_map(desk_map, p_base: np.ndarray) -> bool:
    if desk_map is None:
        return True
    p = desk_map["T_desk_from_base"] @ np.array([p_base[0], p_base[1], p_base[2], 1.0])
    idx = ((p[:2] - desk_map["lo_desk"]) / desk_map["cell"]).astype(int)
    g = desk_map["grid"]
    return bool(0 <= idx[0] < g.shape[0] and 0 <= idx[1] < g.shape[1] and g[idx[0], idx[1]])


def view_on_desk(desk_map, T_cam, K, desk_z) -> bool:
    """Optical centre and at least 3 of 4 half-way image corners hit the mapped desk."""
    if desk_map is None:
        return True
    hits = 0
    for u, v in (
        (W * 0.25, H * 0.25),
        (W * 0.75, H * 0.25),
        (W * 0.25, H * 0.75),
        (W * 0.75, H * 0.75),
    ):
        ray = T_cam[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        if ray[2] < -1e-3:
            hits += on_desk_map(desk_map, T_cam[:3, 3] + ray * ((desk_z - T_cam[2, 3]) / ray[2]))
    return hits >= 3


def sample_pose(model, data, rng, T_grip_cam, desk_z, desk_map=None, K=None):
    """A joint configuration matching a (distance, pitch) target drawn from POSE_MIXTURE."""
    lo, hi = model.jnt_range[:6, 0].copy(), model.jnt_range[:6, 1].copy()
    grip = model.body("gripper").id
    weights = np.array([m[0] for m in POSE_MIXTURE])
    for _ in range(200):
        comp = POSE_MIXTURE[rng.choice(len(POSE_MIXTURE), p=weights / weights.sum())]
        target_d = _draw(rng, comp[1])
        target_pitch = float(np.clip(_draw(rng, comp[2]), 0, 85))
        for _ in range(4000):
            q = rng.uniform(lo, hi)
            # Real grippers are mostly nearly closed (median opening 14/100).
            q[5] = 1.48 * rng.beta(1.2, 4.0)
            data.qpos[:6] = q
            mujoco.mj_kinematics(model, data)
            T = np.eye(4)
            T[:3, :3] = data.xmat[grip].reshape(3, 3)
            T[:3, 3] = data.xpos[grip]
            T_cam = T @ T_grip_cam
            height, axis = T_cam[2, 3] - desk_z, T_cam[:3, 2]
            if height < 0.03 or axis[2] >= -0.05:
                continue
            pitch = np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))
            dist = height / -axis[2]
            if (
                abs(dist - target_d) > 0.02 * max(1.0, target_d / 0.2)
                or abs(pitch - target_pitch) > 8
            ):
                continue
            look = T_cam[:3, 3] + axis * dist
            if not 0.08 < np.linalg.norm(look[:2]) < 0.6:
                continue
            if on_desk_map(desk_map, look) and view_on_desk(desk_map, T_cam, K, desk_z):
                return q, T_cam, look
    raise RuntimeError("no valid pose found")


def place_objects(model, data, rng, slots, look, desk_z, tips, T_cam, K):
    """Random sizes, colours and resting poses near the camera's look-at point."""
    placed = []
    for i, name in enumerate(slots):
        gid = model.geom(f"{name}_geom").id
        bid = model.body(name).mocapid[0]
        if rng.random() < 0.55:  # leave some slots empty
            data.mocap_pos[bid] = [0, 0, -5]
            continue
        gtype = model.geom_type[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r = rng.uniform(0.015, 0.06) if i < 2 else rng.uniform(0.06, 0.13)  # cup/can vs plate
            h = rng.uniform(0.02, 0.12) if i < 2 else rng.uniform(0.008, 0.025)
            model.geom_size[gid] = [r, h / 2, 0]
            half_z, foot = h / 2, r
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            s = rng.uniform([0.01, 0.01, 0.008], [0.06, 0.05, 0.06])
            model.geom_size[gid] = s
            half_z, foot = s[2], float(np.hypot(s[0], s[1]))
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = rng.uniform(0.012, 0.04)
            model.geom_size[gid] = [r, 0, 0]
            half_z, foot = r, r
        else:  # capsule lying on the desk
            r, h = rng.uniform(0.006, 0.02), rng.uniform(0.02, 0.08)
            model.geom_size[gid] = [r, h / 2, 0]
            half_z, foot = r, h / 2 + r
        if desk_z + 2 * half_z > T_cam[2, 3] - 0.03:  # never reaches up to the camera
            data.mocap_pos[bid] = [0, 0, -5]
            continue
        for _ in range(50):
            # A random pixel in the image, projected onto the desk: the object lands in view.
            u, v = rng.uniform(0.1, 0.9) * W, rng.uniform(0.05, 0.85) * H
            ray = T_cam[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
            if ray[2] >= -1e-3:
                continue
            xy = (T_cam[:3, 3] + ray * ((desk_z - T_cam[2, 3]) / ray[2]))[:2]
            clear_tips = all(
                np.linalg.norm(xy - t[:2]) > foot + 0.012 or t[2] > desk_z + 2 * half_z + 0.008
                for t in tips
            )
            clear_objs = all(np.linalg.norm(xy - p) > foot + f for p, f in placed)
            if clear_tips and clear_objs:
                break
        else:
            data.mocap_pos[bid] = [0, 0, -5]
            continue
        placed.append((xy, foot))
        yaw = rng.uniform(0, 2 * np.pi)
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:  # capsule axis horizontal
            R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [np.sin(yaw), 0, -np.cos(yaw)], [0, 1, 0]])
        else:
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        data.mocap_pos[bid] = [xy[0], xy[1], desk_z + half_z]
        data.mocap_quat[bid] = mat_to_quat(R)
        white = rng.random() < 0.5
        rgb = rng.uniform(0.85, 0.97, 3) if white else rng.uniform(0.05, 0.95, 3)
        model.geom_rgba[gid] = [*rgb, 1.0]


def fingertips(model, data):
    out = []
    for body in ("gripper", "moving_jaw_so101_v1"):
        bid = model.body(body).id
        R, p = data.xmat[bid].reshape(3, 3), data.xpos[bid]
        out.append(p + R @ np.array([0.0, 0.0, -0.1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("splat_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--desk-map", type=Path, help="make_desk_map.py output; keep views on the desk")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    reg = json.loads((args.splat_dir / "robot_in_twin.json").read_text())
    T_desk_base = np.array(reg["T_desk_from_base"])
    desk_z = float(
        np.linalg.inv(T_desk_base)[2, 3]
    )  # desk plane z = 0 in desk frame, in base frame
    wrist = json.loads((SO101 / "calib/real/wrist.json").read_text())
    K = np.array(wrist["K"])
    T_grip_cam = np.array(wrist["T_gripper_from_cam_cv"])

    desk_map = dict(np.load(args.desk_map)) if args.desk_map else None
    model, slots = build(desk_z, K)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = model.camera("wrist").id
    opt_all = mujoco.MjvOption()
    for g in range(6):
        opt_all.geomgroup[g] = g in (OBJ_GROUP, DESK_GROUP)
    opt_desk = mujoco.MjvOption()
    for g in range(6):
        opt_desk.geomgroup[g] = g == DESK_GROUP
    arm_geoms = np.where(model.geom_group == ARM_GROUP)[0]
    obj_geoms = np.where(model.geom_group == OBJ_GROUP)[0]
    desk_geom = model.geom("desk").id

    # The first offscreen render after creating the context comes back empty.
    renderer.update_scene(data, camera=cam, scene_option=opt_all)
    renderer.render()

    meta = []
    for i in range(args.frames):
        q, T_cam, look = sample_pose(model, data, rng, T_grip_cam, desk_z, desk_map, K)
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)
        place_objects(model, data, rng, slots, look, desk_z, fingertips(model, data), T_cam, K)
        # Camera before mj_forward: rendering reads the camera pose mj_forward computes.
        model.cam_pos[cam] = T_cam[:3, 3]
        model.cam_quat[cam] = mat_to_quat(T_cam[:3, :3] @ CV_TO_GL)
        mujoco.mj_forward(model, data)
        # Random key light: room ceiling light, somewhat off-axis.
        d = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0])
        model.light_dir[0] = d / np.linalg.norm(d)
        model.light_pos[0] = look - 1.0 * model.light_dir[0]
        model.light_diffuse[0] = [rng.uniform(0.5, 0.8)] * 3

        renderer.update_scene(data, camera=cam, scene_option=opt_all)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        seg = renderer.render()[..., 0].copy()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=cam, scene_option=opt_desk)
        bare = renderer.render().astype(np.float32)

        cls = np.zeros((H, W), np.uint8)
        cls[seg == desk_geom] = 1
        cls[np.isin(seg, arm_geoms)] = 2
        cls[np.isin(seg, obj_geoms)] = 3
        lum = rgb.astype(np.float32).mean(-1)
        shadow = np.clip(lum / np.maximum(bare.mean(-1), 1.0), 0.15, 1.0)
        shadow[cls != 1] = 1.0
        np.savez_compressed(
            args.out_dir / f"{i:05d}.npz",
            rgb=rgb,
            depth=depth.astype(np.float16),
            cls=cls,
            shadow=(shadow * 255).astype(np.uint8),
        )
        meta.append({"frame": i, "q": q.tolist(), "T_base_cam_cv": T_cam.tolist()})
        if i % 200 == 0:
            print(f"{i}/{args.frames}", flush=True)
    (args.out_dir / "meta.json").write_text(
        json.dumps(
            {
                "K": K.tolist(),
                "w": W,
                "h": H,
                "desk_z_base": desk_z,
                "T_desk_from_base": T_desk_base.tolist(),
                "frames": meta,
            }
        )
    )


if __name__ == "__main__":
    main()
