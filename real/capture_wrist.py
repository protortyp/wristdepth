"""Capture real wrist-camera test frames for the depth pilot (run from the so101 repo).

    cd ~/dev/mine/so101 && PYTHONPATH=. uv run --locked python \
        ~/dev/mine/depthpro2-onnx/research/wrist_depth/real/capture_wrist.py OUT_DIR [--n 40] [--dry]

Moves the arm through poses like the synthetic ones (wrist camera 6-30 cm above the desk,
looking down at the work area), waits for it to settle and saves the raw wrist frame with the
settled joint angles. RealArm.move enforces the usual fingertip and link height limits. The arm
parks and relaxes at the end. Leave the ChArUco board where it is: it is the ground truth.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from pickplace import real_exec as rx
from pickplace.arm import MODEL
from pickplace.real import Camera

PAD = np.array([0.016, 0.0, -0.096])
# RealArm.check allows fingertips down to 1.5 mm, which grasping needs. Nothing here touches the
# desk, so demand real clearance: calibration error and gravity sag cost several millimetres.
TIP_CLEARANCE_M = 0.04
LINK_CLEARANCE_M = 0.06


def clearance(fk, q: np.ndarray) -> tuple[float, float]:
    """Lowest fingertip and lowest link above the desk (desk frame), in metres."""
    with torch.no_grad():
        pos, rot = fk.link_transforms(torch.tensor(np.asarray(q, dtype=np.float64)[None]))
    tips = pos[0, 5].numpy() + rx.finger_tips(float(q[5])) @ rot[0, 5].numpy().T
    return float(tips[:, 2].min()), float(pos[0, 2:, 2].numpy().min())


def path_clear(fk, a: np.ndarray, b: np.ndarray) -> bool:
    for s in np.linspace(0, 1, 41):
        tip, link = clearance(fk, a + s * (b - a))
        if tip < TIP_CLEARANCE_M or link < LINK_CLEARANCE_M:
            return False
    return True


def candidate_poses(n: int, seed: int, T_grip_cam: np.ndarray, desk_z: float) -> list[np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    lo, hi = model.jnt_range[:6, 0].copy(), model.jnt_range[:6, 1].copy()
    lo[5], hi[5] = 0.0, 0.9
    lo[0], hi[0] = -0.9, 0.9  # stay in front of the robot
    grip = model.body("gripper").id
    out = []
    while len(out) < n:
        q = rng.uniform(lo, hi)
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = data.xmat[grip].reshape(3, 3), data.xpos[grip]
        T_cam = T @ T_grip_cam
        height, axis = T_cam[2, 3] - desk_z, T_cam[:3, 2]
        if not (0.06 < height < 0.30) or axis[2] > -0.6:
            continue
        look = T_cam[:3, 3] + axis * (height / -axis[2])
        if 0.14 < np.linalg.norm(look[:2]) < 0.38 and look[0] > 0.05:
            out.append(q)
    # Visit in pan order so consecutive moves stay short.
    return sorted(out, key=lambda q: q[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--dry", action="store_true", help="check poses against the safety limits only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wrist = json.loads((rx.CALIB / "wrist.json").read_text())
    T_grip_cam = np.array(wrist["T_gripper_from_cam_cv"])
    desk_z = -float(json.loads((rx.CALIB / "setup.json").read_text()).get("base_above_desk_m", 0.0))

    arm = rx.RealArm(dry=args.dry)
    fk = rx.DeskKinematics().double()
    poses = []
    for q in candidate_poses(args.n * 20, args.seed, T_grip_cam, desk_z):
        tip, link = clearance(fk, q)
        if tip < TIP_CLEARANCE_M or link < LINK_CLEARANCE_M:
            continue
        if poses and not path_clear(fk, poses[-1], q):
            continue
        try:
            arm.check(q, PAD)
        except RuntimeError:
            continue
        poses.append(q)
        if len(poses) == args.n:
            break
    worst = min(clearance(fk, q)[0] for q in poses)
    print(
        f"{len(poses)} poses; lowest fingertip {worst * 1000:.0f} mm above the desk "
        f"(targets and paths >= {TIP_CLEARANCE_M * 1000:.0f} mm)"
    )
    if args.dry:
        return

    cam = Camera(rx.WRIST_DEV)
    records = []
    try:
        arm.to_perceive_pose()
        for i, q in enumerate(poses):
            arm.move(q, 3.0, PAD)
            time.sleep(0.4)
            settled = arm.settled_q()
            frame = cam.frame()
            name = f"wrist_{i:03d}.png"
            cv2.imwrite(str(args.out / name), frame)
            records.append({"file": name, "q_target": q.tolist(), "q_settled": settled.tolist()})
            print(f"{i + 1}/{len(poses)} {name}", flush=True)
            (args.out / "meta.json").write_text(
                json.dumps(
                    {
                        "K": wrist["K"],
                        "dist": wrist["dist"],
                        "T_gripper_from_cam_cv": T_grip_cam.tolist(),
                        "joint_offsets_rad": wrist["joint_offsets_rad"],
                        "frames": records,
                    },
                    indent=1,
                )
            )
    finally:
        cam.close()
        arm.park_and_relax()


if __name__ == "__main__":
    main()
