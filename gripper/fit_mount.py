"""Fit the wrist-camera mount pose and jaw-angle mapping to real finger silhouettes.

    MUJOCO_GL=egl python fit_mount.py REAL_META.json FINGERS_DIR OUT.json   (so101 venv)

The hand-eye calibration places the camera from board views far from the gripper, so a few
millimetres of mount error barely matter there but shift the fingers, 3-10 cm from the lens,
by tens of pixels. Fitting the CAD jaw silhouettes to real jaw shots (one per opening) fixes
the pose where it matters for rendering fingers. Fit: small rotation + translation of the
camera in the gripper frame, plus jaw angle = a + b * commanded angle.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# SO-101 MJCF from TheRobotStudio/SO-ARM100 (Apache-2.0).
ARM_XML = os.environ.get("SO101_XML", "vendor/SO-ARM100/Simulation/SO101/so101_new_calib.xml")
W, H = 360, 270


class SilhouetteRenderer:
    def __init__(self, K: np.ndarray):
        self.K = K
        spec = mujoco.MjSpec.from_file(ARM_XML)
        spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
        cam = spec.body("gripper").add_camera(name="wrist")
        cam.sensor_size = [W / K[0, 0] * 1e-3, H / K[1, 1] * 1e-3]
        cam.focal_length = [1e-3, 1e-3]
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.cam = self.model.camera("wrist").id
        self.r = mujoco.Renderer(self.model, H, W)
        self.r.enable_segmentation_rendering()
        # MuJoCo renders with a centred principal point; shift to the calibrated one.
        self.shift = np.float32([[1, 0, K[0, 2] - W / 2], [0, 1, K[1, 2] - H / 2]])

    def render(self, T_grip_cam: np.ndarray, jaw: float) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (T_grip_cam[:3, :3] @ np.diag([1, -1, -1])).flatten())
        self.model.cam_pos[self.cam], self.model.cam_quat[self.cam] = T_grip_cam[:3, 3], q
        self.data.qpos[:] = 0
        self.data.qpos[5] = jaw
        mujoco.mj_forward(self.model, self.data)
        self.r.update_scene(self.data, camera="wrist")
        sil = (self.r.render()[..., 0] >= 0).astype(np.uint8)
        return cv2.warpAffine(sil, self.shift, (W, H), flags=cv2.INTER_NEAREST) > 0


def apply(T0: np.ndarray, x: np.ndarray) -> np.ndarray:
    T = T0.copy()
    T[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = T0[:3, 3] + x[3:6]
    return T


def main() -> None:
    meta = json.loads(Path(sys.argv[1]).read_text())
    fingers, out = sys.argv[2], sys.argv[3]
    T0 = np.array(meta["T_gripper_from_cam_cv"])
    index = json.loads((Path(fingers) / "index.json").read_text())
    if not index:
        raise ValueError("At least one gripper sprite is required")
    masks = [cv2.imread(f"{fingers}/{it['file']}", cv2.IMREAD_UNCHANGED)[..., 3] for it in index]
    source_height, source_width = masks[0].shape
    if any(mask.shape != (source_height, source_width) for mask in masks):
        raise ValueError("All gripper sprites must share one image size")
    # Older capture metadata omits the calibration size. Its sprites are cut from
    # the undistorted wrist images on the original calibration pixel grid.
    calib_width, calib_height = meta.get("size", (source_width, source_height))
    K = np.diag([W / calib_width, H / calib_height, 1.0]) @ np.asarray(meta["K"])
    real = [cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0 for mask in masks]
    rr = SilhouetteRenderer(K)

    def ious(x):
        T = apply(T0, x)
        out = []
        for it, b in zip(index, real):
            a = rr.render(T, x[6] + x[7] * it["gripper_rad"])
            out.append((a & b).sum() / max(1, (a | b).sum()))
        return np.array(out)

    x = np.array([0, 0, 0, 0, 0, 0, 0, 1.0])
    best = -ious(x).mean()
    print(f"calibrated mount IoU {-best:.3f}", flush=True)
    rng = np.random.default_rng(0)
    steps = np.array([0.03, 0.03, 0.03, 0.003, 0.003, 0.003, 0.05, 0.05])
    for i in range(800):
        c = x + rng.normal(0, 1, 8) * steps * (0.3 if i > 400 else 1.0)
        v = -ious(c).mean()
        if v < best:
            x, best = c, v
    res = minimize(
        lambda c: -ious(c).mean(),
        x,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
    )
    if res.fun < best:
        x = res.x
    per = ious(x)
    print(f"fitted IoU {per.mean():.3f} per opening {np.round(per, 2)}")
    print(
        f"rotation {np.degrees(np.linalg.norm(x[:3])):.1f} deg, translation {np.linalg.norm(x[3:6]) * 1000:.1f} mm,"
        f" jaw = {x[6]:+.3f} + {x[7]:.3f} * cmd"
    )
    result = {
        "T_gripper_from_cam_cv": apply(T0, x).tolist(),
        "jaw_offset_rad": x[6],
        "jaw_gain": x[7],
        "iou": per.tolist(),
    }
    Path(out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
