"""Ground-truth desk depth for real wrist test frames (twin venv: gsplat, cv2).

    python eval_gt.py REAL_DIR SPLAT_DIR CKPT FINGERS_DIR OUT_DIR

Per frame: undistort with the wrist calibration (same K), camera pose from the settled joints
(MuJoCo FK + wrist mount + the twin's robot registration), exact desk-plane depth on pixels the
twin knows as desk, minus the fingers and anything the twin shows standing on the desk.
Where the ChArUco board is visible, its PnP pose gives an independent desk plane; the report
compares the two (camera-to-desk distance along the optical axis).
"""

import json
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth"))
from compose import desk_extent, load_splat, on_desk, plane_depth, render_background  # noqa: E402

ARM_XML = Path(os.environ.get("SO101_XML", "vendor/SO-ARM100/Simulation/SO101/so101_new_calib.xml"))
BOARD = cv2.aruco.CharucoBoard(
    (9, 6), 0.028, 0.021, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
)


def main() -> None:
    real, splat_dir, ckpt, fingers, out = (Path(p) for p in sys.argv[1:6])
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((real / "meta.json").read_text())
    K, dist = np.array(meta["K"]), np.array(meta["dist"])
    T_grip_cam = np.array(meta["T_gripper_from_cam_cv"])
    reg = json.loads((splat_dir / "robot_in_twin.json").read_text())
    T_desk_base = np.array(reg["T_desk_from_base"])

    model = mujoco.MjModel.from_xml_path(str(ARM_XML))
    data = mujoco.MjData(model)
    grip = model.body("gripper").id
    splat = load_splat(ckpt, splat_dir)
    extent = desk_extent(splat)
    sprites = json.loads((fingers / "index.json").read_text())
    detector = cv2.aruco.CharucoDetector(BOARD)
    report = []

    for item in meta["frames"]:
        bgr = cv2.undistort(cv2.imread(str(real / item["file"])), K, dist, None, K)
        h, w = bgr.shape[:2]
        q = np.array(item["q_settled"])
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = data.xmat[grip].reshape(3, 3), data.xpos[grip]
        T_desk_cam = T_desk_base @ T @ T_grip_cam

        plane = plane_depth(K, T_desk_cam, w, h)
        _, bg_depth, bg_alpha = render_background(splat, T_desk_cam, K, w, h)
        above = (bg_alpha > 0.5) & (bg_depth < plane - 0.008)
        sprite = min(sprites, key=lambda s: abs(s["gripper_rad"] - q[5]))
        finger = cv2.imread(str(fingers / sprite["file"]), cv2.IMREAD_UNCHANGED)[..., 3] > 0
        finger = cv2.dilate(finger.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        valid = cv2.undistort(np.full((h, w), 255, np.uint8), K, dist, None, K) > 0
        mask = on_desk(extent, K, T_desk_cam, w, h) & ~above & ~finger & valid & np.isfinite(plane)
        mask &= (plane > 0.03) & (plane < 0.6)
        depth = np.where(mask, plane, 0).astype(np.float32)

        entry = {
            "file": item["file"],
            "gt_px": int(mask.sum()),
            "cam_height_m": float(T_desk_cam[2, 3]),
        }
        # Independent check: board PnP gives the desk plane in this camera without FK.
        corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if ids is not None and len(ids) >= 8:
            obj = BOARD.getChessboardCorners()[ids.ravel()]
            ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1, 2), K, None)
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                n_pnp, d_pnp = R[:, 2], float(R[:, 2] @ tvec.ravel())
                T_cam_desk = np.linalg.inv(T_desk_cam)
                n_fk = T_cam_desk[:3, 2]
                d_fk = float(n_fk @ T_cam_desk[:3, 3])
                entry["plane_dist_pnp_m"] = abs(d_pnp)
                entry["plane_dist_fk_m"] = abs(d_fk)
                entry["plane_angle_deg"] = float(np.degrees(np.arccos(min(1.0, abs(n_pnp @ n_fk)))))
        np.savez_compressed(
            out / f"{Path(item['file']).stem}.npz",
            rgb=bgr[..., ::-1].copy(),
            depth=depth,
            mask=mask,
            f_px=np.float32(K[0, 0]),
        )
        report.append(entry)
        print(entry, flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    checks = [e for e in report if "plane_dist_pnp_m" in e]
    if checks:
        diff = np.array([e["plane_dist_fk_m"] - e["plane_dist_pnp_m"] for e in checks]) * 1000
        ang = np.array([e["plane_angle_deg"] for e in checks])
        print(
            f"board check on {len(checks)} frames: desk distance FK - PnP median {np.median(diff):+.1f} mm "
            f"(abs p90 {np.percentile(np.abs(diff), 90):.1f}), normal angle median {np.median(ang):.2f} deg"
        )
    print(f"frames with GT: {sum(e['gt_px'] > 5000 for e in report)} of {len(report)}")


if __name__ == "__main__":
    main()
