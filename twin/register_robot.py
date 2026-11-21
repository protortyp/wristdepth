"""Locate the robot base in the desk twin from the ArUco markers on its base plate.

    python register_robot.py SPLAT_DIR SO101_ROOT

Triangulates the base-plate marker corners from the metric COLMAP model (dense/sparse_metric),
fits base frame -> desk frame (rotation, translation, and a scale that should be 1), and writes
SPLAT_DIR/robot_in_twin.json with the base pose plus the calibrated overhead camera and the
recorded wrist pose from so101's experiments/depthpro2, as desk-frame OpenCV camera poses.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from align_board import read_model, triangulate, umeyama


def main() -> None:
    root, so101 = Path(sys.argv[1]), Path(sys.argv[2])
    design = json.loads((so101 / "hardware/marker_base/markers.json").read_text())
    known = {m["id"]: np.array(m["corners_m"]) for m in design["markers"]}
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, design["dictionary"])), params
    )
    images = read_model(root / "dense" / "sparse_metric")
    obs: dict[tuple[int, int], list] = {}
    for name, (K, R, t) in images.items():
        gray = cv2.imread(str(root / "dense" / "images" / name), cv2.IMREAD_GRAYSCALE)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for mid, c in zip(ids.ravel(), corners, strict=True):
            if int(mid) not in known:
                continue
            for k, uv in enumerate(c.reshape(4, 2)):
                obs.setdefault((int(mid), k), []).append((K, R, t, uv))

    src, dst = [], []
    for (mid, k), o in obs.items():
        if len(o) < 3:
            continue
        X = triangulate(o)
        errs = [
            np.linalg.norm((K @ (R @ X + t))[:2] / (K @ (R @ X + t))[2] - uv) for K, R, t, uv in o
        ]
        if np.median(errs) > 2.0:
            continue
        src.append(known[mid][k])
        dst.append(X)
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1) * 1000
    # Rigid fit for the pose itself; the scale above only checks the twin's metric scale.
    s1, R1, t1 = umeyama(src, dst)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rr = U @ D @ Vt
    tr = mu_d - Rr @ mu_s
    rigid_resid = np.linalg.norm((Rr @ src.T).T + tr - dst, axis=1) * 1000
    print(
        f"marker corners {len(src)}, similarity scale {s:.4f} (1 = twin scale matches the "
        f"plate design), residual median {np.median(resid):.2f} mm; rigid residual median "
        f"{np.median(rigid_resid):.2f} mm, max {rigid_resid.max():.2f} mm"
    )

    T_desk_base = np.eye(4)
    T_desk_base[:3, :3], T_desk_base[:3, 3] = Rr, tr
    cams = {}
    over = json.loads((so101 / "calib/real/overhead.json").read_text())
    cams["overhead"] = {
        "T_desk_from_cam_cv": (T_desk_base @ np.array(over["T_base_from_cam_cv"])).tolist(),
        "K_640x480": over["K"],
        "note": "1920x1080 mode = K scaled by 2.25 with cx + 240",
    }
    pose_file = so101 / "experiments/depthpro2/img/wrist_pose.json"
    if pose_file.exists():
        wp = json.loads(pose_file.read_text())
        cams["wrist_recorded"] = {
            "T_desk_from_cam_cv": (T_desk_base @ np.array(wp["T_base_from_wrist_cv"])).tolist(),
            "K_640x480": wp["K"],
            "q": wp["q"],
        }
    (root / "robot_in_twin.json").write_text(
        json.dumps(
            {
                "T_desk_from_base": T_desk_base.tolist(),
                "marker_corners": len(src),
                "scale_check": s,
                "rigid_residual_mm_median": float(np.median(rigid_resid)),
                "cameras": cams,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
