"""Cut the real gripper fingers out of the wrist jaw shots, as RGBA sprites in the pinhole frame.

    python finger_sprites.py OUT_DIR

The wrist camera is rigid to the fixed finger, so the fingers sit at the same pixels in every
frame; only the moving jaw changes with the gripper angle. The jaw shots (so101 runs/real/
jawshots, bare dark desk, seven gripper angles) are undistorted with the wrist calibration
(same K, so they match the synthetic renders) and the bright fingers are split from the dark
desk by Otsu threshold, keeping components that touch the bottom edge.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Checkout of the so101 robot project (kinematics, vendored SO-ARM100 model).
SO101 = Path(os.environ.get("SO101_ROOT", "../so101"))


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    wrist = json.loads((SO101 / "calib/real/wrist.json").read_text())
    K, dist = np.array(wrist["K"]), np.array(wrist["dist"])
    meta = json.loads((SO101 / "runs/real/jawshots/meta.json").read_text())
    index = []
    for shot in meta:
        img = cv2.undistort(cv2.imread(str(SO101 / shot["file"])), K, dist, None, K)
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        keep = np.zeros_like(mask)
        h = mask.shape[0]
        for i in range(1, n):
            x, y, w, hh, area = stats[i]
            if y + hh >= h - 3 and area > 1500:  # fingers enter from the bottom edge
                keep[labels == i] = 255
        # Undistortion leaves black corners; they are not fingers.
        valid = cv2.undistort(np.full(img.shape[:2], 255, np.uint8), K, dist, None, K) > 0
        keep[~valid] = 0
        keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
        rgba = np.dstack([img, keep])
        name = f"jaw_{shot['jaw_deg_measured']:05.1f}.png"
        cv2.imwrite(str(out / name), rgba)
        index.append(
            {"file": name, "gripper_rad": shot["q_model"][5], "pixels": int((keep > 0).sum())}
        )
        print(name, "finger px", int((keep > 0).sum()))
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
