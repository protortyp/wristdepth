"""Pick the sharpest frame per window from extracted video frames and detect the ChArUco board.

    python select_frames.py RAW_DIR OUT_DIR [--window 2]

Writes the kept frames to OUT_DIR and OUT_DIR/../charuco.json (corner ids and pixels per frame).
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BOARD = cv2.aruco.CharucoBoard(
    (9, 6), 0.028, 0.021, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
)


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.resize(gray, None, fx=0.5, fy=0.5), cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    detector = cv2.aruco.CharucoDetector(BOARD)

    clips = defaultdict(list)
    for path in sorted(args.raw.glob("*.jpg")):
        clips[path.stem.rsplit("_", 1)[0]].append(path)

    kept, charuco = [], {}
    for clip, paths in clips.items():
        scores = [sharpness(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]
        for start in range(0, len(paths), args.window):
            window = range(start, min(start + args.window, len(paths)))
            best = max(window, key=lambda i: scores[i])
            # Drop windows whose best frame is still much blurrier than its neighbourhood.
            local = np.median(scores[max(0, start - 15) : start + 15])
            if scores[best] < 0.35 * local:
                continue
            kept.append(paths[best])
    for path in kept:
        shutil.copy2(path, args.out / path.name)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is not None and len(ids) >= 6:
            charuco[path.name] = {
                "ids": ids.ravel().tolist(),
                "px": corners.reshape(-1, 2).round(2).tolist(),
            }
    (args.out.parent / "charuco.json").write_text(json.dumps(charuco))
    print(
        f"kept {len(kept)} of {sum(len(p) for p in clips.values())} frames; "
        f"board (>=6 corners) in {len(charuco)}"
    )


if __name__ == "__main__":
    main()
