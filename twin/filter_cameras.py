"""Drop mis-registered cameras from the metric text model before splat training.

    python filter_cameras.py SPLAT_DIR [--max-dist 1.2] [--min-z 0.02]

A camera is kept if it is above the desk (z > min-z, desk at z = 0) and within max-dist metres of
the median camera centre. Rewrites dense/sparse_metric/images.txt in place (backup kept) and
lists the dropped frames; views out of windows are the usual culprits.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from align_board import qvec_to_rot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("splat_dir", type=Path)
    ap.add_argument("--max-dist", type=float, default=1.2)
    ap.add_argument("--min-z", type=float, default=0.02)
    args = ap.parse_args()
    path = args.splat_dir / "dense" / "sparse_metric" / "images.txt"
    backup = path.with_suffix(".unfiltered.txt")
    if not backup.exists():
        shutil.copyfile(path, backup)
    lines = [l for l in backup.read_text().splitlines() if not l.startswith("#")]
    pairs = [
        (lines[i], lines[i + 1] if i + 1 < len(lines) else "") for i in range(0, len(lines), 2)
    ]
    pairs = [p for p in pairs if p[0].strip()]
    centers = []
    for header, _ in pairs:
        f = header.split()
        R = qvec_to_rot(np.array(f[1:5], float))
        centers.append(-R.T @ np.array(f[5:8], float))
    centers = np.array(centers)
    dist = np.linalg.norm(centers - np.median(centers, 0), axis=1)
    keep = (centers[:, 2] > args.min_z) & (dist < args.max_dist)
    out = []
    for (header, points), k in zip(pairs, keep, strict=True):
        if k:
            out += [header, points]
    path.write_text("\n".join(out) + "\n")

    # Points must not reference dropped images, or COLMAP aborts on load.
    kept_ids = {p[0].split()[0] for p, k in zip(pairs, keep, strict=True) if k}
    pts = args.splat_dir / "dense" / "sparse_metric" / "points3D.txt"
    pts_backup = pts.with_suffix(".unfiltered.txt")
    if not pts_backup.exists():
        shutil.copyfile(pts, pts_backup)
    new_pts = []
    for line in pts_backup.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        track = [(f[i], f[i + 1]) for i in range(8, len(f), 2) if f[i] in kept_ids]
        if len(track) >= 2:
            new_pts.append(" ".join(f[:8] + [v for pair in track for v in pair]))
    pts.write_text("\n".join(new_pts) + "\n")
    dropped = sorted(p[0].split()[-1] for p, k in zip(pairs, keep, strict=True) if not k)
    print(
        f"kept {keep.sum()} of {len(pairs)} cameras, {len(new_pts)} points; dropped "
        f"{len(dropped)} cameras: {dropped[:6]}{' ...' if len(dropped) > 6 else ''}"
    )


if __name__ == "__main__":
    main()
