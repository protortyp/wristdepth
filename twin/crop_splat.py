"""Crop a 3DGS PLY to the desk volume and drop typical floaters.

    python crop_splat.py IN.ply OUT.ply SPLAT_DIR [--radius 0.75] [--zmin -0.02] [--zmax 0.6]

Coordinates are the board/desk frame (metres, desk z = 0). Keeps Gaussians within `radius` of
the robot base in x/y, between zmin and zmax, with opacity >= 0.005 and largest scale <= 5 cm.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def read_ply(path: Path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode()
            header.append(line)
            if line.strip() == "end_header":
                break
        count = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
        names = [l.split()[-1] for l in header if l.startswith("property float")]
        data = np.fromfile(f, dtype=np.float32, count=count * len(names)).reshape(count, len(names))
    return header, names, data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("splat_dir", type=Path)
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--zmin", type=float, default=-0.02)
    ap.add_argument("--zmax", type=float, default=0.6)
    args = ap.parse_args()

    header, names, data = read_ply(args.src)
    col = {n: i for i, n in enumerate(names)}
    xyz = data[:, [col["x"], col["y"], col["z"]]]
    base = np.array(
        json.loads((args.splat_dir / "robot_in_twin.json").read_text())["T_desk_from_base"]
    )[:3, 3]
    radial = np.linalg.norm(xyz[:, :2] - base[:2], axis=1)
    opacity = 1 / (1 + np.exp(-data[:, col["opacity"]]))
    scale = np.exp(data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1)
    keep = (radial <= args.radius) & (xyz[:, 2] >= args.zmin) & (xyz[:, 2] <= args.zmax)
    # MCMC (opacity_reg) leaves many semi-transparent Gaussians that together form surfaces;
    # only near-invisible ones are floaters.
    keep &= (opacity >= 0.005) & (scale <= 0.05)
    out = data[keep]
    header = [
        f"element vertex {len(out)}\n" if l.startswith("element vertex") else l for l in header
    ]
    with open(args.dst, "wb") as f:
        f.write("".join(header).encode())
        out.astype(np.float32).tofile(f)
    print(f"kept {len(out):,} of {len(data):,} Gaussians ({100 * len(out) / len(data):.0f}%)")


if __name__ == "__main__":
    main()
