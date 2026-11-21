"""Export the desk-twin's desk area as an occupancy map in the robot base frame.

    python make_desk_map.py SPLAT_DIR CKPT OUT.npz       (twin venv)

The pose sampler uses it to keep wrist views on the real desk, where the twin has surface.
"""

import json
import sys
from pathlib import Path

import numpy as np
from compose import desk_extent, load_splat


def main() -> None:
    splat_dir, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    grid, lo, cell = desk_extent(load_splat(ckpt, splat_dir))
    reg = json.loads((splat_dir / "robot_in_twin.json").read_text())
    np.savez(
        out, grid=grid, lo_desk=lo, cell=cell, T_desk_from_base=np.array(reg["T_desk_from_base"])
    )
    print(f"desk map {grid.shape} cells of {cell * 1000:.0f} mm, {grid.mean():.0%} desk")


if __name__ == "__main__":
    main()
