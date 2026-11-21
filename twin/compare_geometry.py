"""Depth accuracy of trained splats on the ChArUco board, whose plane (z = 0) is known exactly.

    python compare_geometry.py SPLAT_DIR NAME=CKPT [NAME=CKPT ...]

Renders depth from every training camera that sees enough of the board (inset by 1 cm) and compares
it per pixel with the analytic depth of the board plane. 3DGS checkpoints render expected depth;
2DGS checkpoints (they carry "features") render median depth, the usual 2DGS surface estimate.
Reports absolute error in mm, overall and by camera distance to the board.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from align_board import read_model
from gsplat import rasterization, rasterization_2dgs

SCALE = 0.5
INSET = 0.01


def plane_depth(K, R, t, w, h):
    """z-depth of the desk plane z = 0 at every pixel (inf where the ray misses it)."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays_cam = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1)
    rays_world = rays_cam @ R  # R^T d for each pixel, as row vectors
    center = -R.T @ t
    s = -center[2] / rays_world[..., 2]
    return np.where(s > 0, s, np.inf)


def board_mask(K, R, t, w, h, center, size):
    hx, hy = size[0] / 2 - INSET, size[1] / 2 - INSET
    corners = np.array(
        [
            [center[0] + sx * hx, center[1] + sy * hy, 0.0]
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )
    cam = (R @ corners.T).T + t
    if (cam[:, 2] <= 0.02).any():
        return None
    px = (K @ cam.T).T
    px = px[:, :2] / px[:, 2:]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [px.round().astype(np.int32)], 1)
    return mask.astype(bool)


@torch.no_grad()
def render_depth(splats, is_2dgs, K, R, t, w, h):
    viewmat = torch.eye(4, device="cuda")
    viewmat[:3, :3], viewmat[:3, 3] = torch.tensor(R), torch.tensor(t)
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    args = dict(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        viewmats=viewmat[None].float(),
        Ks=Ks,
        width=w,
        height=h,
        near_plane=0.005,
    )
    if is_2dgs:
        colors = torch.zeros(len(splats["means"]), 3, device="cuda")
        out = rasterization_2dgs(colors=colors[None], render_mode="RGB+D", **args)
        return out[5][0, ..., 0].cpu().numpy()  # median depth
    colors = torch.zeros(len(splats["means"]), 3, device="cuda")
    out, _, _ = rasterization(colors=colors, render_mode="ED", **args)
    return out[0, ..., 0].cpu().numpy()


def main() -> None:
    root = Path(sys.argv[1])
    sim3 = json.loads((root / "board_sim3.json").read_text())
    center, size = sim3["board_center"], (9 * 0.028, 6 * 0.028)
    images = read_model(root / "dense" / "sparse_metric")
    first = next(iter(images.values()))[0]
    w0, h0 = int(round(first[0, 2] * 2)), int(round(first[1, 2] * 2))
    w, h = int(w0 * SCALE), int(h0 * SCALE)

    views = []
    for name, (K, R, t) in sorted(images.items()):
        Ks = K.copy()
        Ks[:2] *= SCALE
        mask = board_mask(Ks, R, t, w, h, center, size)
        if mask is None or mask.sum() < 0.05 * w * h:
            continue
        dist = float(np.linalg.norm(-R.T @ t - np.array([*center[:2], 0.0])))
        views.append((name, Ks, R, t, mask, dist))
    print(
        f"{len(views)} cameras see at least 5% board pixels (distance "
        f"{min(v[5] for v in views):.2f}-{max(v[5] for v in views):.2f} m)"
    )

    for spec in sys.argv[2:]:
        label, path = spec.split("=", 1)
        ckpt = torch.load(path, map_location="cuda", weights_only=False)
        splats = {k: v.float() for k, v in ckpt["splats"].items()}
        is_2dgs = "features" in splats
        errs, near, far, mid = [], [], [], []
        for name, K, R, t, mask, dist in views:
            depth = render_depth(splats, is_2dgs, K, R, t, w, h)
            truth = plane_depth(K, R, t, w, h)
            e = np.abs(depth - truth)[mask] * 1000
            errs.append(e)
            (near if dist < 0.35 else far).append(e)
            mid.append(e) if 0.35 <= dist < 0.6 else None
        allerr = np.concatenate(errs)
        line = f"{label:<6} median {np.median(allerr):5.2f} mm, p90 {np.percentile(allerr, 90):5.2f} mm"
        for tag, group in (("<0.35 m", near), ("0.35-0.6 m", mid), (">=0.35 m", far)):
            if group:
                g = np.concatenate(group)
                line += f" | {tag}: median {np.median(g):5.2f}, p90 {np.percentile(g, 90):5.2f}"
        print(line)


if __name__ == "__main__":
    main()
