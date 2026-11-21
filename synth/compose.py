"""Composite synthetic wrist frames: desk-twin splat + MuJoCo objects + real finger sprites.

    python compose.py LAYERS_DIR SPLAT_DIR CKPT FINGERS_DIR OUT_DIR [--seed 0]

Per frame (pose from LAYERS_DIR/meta.json, robot base frame):
  background  gsplat render of the desk twin (cropped to the desk volume) at the wrist pose
  depth       desk-like splat pixels -> exact desk plane (trust 1); other splat surfaces ->
              splat depth (trust 0.3); holes, floaters -> no label
  objects     MuJoCo pixels in front of the background, exact depth (trust 1); their shadow
              ratio darkens the desk
  fingers     real sprite for the nearest gripper angle, no depth label
  camera      noise, JPEG, blur, exposure and white-balance jitter
Labels only within 0.03-0.6 m. Writes OUT_DIR/<frame>.npz: rgb (uint8), depth (float16, m),
trust (uint8, 0-255), f_px.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

NEAR, FAR = 0.03, 0.60
DESK_TOL = 0.012
# Frames with less than this share of exactly labelled pixels are skipped.
MIN_LABELLED = float(__import__("os").environ.get("MIN_LABELLED", "0.6"))
ABOVE_MIN_OPACITY = float(__import__("os").environ.get("ABOVE_MIN_OPACITY", "0.05"))
ABOVE_MAX_SIZE = float(__import__("os").environ.get("ABOVE_MAX_SIZE", "0.012"))


def load_splat(ckpt: Path, splat_dir: Path):
    splats = torch.load(ckpt, map_location="cuda", weights_only=False)["splats"]
    reg = json.loads((splat_dir / "robot_in_twin.json").read_text())
    base = torch.tensor(
        np.array(reg["T_desk_from_base"])[:3, 3], dtype=torch.float32, device="cuda"
    )
    means = splats["means"]
    radial = torch.linalg.norm(means[:, :2] - base[:2], dim=1)
    keep = (radial <= 0.75) & (means[:, 2] >= -0.02) & (means[:, 2] <= 0.6)
    opacity = torch.sigmoid(splats["opacities"])
    size = torch.exp(splats["scales"]).max(1).values
    keep &= (opacity >= 0.005) & (size <= 0.05)
    # Above the desk, keep only opaque, small Gaussians (real objects). Seen from viewpoints the
    # phone never visited, faint large ones float between camera and desk: they cloud the image
    # and pull the rendered depth toward the camera, so plain desk would read as clutter.
    above = means[:, 2] > 0.012
    keep &= ~above | ((opacity >= ABOVE_MIN_OPACITY) & (size <= ABOVE_MAX_SIZE))
    out = {k: v[keep].float() for k, v in splats.items()}
    print(f"splat: {int(keep.sum()):,} of {len(keep):,} Gaussians in the desk volume")
    return out


@torch.no_grad()
def render_background(sp, T_desk_cam: np.ndarray, K: np.ndarray, w: int, h: int):
    viewmat = torch.tensor(np.linalg.inv(T_desk_cam), dtype=torch.float32, device="cuda")[None]
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    out, alpha, _ = rasterization(
        means=sp["means"],
        quats=sp["quats"],
        scales=torch.exp(sp["scales"]),
        opacities=torch.sigmoid(sp["opacities"]),
        colors=colors,
        viewmats=viewmat,
        Ks=Ks,
        width=w,
        height=h,
        sh_degree=3,
        near_plane=0.005,
        render_mode="RGB+ED",
    )
    rgb = (out[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
    return rgb, out[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


def desk_extent(sp, cell: float = 0.005, grow: float = 0.02):
    """Occupancy grid (desk frame xy) of where the splat has surface on the desk plane."""
    means = sp["means"].cpu().numpy()
    on = (np.abs(means[:, 2]) < 0.01) & (torch.sigmoid(sp["opacities"]).cpu().numpy() > 0.3)
    xy = means[on, :2]
    lo = xy.min(0) - grow
    idx = ((xy - lo) / cell).astype(int)
    grid = np.zeros(idx.max(0) + 1 + int(2 * grow / cell), np.uint8)
    grid[idx[:, 0], idx[:, 1]] = 1
    k = int(grow / cell) * 2 + 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    # A desk has no holes: fill everything not reachable from the grid border.
    padded = np.pad(grid, 1)
    outside = padded.copy()
    cv2.floodFill(outside, None, (0, 0), 2)
    grid = (outside[1:-1, 1:-1] != 2).astype(np.uint8)
    return grid.astype(bool), lo, cell


def on_desk(extent, K, T_desk_cam, w, h):
    """Pixels whose ray meets the desk plane inside the splat's desk area."""
    grid, lo, cell = extent
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1)
    world = rays @ T_desk_cam[:3, :3].T
    s = -T_desk_cam[2, 3] / world[..., 2]
    hit = T_desk_cam[:3, 3][:2] + world[..., :2] * s[..., None]
    idx = ((hit - lo) / cell).astype(int)
    ok = (s > 0) & (idx[..., 0] >= 0) & (idx[..., 1] >= 0)
    ok &= (idx[..., 0] < grid.shape[0]) & (idx[..., 1] < grid.shape[1])
    out = np.zeros((h, w), bool)
    out[ok] = grid[idx[ok][:, 0], idx[ok][:, 1]]
    return out


def plane_depth(K, T_desk_cam, w, h):
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1)
    world = rays @ T_desk_cam[:3, :3].T
    s = -T_desk_cam[2, 3] / world[..., 2]
    return np.where(s > 0, s, np.inf)


def sharpen(bgr: np.ndarray, rng) -> np.ndarray:
    """Unsharp mask for the splat background, which is softer than any real camera."""
    img = bgr.astype(np.float32)
    s, a = rng.uniform(1.0, 1.6), rng.uniform(0.3, 0.9)
    return np.clip(img + a * (img - cv2.GaussianBlur(img, (0, 0), s)), 0, 255)


def vignette(h: int, w: int, k: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / np.hypot(w / 2, h / 2)
    return 1 - k * r**4


def background_color(bgr: np.ndarray, rng) -> np.ndarray:
    """Cooler, less saturated splat colours; the renders are warmer than real wrist cameras.

    Applied to the background only: the finger sprites are already neutral white.
    """
    img = bgr.astype(np.float32)[..., ::-1] / 255  # RGB
    img = img * np.array([0.98, 1.0, 1.10])
    luma = (img @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    img = luma + rng.uniform(0.5, 0.95) * (img - luma)
    return np.clip(img * 255, 0, 255)[..., ::-1]


def camera_effects(rgb: np.ndarray, rng) -> np.ndarray:
    """Real wrist-camera look on an RGB composite; targets from research_notes/03_appearance.md.

    White-balance jitter, contrast, exposure, gamma, vignetting (1 - k r^4, k 0.15-0.45),
    sensor noise, and JPEG q45-80 on 60% of frames (real cameras are raw or AV1-compressed).
    """
    img = rgb.astype(np.float32) / 255
    h, w = img.shape[:2]
    img = img * rng.uniform(0.94, 1.06, 3)
    mean = img.mean()
    img = mean + rng.uniform(0.65, 1.1) * (img - mean)
    img = img * np.exp(rng.uniform(np.log(0.8), np.log(1.6)))
    img = np.clip(img, 0, 1) ** rng.uniform(0.85, 1.2)
    img = img * vignette(h, w, rng.uniform(0.15, 0.45))[..., None]
    img = img * 255 + rng.normal(0, rng.uniform(0.3, 1.2), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        q = int(rng.integers(45, 81))
        enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("layers", type=Path)
    ap.add_argument("splat_dir", type=Path)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("fingers", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.layers / "meta.json").read_text())
    K, w, h = np.array(meta["K"]), meta["w"], meta["h"]
    T_desk_base = np.array(meta["T_desk_from_base"])
    splat = load_splat(args.ckpt, args.splat_dir)
    extent = desk_extent(splat)
    sprites = json.loads((args.fingers / "index.json").read_text())
    sprite_imgs = {
        s["file"]: cv2.imread(str(args.fingers / s["file"]), cv2.IMREAD_UNCHANGED) for s in sprites
    }

    skipped = 0
    for item in meta["frames"]:
        i = item["frame"]
        layer = np.load(args.layers / f"{i:05d}.npz")
        T_desk_cam = T_desk_base @ np.array(item["T_base_cam_cv"])
        bg_rgb, bg_depth, bg_alpha = render_background(splat, T_desk_cam, K, w, h)
        bg_rgb = bg_rgb[..., ::-1].copy()  # BGR like the other layers in this script
        # Fill splat holes (no Gaussians) from their surroundings instead of leaving black.
        holes = (bg_alpha < 0.3).astype(np.uint8)
        if holes.any():
            bg_rgb = cv2.inpaint(bg_rgb, holes, 5, cv2.INPAINT_TELEA)
        bg_rgb = background_color(sharpen(bg_rgb, rng), rng)
        plane = plane_depth(K, T_desk_cam, w, h)

        depth = np.full((h, w), np.nan, np.float32)
        trust = np.zeros((h, w), np.float32)
        # The desk is flat and its plane is known exactly, so every pixel is desk unless the
        # splat clearly shows something above it (clutter) or beyond it (past the desk edge).
        solid = bg_alpha > 0.5
        above = solid & (bg_depth < plane - DESK_TOL)
        beyond = solid & (bg_depth > plane + 0.03)
        desk = ~above & ~beyond & np.isfinite(plane) & on_desk(extent, K, T_desk_cam, w, h)
        depth[desk], trust[desk] = plane[desk], 1.0
        depth[above], trust[above] = bg_depth[above], 0.3

        shadow = cv2.GaussianBlur(layer["shadow"].astype(np.float32) / 255, (7, 7), 0)
        rgb = bg_rgb.astype(np.float32) * shadow[..., None]
        mj_depth = layer["depth"].astype(np.float32)
        front = np.where(above, bg_depth, np.inf)  # objects stand on the desk, behind clutter
        obj = (layer["cls"] == 3) & (mj_depth < front - 0.002)
        mj_bgr = layer["rgb"][..., ::-1].astype(np.float32)
        rgb[obj] = mj_bgr[obj]
        depth[obj], trust[obj] = mj_depth[obj], 1.0

        grip = item["q"][5]
        sprite = min(sprites, key=lambda s: abs(s["gripper_rad"] - grip))
        sp = sprite_imgs[sprite["file"]]
        fmask = sp[..., 3] > 0
        # The sprites come from a dim, colour-cast night shot; real fingers are white PLA.
        fin = np.repeat(sp[..., :3].astype(np.float32).mean(-1, keepdims=True), 3, axis=-1)
        target = rng.uniform(170, 245)
        fin *= target / max(np.median(fin[fmask]), 1.0)
        rgb[fmask] = fin[fmask]
        trust[fmask] = 0.0

        # Views mostly past the desk show splat floaters and unreconstructed room, not a
        # realistic scene; keep only frames that are mostly labelled desk and objects.
        if (trust > 0.5).mean() < MIN_LABELLED:
            skipped += 1
            continue
        out_of_range = ~np.isfinite(depth) | (depth < NEAR) | (depth > FAR)
        trust[out_of_range] = 0.0
        depth[out_of_range] = 0.0
        final = camera_effects(np.clip(rgb, 0, 255).astype(np.uint8)[..., ::-1].copy(), rng)
        np.savez_compressed(
            args.out / f"{i:05d}.npz",
            rgb=final,
            depth=depth.astype(np.float16),
            trust=(trust * 255).astype(np.uint8),
            f_px=np.float32(K[0, 0]),
        )
        if i % 200 == 0 or i == meta["frames"][-1]["frame"]:
            print(
                f"{i}: desk {desk.mean():.0%} objects {obj.mean():.0%} fingers {fmask.mean():.0%} "
                f"labelled {(trust > 0).mean():.0%} skipped so far {skipped}",
                flush=True,
            )


if __name__ == "__main__":
    main()
