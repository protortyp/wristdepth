"""Turn pinhole BlenderProc renders into wrist-camera training frames.

    python postprocess.py RENDER_DIR OUT_DIR [--seed S]        (twin venv: cv2)

Adds the lens and sensor of a real low-cost wrist camera: barrel distortion drawn from the
census mix (10% low, 59% moderate, 31% strong fisheye), then the shared camera look
(synth/compose.py: white balance, contrast, exposure, vignetting, noise, JPEG), after
auto-exposure and census-rate motion/focus blur. Depth is
resampled with nearest neighbour so no interpolated surfaces appear at edges. Output matches
the training format: rgb, depth (m), trust (uint8), f_px.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth"))
from compose import camera_effects  # noqa: E402

BACKGROUND = 4
# Census lens classes: share, k1 range (OpenCV radial model, normalised coordinates).
LENSES = [(0.10, (-0.05, 0.0)), (0.59, (-0.25, -0.08)), (0.31, (-0.42, -0.25))]


def fill_zoom(K, dist, w, h) -> float:
    """Largest focal scale at which every border pixel's ray still lands inside the render."""
    border = np.concatenate(
        [
            np.stack([np.linspace(0, w - 1, 64), np.zeros(64)], 1),
            np.stack([np.linspace(0, w - 1, 64), np.full(64, h - 1)], 1),
            np.stack([np.zeros(48), np.linspace(0, h - 1, 48)], 1),
            np.stack([np.full(48, w - 1), np.linspace(0, h - 1, 48)], 1),
        ]
    ).astype(np.float32)[:, None]
    lo, hi = 1.0, 3.0
    for _ in range(20):
        s = (lo + hi) / 2
        Kd = K.copy()
        Kd[0, 0] *= s
        Kd[1, 1] *= s
        und = cv2.undistortPoints(border, Kd, dist, P=K).reshape(-1, 2)
        ok = (
            (und[:, 0] >= 0).all()
            and (und[:, 0] <= w - 1).all()
            and (und[:, 1] >= 0).all()
            and (und[:, 1] <= h - 1).all()
        )
        lo, hi = (lo, s) if ok else (s, hi)
    return lo if lo > 1.0 else hi


def distort(rgb, depth, seg, K, rng):
    """Remap a pinhole render to a barrel-distorted image that fills the same pixel grid.

    The distorted camera's focal length is scaled up just enough that its corners still see
    rendered pixels (a real fisheye has no empty border). The returned f_px is that camera's
    centre focal length, which is what a user would get from calibrating the lens.
    """
    h, w = depth.shape
    w_share = np.array([c[0] for c in LENSES])
    k1 = rng.uniform(*LENSES[rng.choice(len(LENSES), p=w_share / w_share.sum())][1])
    dist = np.array([k1, rng.uniform(0, 0.4) * k1**2, 0, 0, 0])
    Kd = K.copy()
    zoom = fill_zoom(K, dist, w, h)
    Kd[0, 0] *= zoom
    Kd[1, 1] *= zoom
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], 1)[:, None]
    # Each output (distorted) pixel looks along the ray of its undistorted position.
    und = cv2.undistortPoints(pts, Kd, dist, P=K).reshape(h, w, 2)
    mx, my = und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)
    inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    out_rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out_depth = cv2.remap(depth, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    out_seg = cv2.remap(
        seg, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND
    )
    out_depth[~inside] = 0
    return out_rgb, out_depth, out_seg, k1, float(Kd[0, 0])


def expose(rgb, rng):
    """Auto-exposure: real wrist cameras meter the scene, so few frames are very dark."""
    img = rgb.astype(np.float32) / 255
    luma = float(np.median(img @ np.array([0.299, 0.587, 0.114], np.float32)))
    target = rng.uniform(0.3, 0.6)
    gain = np.clip(target / max(luma, 1e-3), 0.5, 8.0)
    return np.clip(img * gain * 255, 0, 255).astype(np.uint8)


def blur(rgb, rng):
    """Motion or focus blur on about a fifth of frames, as in the census (sharp otherwise)."""
    u = rng.random()
    if u < 0.12:
        length = int(rng.integers(3, 12))
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1 / length
        rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1)
        k = cv2.warpAffine(k, rot, (length, length))
        return cv2.filter2D(rgb, -1, k / max(k.sum(), 1e-6))
    if u < 0.20:
        return cv2.GaussianBlur(rgb, (0, 0), rng.uniform(0.8, 2.0))
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("s*_*.npz"))
    for i, f in enumerate(files):
        rng = np.random.default_rng([args.seed, i])
        z = np.load(f)
        rgb, depth, seg, K = (
            z["rgb"],
            z["depth"].astype(np.float32),
            z["seg"].astype(np.uint8),
            z["K"],
        )
        if rgb.mean() < 3:  # lens blocked or unlit: nothing a camera would record usefully
            continue
        rgb, depth, seg, k1, f_px = distort(rgb, depth, seg, K, rng)
        rgb = camera_effects(blur(expose(rgb, rng), rng), rng)
        trust = np.where(depth > 0, 255, 0).astype(np.uint8)
        np.savez_compressed(
            args.out / f.name,
            rgb=rgb,
            depth=depth,
            trust=trust,
            f_px=np.float32(f_px),
            seg=seg,
            k1=np.float32(k1),
        )
        if i % 500 == 0:
            print(f"{i}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
