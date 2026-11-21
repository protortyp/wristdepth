"""Convert procedural renders into a Hugging Face dataset: Parquet shards with Image columns.

    uv run --with datasets --with pillow python build_hf.py OUT --raw DIR... --wrist DIR...

Two configs share frame ids:
  pinhole  clean BlenderProc render, exact pinhole intrinsics
  wrist    the same frame through a wrist-camera post-process (fisheye or pinhole, exposure,
           blur, noise, JPEG); f_px is the centre focal length

Depth is a 16-bit PNG in units of 0.1 mm (meters = value / 10000, 0 = no label), which the
Image feature decodes losslessly. Splits are by scene, so frames of one table never straddle
train and test.
"""

import argparse
import io
import json
import zlib
from pathlib import Path

import datasets
import numpy as np
from PIL import Image

DEPTH_SCALE = 10000  # PNG value per meter
SHARD_FRAMES = 500


def png(array: np.ndarray) -> dict:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", compress_level=6)
    return {"bytes": buf.getvalue(), "path": None}


def jpeg(array: np.ndarray, quality: int = 95) -> dict:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="JPEG", quality=quality)
    return {"bytes": buf.getvalue(), "path": None}


# Class ids as a palette PNG: values stay 0-4 for code, viewers show colours. 0 no surface
# (environment map), 1 gripper, 2 table and flat items, 3 objects, 4 room.
SEG_PALETTE = [0, 0, 0, 230, 120, 30, 150, 150, 150, 40, 110, 220, 40, 90, 50]


def seg_png(ids: np.ndarray) -> dict:
    img = Image.fromarray(ids.astype(np.uint8), mode="P")
    img.putpalette(SEG_PALETTE + [0] * (768 - len(SEG_PALETTE)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return {"bytes": buf.getvalue(), "path": None}


def depth_png(depth: np.ndarray) -> dict:
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    return png(np.clip(np.round(d * DEPTH_SCALE), 0, 65535).astype(np.uint16))


def split_of(scene: str) -> str:
    # Stable across runs and machines: 2% validation, 2% test by scene.
    h = zlib.crc32(scene.encode()) % 100
    return "validation" if h < 2 else "test" if h < 4 else "train"


def generator_version(seed: int) -> str:
    if seed >= 800:
        return "v5"  # cameras kept inside walls, with more varied dishware
    if seed >= 700:
        return "v4"  # distance bands per scene, wide FOV, jaw-size and gripper-free variation
    if seed >= 600:
        return "v3-16spp"
    if seed >= 500:
        return "v3"  # flat prints and mats
    return "v2" if seed >= 200 else "v1"


def common(name: str, depth: np.ndarray, seg: np.ndarray, meta: dict) -> dict:
    seed, scene, frame = name[:-4].split("_")
    valid = depth[depth > 0]
    T = meta.get("T_world_cam_cv")
    return {
        "frame_id": name[:-4],
        "scene_id": f"{seed}_{scene}",
        "frame_index": int(frame),
        "generator_version": generator_version(int(seed[1:])),
        "depth": depth_png(depth),
        "segmentation": seg_png(seg),
        "camera_to_world": [float(x) for x in np.array(T).ravel()] if T is not None else None,
        "distance_m": float(meta.get("dist_m", np.nan)),
        "gripper_visible": bool(meta.get("gripper", True)),
        "gripper_scale": float(meta.get("gripper_scale", 1.0)),
        "jaw_command_rad": float(meta.get("jaw_cmd", np.nan)),
        "depth_median_m": float(np.median(valid)) if valid.size else 0.0,
        "valid_ratio": float((depth > 0).mean()),
    }


FEATURES_COMMON = {
    "frame_id": datasets.Value("string"),
    "scene_id": datasets.Value("string"),
    "frame_index": datasets.Value("int32"),
    "generator_version": datasets.Value("string"),
    "image": datasets.Image(),
    "depth": datasets.Image(),
    "segmentation": datasets.Image(),
    "fx": datasets.Value("float32"),
    "fy": datasets.Value("float32"),
    "cx": datasets.Value("float32"),
    "cy": datasets.Value("float32"),
    "camera_model": datasets.Value("string"),
    "k1": datasets.Value("float32"),
    "camera_to_world": datasets.Sequence(datasets.Value("float32"), length=16),
    "distance_m": datasets.Value("float32"),
    "gripper_visible": datasets.Value("bool"),
    "gripper_scale": datasets.Value("float32"),
    "jaw_command_rad": datasets.Value("float32"),
    "depth_median_m": datasets.Value("float32"),
    "valid_ratio": datasets.Value("float32"),
}


def pinhole_row(path: Path) -> dict:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    K = z["K"]
    row = common(path.name, z["depth"].astype(np.float32), z["seg"], meta)
    row.update(
        image=png(z["rgb"]),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        camera_model="pinhole",
        k1=0.0,
    )
    return row


def wrist_row(path: Path, raw: Path) -> dict:
    z, r = np.load(path), np.load(raw)
    meta = json.loads(str(r["meta"]))
    K = r["K"]
    k1 = float(z["k1"]) if "k1" in z.files else 0.0
    row = common(path.name, z["depth"].astype(np.float32), z["seg"], meta)
    f = float(z["f_px"])
    row.update(
        image=jpeg(z["rgb"]),
        fx=f,
        fy=f,
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        camera_model="radial" if k1 != 0 else "pinhole",
        k1=k1,
    )
    return row


def write(rows_iter, features, out: Path, config: str, worker: int) -> dict:
    buffers, counts = {}, {}

    def flush(split):
        rows = buffers.pop(split)
        i = counts.get(split + "_shards", 0)
        ds = datasets.Dataset.from_list(rows, features=features)
        target = out / config / f"{split}-w{worker:02d}-{i:05d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(target))
        counts[split + "_shards"] = i + 1

    for row in rows_iter:
        split = split_of(row["scene_id"])
        buffers.setdefault(split, []).append(row)
        counts[split] = counts.get(split, 0) + 1
        if len(buffers[split]) >= SHARD_FRAMES:
            flush(split)
    for split in list(buffers):
        flush(split)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--raw", type=Path, nargs="+", required=True)
    ap.add_argument("--wrist", type=Path, nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=["pinhole", "wrist"])
    ap.add_argument("--worker", type=int, default=0, help="this process's index")
    ap.add_argument("--workers", type=int, default=1, help="processes splitting the scenes")
    args = ap.parse_args()
    datasets.disable_progress_bars()
    raw = {p.name: p for d in args.raw for p in d.glob("s*_*.npz")}
    wrist = {p.name: p for d in args.wrist for p in d.glob("s*_*.npz")}
    names = sorted(set(raw) & set(wrist))
    # Workers split by scene, so each scene's frames stay in one worker's shards.
    names = [
        n for n in names if zlib.crc32(n.rsplit("_", 1)[0].encode()) % args.workers == args.worker
    ]
    print(
        f"{len(names)} frames with both versions ({len(raw)} raw, {len(wrist)} wrist)", flush=True
    )
    features = datasets.Features(FEATURES_COMMON)
    stats = {}
    if "pinhole" in args.configs:
        rows = (pinhole_row(raw[n]) for n in names)
        stats["pinhole"] = write(rows, features, args.out, "pinhole", args.worker)
        print("pinhole", stats["pinhole"], flush=True)
    if "wrist" in args.configs:
        rows = (wrist_row(wrist[n], raw[n]) for n in names)
        stats["wrist"] = write(rows, features, args.out, "wrist", args.worker)
        print("wrist", stats["wrist"], flush=True)
    (args.out / f"build_stats_w{args.worker:02d}.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
