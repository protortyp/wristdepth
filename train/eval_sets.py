"""Evaluate checkpoints on every real wrist test set we have.

    uv run --extra export python research/wrist_depth/train/eval_sets.py [NAME=CKPT ...]

Sets:
  so101   our SO-101 wrist frames with desk-plane GT from kinematics + twin (40 frames)
  droid   SpatialBench DROID wrist frames (Franka + Robotiq, ZED Mini, S2M2 stereo depth;
          internal evaluation only, the split has no stated license)

Per set: raw metric scale (median of pred/GT per frame), AbsRel and delta1 without any
alignment, and AbsRel after per-frame median scaling (the scale-invariant number SpatialBench
reports). "base" is the released Depth Pro 2 S.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from depthpro2_onnx.model import load_model
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import predict_depth, to_input

ROOT = Path(os.environ.get("WRIST_DEPTH_DATA", "data"))
# Every n-th DROID frame; training uses a stride to keep periodic scoring cheap.
DROID_STRIDE = int(os.environ.get("DROID_STRIDE", "1"))
DROID_WIDTH = 640


def so101_frames():
    for f in sorted((ROOT / "real_gt").glob("*.npz")):
        z = np.load(f)
        yield z["rgb"], z["depth"], z["mask"] & (z["depth"] > 0), float(z["f_px"])


def spatialbench_frames(
    source: str, splits=("single", "medium"), stride: int = 1, part: str = "all"
):
    """Wrist frames of one SpatialBench source, resized to 640 px wide and cropped to % 32."""
    base = ROOT / "eval" / "spatialbench"
    scene_index = -1
    for split in splits:
        for meta_path in sorted((base / split / source).rglob("meta.json")):
            scene = meta_path.parent
            meta = json.loads(meta_path.read_text())
            if meta["tags"].get("view_type") != "wrist":
                continue
            scene_index += 1
            if part != "all" and (scene_index % 2 == 0) != (part == "val"):
                continue
            K_all = np.load(scene / "intrinsics" / "intrinsic.npy")
            for i, img in enumerate(sorted((scene / "images").glob("*"))):
                if i % stride:
                    continue
                stem = img.stem.removesuffix("_rgb")
                depth_png, depth_npy = (
                    scene / "depths" / f"{stem}_depth.png",
                    scene / "depths" / f"{stem}.npy",
                )
                if depth_npy.exists():
                    depth_full = Image.fromarray(np.load(depth_npy).astype(np.float32))
                elif depth_png.exists():
                    depth_full = Image.fromarray(
                        np.asarray(Image.open(depth_png), np.float32) / 1000
                    )
                else:
                    continue
                rgb = Image.open(img).convert("RGB")
                s = DROID_WIDTH / rgb.width
                size = (DROID_WIDTH, round(rgb.height * s))
                rgb = np.asarray(rgb.resize(size, Image.Resampling.BILINEAR))
                depth = np.asarray(depth_full.resize(size, Image.Resampling.NEAREST), np.float32)
                mask = (depth > 0.03) & (depth < 4.0)
                top = (rgb.shape[0] % 32) // 2
                rows = slice(top, top + rgb.shape[0] // 32 * 32)
                K = K_all[i] if K_all.ndim == 3 else K_all
                yield rgb[rows], depth[rows], mask[rows], float(K[0, 0]) * s


def droid_frames(part: str = "all"):
    """DROID wrist frames; part "val" or "test" splits by scene (alternating, sorted by path) so
    checkpoints are chosen on scenes that the reported numbers never see."""
    base = ROOT / "eval" / "spatialbench"
    scene_index = -1
    # The medium split streams in from a 20 GB tar; use it only once fully extracted.
    splits = ("single", "medium") if (base / "medium.done").exists() else ("single",)
    for split in splits:
        for scene in sorted((base / split / "droid").glob("*/*")):
            meta = json.loads((scene / "meta.json").read_text())
            if meta["tags"].get("view_type") != "wrist":
                continue
            scene_index += 1
            if part != "all" and (scene_index % 2 == 0) != (part == "val"):
                continue
            K_all = np.load(scene / "intrinsics" / "intrinsic.npy")
            for i, img in enumerate(sorted((scene / "images").glob("*.png"))):
                if i % DROID_STRIDE:
                    continue
                stem = img.stem
                depth_png = scene / "depths" / f"{stem}_depth.png"
                mask_png = scene / "depth_masks" / f"{stem}_depth.png"
                if not depth_png.exists():
                    continue
                rgb = Image.open(img).convert("RGB")
                w, h = rgb.size
                s = DROID_WIDTH / w
                size = (DROID_WIDTH, round(h * s))
                rgb = np.asarray(rgb.resize(size, Image.Resampling.BILINEAR))
                depth = (
                    np.asarray(
                        Image.open(depth_png).resize(size, Image.Resampling.NEAREST), np.float32
                    )
                    / 1000
                )
                mask = depth > 0.03
                if mask_png.exists():
                    mask &= (
                        np.asarray(Image.open(mask_png).resize(size, Image.Resampling.NEAREST)) > 0
                    )
                # The ViT needs sides divisible by 16; a centred crop keeps the focal length.
                top = (rgb.shape[0] % 16) // 2
                rows = slice(top, top + rgb.shape[0] // 16 * 16)
                K = K_all[i] if K_all.ndim == 3 else K_all
                yield rgb[rows], depth[rows], mask[rows], float(K[0, 0]) * s


SETS = {
    "so101": so101_frames,
    "droid_val": lambda: droid_frames("val"),
    "droid_test": lambda: droid_frames("test"),
    "robolab": lambda: spatialbench_frames("robolab", stride=DROID_STRIDE),
}


@torch.no_grad()
def evaluate(model, frames) -> dict:
    ratios, rels, d1s, rels_aligned, logs = [], [], [], [], []
    for rgb, gt_np, mask_np, f_px in frames:
        if mask_np.sum() < 500:
            continue
        gt = torch.from_numpy(gt_np).cuda()
        mask = torch.from_numpy(mask_np).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = predict_depth(model, to_input(rgb, "cuda"), f_px)[0].float()
        p, g = pred[mask], gt[mask]
        r = float((p / g).median())
        ratios.append(r)
        rels.append(float(((p - g).abs() / g).mean()))
        d1s.append(float((torch.maximum(p / g, g / p) < 1.25).float().mean()))
        rels_aligned.append(float(((p / r - g).abs() / g).mean()))
        logs.append(float((p / g).log().abs().mean()))
    return {
        "frames": len(ratios),
        "mae_log": float(np.mean(logs)),
        "scale": float(np.median(ratios)),
        "scale_p10_p90": [float(np.percentile(ratios, 10)), float(np.percentile(ratios, 90))],
        "abs_rel": float(np.mean(rels)),
        "delta1": float(np.mean(d1s)),
        "abs_rel_scale_aligned": float(np.mean(rels_aligned)),
    }


def main() -> None:
    # NAME=CKPT evaluates Depth Pro 2 S; NAME=B:CKPT a B checkpoint; base=B makes the stock
    # reference B. --json PATH writes all results.
    args = sys.argv[1:]
    out = Path(args.pop(args.index("--json") + 1)) if "--json" in args else None
    if out:
        args.remove("--json")
    base = [a for a in args if a.startswith("base=")]
    args = [a for a in args if not a.startswith("base=")]
    base_size = base[0].split("=", 1)[1] if base else "S"
    specs = [("base", f"{base_size}:")] + [tuple(a.split("=", 1)) for a in args]
    rows = {}
    for name, ckpt in specs:
        size, path = ckpt.split(":", 1) if ckpt and ckpt[:2] in ("S:", "B:") else ("S", ckpt)
        model = load_model(size, "cuda", torch.bfloat16, weights=path or None).eval()
        for set_name, frames in SETS.items():
            rows[(name, set_name)] = r = evaluate(model, frames())
            print(
                f"{name:>10} {set_name:>6}: n={r['frames']:4d} scale {r['scale']:.2f} "
                f"(p10-p90 {r['scale_p10_p90'][0]:.2f}-{r['scale_p10_p90'][1]:.2f}) "
                f"AbsRel {r['abs_rel']:.3f} d1 {r['delta1']:.2f} | aligned AbsRel {r['abs_rel_scale_aligned']:.3f}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()
    if out:
        out.write_text(json.dumps({f"{m}/{s}": r for (m, s), r in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
