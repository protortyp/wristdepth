"""Where does scale error come from: per-depth-bin pred/gt and per-frame scale vs depth and f_px.

    uv run --extra export python depth_bins.py NAME=B:CKPT ... [--sets droid_val so101]

A flat pred/gt across depth bins means a global offset (focal length or scale prior); a ratio
that falls with depth is range compression. Frame-level correlations separate the two causes:
log(frame scale) against log(median GT depth) and against log(f_px).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import predict_depth, to_input
from depthpro2_onnx.model import load_model
from eval_sets import ROOT, SETS

UNSEEN_SYNTH = ROOT / "procgen_pp_v4_tail"


def synth_frames(stride: int = 5):
    """v4 renders no run trained on (the seed-701 scenes rendered after the ablation)."""
    for i, f in enumerate(sorted(UNSEEN_SYNTH.glob("*.npz"))):
        if i % stride:
            continue
        z = np.load(f)
        d = z["depth"].astype(np.float32)
        yield z["rgb"], d, (d > 0.03) & (d < 6.0), float(z["f_px"])


SETS = {**SETS, "synth_v4": synth_frames}
BINS = [0.0, 0.15, 0.3, 0.6, 1.2, 100.0]


@torch.no_grad()
def analyse(model, frames):
    bin_logs = [[] for _ in BINS[:-1]]
    frame_scale, frame_depth, frame_f, inner = [], [], [], []
    for rgb, gt_np, mask_np, f_px in frames:
        if mask_np.sum() < 500:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = predict_depth(model, to_input(rgb, "cuda"), f_px)[0].float().cpu().numpy()
        p, g = pred[mask_np], gt_np[mask_np]
        lr = np.log(p / g)
        for b in range(len(BINS) - 1):
            sel = (g >= BINS[b]) & (g < BINS[b + 1])
            if sel.sum() >= 50:
                bin_logs[b].append(float(np.median(lr[sel])))
        frame_scale.append(float(np.median(lr)))
        lg = np.log(g)
        if lg.std() > 0.1:  # slope of log pred on log GT within one image; 1 = no compression
            inner.append(float(np.polyfit(lg, np.log(p), 1)[0]))
        frame_depth.append(float(np.log(np.median(g))))
        frame_f.append(float(np.log(f_px / rgb.shape[1])))
    s, d, f = map(np.array, (frame_scale, frame_depth, frame_f))
    corr = lambda a, b: (
        float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    )
    return {
        "bins": [
            (f"{BINS[b]:.2f}-{BINS[b + 1]:.2f}", len(v), float(np.exp(np.median(v))) if v else None)
            for b, v in enumerate(bin_logs)
        ],
        "slope_scale_vs_depth": float(np.polyfit(d, s, 1)[0]),
        "corr_scale_depth": corr(s, d),
        "corr_scale_focal": corr(s, f),
        "frame_scale_log_std": float(s.std()),
        "within_frame_slope": float(np.median(inner)) if inner else float("nan"),
    }


def main() -> None:
    args = sys.argv[1:]
    sets = ["droid_val", "droid_test", "so101", "robolab"]
    if "--sets" in args:
        i = args.index("--sets")
        sets, args = args[i + 1 :], args[:i]
    for spec in args:
        name, ckpt = spec.split("=", 1)
        size, path = ckpt.split(":", 1)
        model = load_model(size, "cuda", torch.bfloat16, weights=path or None).eval()
        for set_name in sets:
            r = analyse(model, SETS[set_name]())
            bins = "  ".join(
                f"{k}m n={n} {v:.2f}x" if v else f"{k}m -"
                for k, v, n in [(b[0], b[2], b[1]) for b in r["bins"]]
            )
            print(
                f"{name:>8} {set_name:>10}: {bins} | slope {r['slope_scale_vs_depth']:+.2f} "
                f"corr(depth) {r['corr_scale_depth']:+.2f} corr(f) {r['corr_scale_focal']:+.2f} "
                f"log-std {r['frame_scale_log_std']:.2f} in-frame slope {r['within_frame_slope']:.2f}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
