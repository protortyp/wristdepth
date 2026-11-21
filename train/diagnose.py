"""Per-frame depth ratio (prediction / GT) against distance, for several checkpoints.

    uv run --extra export python research/wrist_depth/train/diagnose.py REAL_GT_DIR [NAME=CKPT ...]

A ratio that is the same at every distance points to something global (intrinsics, appearance,
a learned overall scale); one that changes with distance points to learned near/far priors.
Also splits each frame into its near and far half of desk pixels.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from depthpro2_onnx.model import load_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import predict_depth, to_input  # noqa: E402


@torch.no_grad()
def main() -> None:
    real = sorted(Path(sys.argv[1]).glob("*.npz"))
    specs = [("base", None)] + [tuple(s.split("=", 1)) for s in sys.argv[2:]]
    rows = {}
    for name, ckpt in specs:
        model = load_model("S", "cuda", torch.bfloat16).eval()
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        for f in real:
            z = np.load(f)
            gt = torch.from_numpy(z["depth"]).cuda()
            mask = torch.from_numpy(z["mask"]).cuda() & (gt > 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = predict_depth(model, to_input(z["rgb"], "cuda"), float(z["f_px"]))[0]
            g, p = gt[mask], pred[mask]
            med = g.median()
            near, far = g < med, g >= med
            rows.setdefault(f.stem, {"gt_median": float(med)})[name] = (
                float((p / g).median()),
                float((p[near] / g[near]).median()),
                float((p[far] / g[far]).median()),
            )
    names = [n for n, _ in specs]
    print(f"{'frame':<10} {'GT m':>5} " + " ".join(f"{n + ' all/near/far':>22}" for n in names))
    for frame, r in sorted(rows.items(), key=lambda kv: kv[1]["gt_median"]):
        cells = " ".join(f"{r[n][0]:6.2f}/{r[n][1]:5.2f}/{r[n][2]:5.2f}   " for n in names)
        print(f"{frame:<10} {r['gt_median']:5.2f} {cells}")
    for n in names:
        gtm = np.array([r["gt_median"] for r in rows.values()])
        ratio = np.array([r[n][0] for r in rows.values()])
        slope = np.polyfit(gtm, ratio, 1)[0]
        print(
            f"{n}: median ratio {np.median(ratio):.2f}, spread p10-p90 "
            f"{np.percentile(ratio, 10):.2f}-{np.percentile(ratio, 90):.2f}, "
            f"slope vs distance {slope:+.2f} per m"
        )


if __name__ == "__main__":
    main()
