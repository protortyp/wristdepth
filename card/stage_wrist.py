"""Assemble a local Hugging Face folder for a wrist-depth release candidate.

    uv run python research/wrist_depth/card/stage_wrist.py RC_DIR --name NAME --size B \
        --train-frames 4000 --n-frames 36464

RC_DIR holds onnx/, parity_*.json (verify_parity.py), parity_trt_fp16.json and eval.json
(eval_sets.py --json). Every number in the card is read from those files. Writes RC_DIR/hf/;
nothing is uploaded.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GITHUB = "https://github.com/protortyp/depthpro2-onnx"
MODES = {
    "model": "single",
    "model_fp16": "single",
    "model_int8": "single",
    "model_clip4": "clip of 4",
    "model_clip4_fp16": "clip of 4",
    "model_stream4": "stream, window 4",
    "model_stream4_fp16": "stream, window 4",
}
SETS = [
    ("so101", "SO-101"),
    ("droid_val", "DROID val"),
    ("droid_test", "DROID test"),
    ("robolab", "RoboLab (sim)"),
]


def results_table(ev: dict, name: str, size: str) -> str:
    rows = []
    for key, label in SETS:
        for model, model_label in (("base", f"Depth Pro 2 {size}"), ("rc", name)):
            r = ev[f"{model}/{key}"]
            bold = "**" if model == "rc" else ""
            rows.append(
                f"| {label} | {bold}{model_label}{bold} | {r['scale']:.2f} | {r['abs_rel']:.3f} | "
                f"{r['delta1']:.2f} | {r['abs_rel_scale_aligned']:.3f} |"
            )
    return "\n".join(rows)


def files_table(rc: Path) -> str:
    rows = []
    for stem, mode in MODES.items():
        onnx = rc / "onnx" / f"{stem}.onnx"
        parity = rc / f"parity_{stem}.json"
        if not onnx.exists() or not parity.exists():
            continue
        p = json.loads(parity.read_text())
        v = p["vs_fp32"]
        rows.append(
            f"| `onnx/{stem}.onnx` | {mode} | {p['config']['precision'].upper()} | "
            f"{onnx.stat().st_size / 1e6:.0f} | {v['abs_rel']:.4f} | {v['delta1']:.4f} |"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rc", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", choices=["S", "B"], required=True)
    ap.add_argument("--train-frames", type=int, required=True)
    ap.add_argument("--n-frames", type=int, required=True, help="procedural frames available")
    args = ap.parse_args()
    ev = {
        k.replace(k.split("/")[0], "rc" if not k.startswith("base/") else "base", 1): v
        for k, v in json.loads((args.rc / "eval.json").read_text()).items()
    }
    trt = json.loads((args.rc / "parity_trt_fp16.json").read_text())["vs_fp32"]["abs_rel"]
    base_ref = ROOT / "release" / f"DepthPro2-{args.size}-ONNX"

    out = args.rc / "hf"
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)
    for stem in MODES:
        src = args.rc / "onnx" / f"{stem}.onnx"
        if src.exists():
            os.link(src, out / "onnx" / src.name)
    for f in ("LICENSE", "build_tensorrt.sh", "example.py"):
        shutil.copy(base_ref / f, out / f)
    figure = args.rc / "droid_examples.png"
    if figure.exists():
        shutil.copy(figure, out / figure.name)

    rc_droid = ev["rc/droid_test"]
    base_droid = ev["base/droid_test"]
    values = {
        "name": args.name,
        "size": args.size,
        "base_model": f"apple/DepthPro2-{args.size}",
        "github": GITHUB,
        "train_frames": f"{args.train_frames:,}",
        "n_frames": f"{args.n_frames:,}",
        "base_so101_scale": f"{ev['base/so101']['scale']:.1f}",
        "base_droid_scale": f"{base_droid['scale']:.1f}",
        "so101_scale": f"{ev['rc/so101']['scale']:.2f}",
        "droid_scale": f"{rc_droid['scale']:.2f}",
        "droid_val_scale": f"{ev['rc/droid_val']['scale']:.2f}",
        "gain": f"{base_droid['abs_rel'] / rc_droid['abs_rel']:.0f}",
        "results_table": results_table(ev, args.name, args.size),
        "files_table": files_table(args.rc),
        "trt_abs_rel": f"{trt:.4f}",
    }
    template = (Path(__file__).parent / "MODEL_CARD_DRAFT.md").read_text()
    card = re.sub(r"{{(\w+)}}", lambda m: values[m.group(1)], template)
    (out / "README.md").write_text(card)

    config = json.loads((base_ref / "config.json").read_text())
    config["base_model"] = f"apple/DepthPro2-{args.size}"
    config["fine_tuned"] = {
        "task": "metric depth for robot wrist cameras",
        "data": "procedural BlenderProc scenes (CC0 assets, Apache-2.0 SO-ARM100 CAD) "
        "+ self-distillation on unlabelled community wrist frames",
        "evaluation": {k.split("/", 1)[1]: v for k, v in ev.items() if k.startswith("rc/")},
    }
    config["variants"] = {k: v for k, v in config.get("variants", {}).items() if (out / k).exists()}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"staged {out}: {len(list((out / 'onnx').iterdir()))} graphs, "
        f"README {len(card.split())} words"
    )


if __name__ == "__main__":
    main()
