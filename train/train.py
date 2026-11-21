"""Fine-tune Depth Pro 2 S on synthetic wrist frames; evaluate on real wrist frames each epoch.

    uv run --extra export python research/wrist_depth/train/train.py \
        --data $WRIST_DEPTH_DATA/procgen_pp --distill $WRIST_DEPTH_DATA/community/frames \
        --out $WRIST_DEPTH_DATA/runs/example --size B --stage b --encoder-blocks 4 --ema 0.995

Stage a: decoder and head only (encoder frozen). Stage b: also the encoder at 1/10 the LR.
Loss on labelled pixels, weighted by trust: |log d - log d*| + 0.5 * gradient matching in log
depth. DepthPro2.forward treats a batch as one clip, so frames go one at a time with gradient
accumulation. Depth = f_px / (W * inverse_depth), as at inference.

With --distill, unlabelled real images are mixed in: the frozen released model is the teacher
and the loss is scale-invariant, so the student keeps the teacher's depth shape on real
cameras while the synthetic frames set the metric scale. Checkpoints are scored on every
real test set in eval_sets.py, not on synthetic validation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from depthpro2_onnx.model import load_model
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import predict_depth, to_input
from eval_sets import SETS
from eval_sets import evaluate as evaluate_real


def crop_resize(rgb, depth, trust, f_px, rng, min_scale=0.6):
    """Random same-aspect crop resized back to full size; depth is unchanged, focal scales.

    Shows the network one scene at several apparent distances, so it cannot read metric depth
    from image-space size alone.
    """
    h, w = depth.shape
    s = rng.uniform(min_scale, 1.0)
    ch, cw = int(h * s), int(w * s)
    y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    rgb = np.asarray(
        Image.fromarray(np.ascontiguousarray(rgb[y : y + ch, x : x + cw])).resize(
            (w, h), Image.Resampling.BILINEAR
        )
    )
    # Nearest-neighbour for labels: interpolating depth across edges invents surfaces.
    rows = y + (np.arange(h) * ch // h)
    cols = x + (np.arange(w) * cw // w)
    depth = depth[np.ix_(rows, cols)]
    trust = trust[np.ix_(rows, cols)]
    # The principal point moves with the crop, but depth = f_px / (W * inv) only needs fx.
    return rgb, depth, trust, f_px * w / cw


def load_unlabelled(path: Path, max_width: int = 640) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        img = img.resize(
            (max_width, round(img.height * max_width / img.width)), Image.Resampling.BILINEAR
        )
    rgb = np.asarray(img)
    # Encoder patches are 16 px and the decoder halves resolution once more: sides % 32.
    h, w = rgb.shape[0] // 32 * 32, rgb.shape[1] // 32 * 32
    top, left = (rgb.shape[0] - h) // 2, (rgb.shape[1] - w) // 2
    return np.ascontiguousarray(rgb[top : top + h, left : left + w])


def distill_loss(student, teacher, kind: str = "global"):
    """Scale-invariant log-depth loss against the frozen base model.

    global: variance of the log ratio plus its gradients; it also copies the base model's
    near-to-far range, which is compressed on wrist views.
    grad: multi-scale gradients of the log ratio only, so edges and surfaces are kept while the
    overall depth range is left to the labelled loss.
    """
    d = torch.log(student.clamp_min(1e-4)) - torch.log(teacher.clamp_min(1e-4))
    if kind == "global":
        si = (d - d.mean()).pow(2).mean()
        gx = (d[..., :, 1:] - d[..., :, :-1]).abs().mean()
        gy = (d[..., 1:, :] - d[..., :-1, :]).abs().mean()
        return si + 0.5 * (gx + gy)
    # The base model's sky and far background (clamped inverse depth) carry no shape signal.
    m = ((teacher > 0.03) & (teacher < 10.0)).float()
    total = 0.0
    for s in (1, 2, 4, 8):
        ds = torch.nn.functional.avg_pool2d((d * m)[:, None], s)[:, 0] if s > 1 else d * m
        ms = torch.nn.functional.avg_pool2d(m[:, None], s)[:, 0] if s > 1 else m
        ds = ds / ms.clamp_min(1e-6)
        mx = (ms[..., :, 1:] > 0.99) & (ms[..., :, :-1] > 0.99)
        my = (ms[..., 1:, :] > 0.99) & (ms[..., :-1, :] > 0.99)
        gx = (ds[..., :, 1:] - ds[..., :, :-1]).abs()[mx]
        gy = (ds[..., 1:, :] - ds[..., :-1, :]).abs()[my]
        total = total + (gx.sum() + gy.sum()) / (mx.sum() + my.sum()).clamp_min(1)
    return total


# Labels outside this range are near-clip gripper slivers or far background; their log error
# dominates both gradients and mean metrics.
DEPTH_MIN, DEPTH_MAX = 0.03, 6.0


def loss_fn(pred, gt, w):
    w = w * ((gt > DEPTH_MIN) & (gt < DEPTH_MAX))
    lp, lg = torch.log(pred.clamp_min(1e-4)), torch.log(gt.clamp_min(1e-4))
    diff = (lp - lg) * (w > 0)
    l1 = (diff.abs() * w).sum() / w.sum().clamp_min(1)
    # Gradient matching on log depth: sharp edges without fixing the absolute scale twice.
    m = (w > 0.5).float()
    gx = (diff[..., :, 1:] - diff[..., :, :-1]).abs() * m[..., :, 1:] * m[..., :, :-1]
    gy = (diff[..., 1:, :] - diff[..., :-1, :]).abs() * m[..., 1:, :] * m[..., :-1, :]
    pairs = (m[..., :, 1:] * m[..., :, :-1]).sum() + (m[..., 1:, :] * m[..., :-1, :]).sum()
    grad = (gx.sum() + gy.sum()) / pairs.clamp_min(1)
    return l1 + 0.5 * grad


@torch.no_grad()
def evaluate(model, files, device, key_depth="depth", key_mask=None) -> dict:
    rels, ratios, d1, logs = [], [], [], []
    for f in files:
        z = np.load(f)
        gt = torch.from_numpy(z[key_depth].astype(np.float32)).to(device)
        mask = (
            torch.from_numpy(z[key_mask]).to(device)
            if key_mask
            else (torch.from_numpy(z["trust"]).to(device) > 200)
        )
        mask &= (gt > DEPTH_MIN) & (gt < DEPTH_MAX)
        if mask.sum() < 1000:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = predict_depth(model, to_input(z["rgb"], device), float(z["f_px"]))[0]
        p, g = pred[mask], gt[mask]
        r = p / g
        rels.append(((p - g).abs() / g).mean().item())
        ratios.append(r.median().item())
        d1.append((torch.maximum(r, 1 / r) < 1.25).float().mean().item())
        logs.append(r.float().log().abs().mean().item())
    return {
        "mae_log": float(np.mean(logs)),
        "abs_rel": float(np.mean(rels)),
        "scale_median": float(np.median(ratios)),
        "delta1": float(np.mean(d1)),
        "frames": len(rels),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--distill", type=Path, help="folder of unlabelled real images (jpg/png)")
    ap.add_argument(
        "--distill-ratio", type=float, default=0.5, help="unlabelled images per synthetic frame"
    )
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--distill-loss", choices=["global", "grad"], default="global")
    ap.add_argument(
        "--bf16-weights",
        action="store_true",
        help="keep trainable weights in bf16 (AdamW then drops most small updates)",
    )
    ap.add_argument(
        "--eval-every", type=int, default=0, help="frames between real evaluations (0 = per epoch)"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", choices=["S", "B"], default="S")
    ap.add_argument("--stage", choices=["a", "b"], default="a")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--val", type=int, default=300, help="synthetic validation frames (max)")
    ap.add_argument("--val-scenes", type=int, default=100, help="scenes held out for validation")
    ap.add_argument("--init", type=Path, help="state dict to start from (e.g. stage a result)")
    ap.add_argument(
        "--encoder-blocks",
        type=int,
        default=2,
        help="stage b: train only the last N encoder blocks (0 = whole encoder)",
    )
    ap.add_argument("--crop-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay per optimizer step (0 = off)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"

    files = sorted(args.data.glob("*.npz"))
    # Hold out whole scenes (s{seed}_{scene}_{frame}.npz): frames of one scene share table,
    # objects and lights, so a frame-level split would leak training scenes into validation.
    scenes = sorted({f.name.rsplit("_", 1)[0] for f in files})
    held = set(
        np.random.default_rng(1234).choice(scenes, min(len(scenes), args.val_scenes), replace=False)
    )
    train_files = [f for f in files if f.name.rsplit("_", 1)[0] not in held]
    val_files = [f for f in files if f.name.rsplit("_", 1)[0] in held][: args.val]
    distill_files = (
        sorted([*args.distill.glob("*.jpg"), *args.distill.glob("*.png")]) if args.distill else []
    )
    teacher = None
    if distill_files:
        teacher = load_model(args.size, device, torch.bfloat16).eval()
        for p in teacher.parameters():
            p.requires_grad = False
    # fp32 master weights: bf16 has an 8-bit mantissa, so at encoder learning rates most
    # AdamW updates round away. Autocast still runs the forward pass in bf16.
    weight_dtype = torch.bfloat16 if args.bf16_weights else torch.float32
    model = load_model(args.size, device, weight_dtype).train()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    for p in model.encoder.parameters():
        p.requires_grad = False
    enc_params = []
    if args.stage == "b":
        blocks = list(model.encoder.blocks)
        chosen = blocks[-args.encoder_blocks :] if args.encoder_blocks else [model.encoder]
        for module in chosen:
            for p in module.parameters():
                p.requires_grad = True
                enc_params.append(p)
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.startswith("encoder")],
            "lr": args.lr,
        }
    ]
    if args.stage == "b":
        groups.append({"params": enc_params, "lr": args.lr / 10})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = args.epochs * len(train_files) // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05
    )

    ema, steps_done = None, 0
    if args.ema:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def scored_weights():
        """Swap in the EMA weights for scoring and saving; returns the live weights to restore."""
        if ema is None:
            return None
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(live[k].dtype) for k, v in ema.items()})
        return live

    def score(tag):
        live = scored_weights()
        model.eval()
        result = {
            "at": tag,
            "synthetic_val": evaluate(model, val_files, device),
            **{name: evaluate_real(model, frames()) for name, frames in SETS.items()},
        }
        model.train()
        if live is not None:
            torch.save(model.state_dict(), args.out / "_scored.pt")
            model.load_state_dict(live)
        return result

    def save(path):
        # With EMA on, the checkpoint is the averaged model that was just scored.
        scored = args.out / "_scored.pt"
        if ema is not None and scored.exists():
            scored.rename(path)
        else:
            torch.save(model.state_dict(), path)

    log = [score("start")]
    print(json.dumps(log[0]), flush=True)
    seen = 0
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        order = rng.permutation(len(train_files))
        opt.zero_grad(set_to_none=True)
        for k, idx in enumerate(order):
            z = np.load(train_files[idx])
            rgb, depth, trust, f_px = (
                z["rgb"],
                z["depth"].astype(np.float32),
                z["trust"],
                float(z["f_px"]),
            )
            if rng.random() < args.crop_prob:
                rgb, depth, trust, f_px = crop_resize(rgb, depth, trust, f_px, rng)
            gt = torch.from_numpy(depth).to(device)[None]
            w = torch.from_numpy(trust).to(device)[None].float() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = predict_depth(model, to_input(np.ascontiguousarray(rgb), device), f_px)
            loss = loss_fn(pred, gt, w) / args.accum
            loss.backward()
            losses.append(loss.item() * args.accum)
            if distill_files and rng.random() < args.distill_ratio:
                x = to_input(
                    load_unlabelled(distill_files[rng.integers(len(distill_files))]), device
                )
                if rng.random() < 0.5:
                    x = x.flip(-1)
                f_guess = 0.8 * x.shape[-1]  # unknown camera; the loss ignores global scale
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t = predict_depth(teacher, x, f_guess).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sdep = predict_depth(model, x, f_guess).float()
                (
                    args.distill_weight * distill_loss(sdep, t, args.distill_loss) / args.accum
                ).backward()
            seen += 1
            if args.eval_every and seen % args.eval_every == 0:
                r = score(f"frame {seen}")
                log.append(r)
                print(json.dumps(r), flush=True)
                save(args.out / f"frame{seen}.pt")
                (args.out / "log.json").write_text(json.dumps(log, indent=1))
            if (k + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    steps_done += 1
                    # Warm up the decay so early checkpoints are not mostly the base model.
                    decay = min(args.ema, (1 + steps_done) / (10 + steps_done))
                    with torch.no_grad():
                        for k, v in model.state_dict().items():
                            ema[k].mul_(decay).add_(v.float(), alpha=1 - decay)
        result = {
            **score(f"epoch {epoch}"),
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "minutes": (time.time() - t0) / 60,
        }
        log.append(result)
        print(json.dumps(result), flush=True)
        save(args.out / f"epoch{epoch}.pt")
        (args.out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
