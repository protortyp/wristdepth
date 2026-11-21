# 06 - Pilot results: twin fine-tuning and cross-lab evaluation (2026-11-21)

Depth Pro 2 S, fine-tuned from the released weights on composites over the private desk twin.
Test sets: `so101` = 40 frames from our SO-101 wrist camera with kinematic desk-plane GT;
`droid` = 16 SpatialBench DROID wrist frames (Franka + Robotiq, other labs, S2M2 stereo GT).
"Shape" = AbsRel after per-frame median scaling (what SpatialBench reports).

| Run | Training data | so101 scale / AbsRel / shape | droid scale / AbsRel / shape |
|---|---|---|---|
| base | - | 4.32 / 3.75 / 0.096 | 2.95 / 2.09 / 0.170 |
| A1 | 2.4k twin, decoder only | 0.79 / 0.25 / 0.136 | 0.48 / 0.49 / 0.435 |
| B2 | 3k distance-balanced twin, + last 2 encoder blocks | 0.85 / 0.21 / 0.084 | 0.49 / 0.51 / 0.466 |
| C5 | 5k twin, community pose mix + v3 camera look | 0.74 / 0.29 / 0.053 | 0.38 / 0.59 / 0.461 |
| D4 | B's poses + v3 camera look | 0.72 / 0.30 / 0.081 | 0.44 / 0.55 / 0.481 |

Findings:
1. Synthetic wrist data fixes metric scale on the scene it was built from (4.3x -> 0.85x).
2. A single-scene twin overfits: on other labs scale overshoots to ~0.45x and depth shape gets
   2.7x worse than the base model. The twin is a diagnostic, not a training source for a
   public model.
3. The v3 camera look gave the best shape on our frames (0.053) but a lower scale; with one
   40-frame test scene and +-0.1 epoch-to-epoch swings, scale differences between runs of
   that size are not reliable.
4. The base model's shape is already good (0.10 / 0.17); the problem is almost entirely
   metric scale. Fine-tuning must fix scale without damaging shape: diverse scenes, keep
   general-depth replay, prefer fewer trainable parameters, and select checkpoints on several
   real sets, never one.

## Procedural data + self-distillation (same day)

Procedural BlenderProc scenes (CC0 assets, fitted SO-101 CAD gripper, census distributions),
self-distillation from the frozen base model on ~3.6k community wrist frames. `droid` now uses
785 SpatialBench DROID wrist frames.

| Run | Data | Distill | so101 scale / AbsRel / shape | droid scale / AbsRel / shape |
|---|---|---|---|---|
| base | - | - | 4.32 / 3.75 / 0.096 | 2.53 / 1.99 / 0.183 |
| E (ep2) | 8.9k procgen (mostly <60 cm) | ratio .5, w1 | 1.67 / 0.95 / 0.203 | 0.77 / 0.33 / 0.175 |
| F (ep1) | same | ratio 1, w3 | 1.26 / 0.39 / 0.111 | 0.76 / 0.33 / 0.168 |
| G (8k frames) | 23.7k procgen, +5 cm-1.5 m mix, walls, flat prints | ratio 1, w3 | 1.17 / 0.34 / 0.118 | 0.73 / 0.35 / 0.180 |
| G (ep1) | same | ratio 1, w3 | 1.64 / 0.78 / 0.124 | 0.92 / 0.35 / 0.176 |

Findings:
1. Distillation weight 3 keeps depth shape at base quality on both sets; weight 1 lets texture
   leak into depth on our desk (printed board read as a raised slab).
2. Cross-lab metric error drops 5-6x (DROID AbsRel 1.99 -> 0.33).
3. Remaining error is distance compression (DROID G: 1.28x below 20 cm, 0.68x beyond 70 cm;
   base: 3.98x -> 1.47x) plus a per-setup offset: at equal distance our desk reads ~1.45x
   farther than DROID. Not lens (raw vs undistorted 1.22 vs 1.26), fingers (masked/cropped
   1.75-1.77), image size, or appearance (gain/grey/saturation/blur 1.5-1.8). GT is
   independent of intrinsics (FK plane vs board PnP agree to mm).
4. With two real test setups, checkpoint choice trades one setup against the other. More real
   setups (board-plane GT needs no twin) are needed before further tuning.

## Depth Pro 2 B, held-out DROID scenes, EMA (2026-11-21)

DROID wrist scenes are split alternately into validation (8) and test (8). Checkpoints are
chosen on SO-101 + DROID-val only; DROID-test is reported, never used for selection.
Format: scale / AbsRel / shape.

| Model | SO-101 | DROID val | DROID test |
|---|---|---|---|
| B base | 2.21 / 1.530 / 0.068 | 1.53 / 0.904 / 0.177 | 2.22 / 1.419 / 0.101 |
| I (B, 36k procgen, distill w3, EMA .998) @ 8k frames | 1.18 / 0.292 / 0.084 | 0.63 / 0.353 / 0.171 | 0.87 / 0.234 / 0.094 |
| I @ end (36k) | 1.32 / 0.389 / 0.104 | 0.59 / 0.392 / 0.172 | 0.81 / 0.262 / 0.096 |

Release candidate 1 = I @ 8k. ONNX parity vs PyTorch FP32: FP32 AbsRel 0.0002, FP16 0.0006,
INT8 (weight-only) 0.0044, delta1 1.000.

Other checks:
- Gripper as a ruler: real SO-101 fingers are predicted 1.4-1.9x too far, the same direction
  as the desk, so the SO-101 offset is image-level for this camera. Per-frame finger anchoring
  is too noisy to help (AbsRel 0.25-0.34 vs 0.29-0.39 raw).
- 5-frame per-camera scale calibration: SO-101 AbsRel 0.39 (base S) vs 0.27 (fine-tuned S).

## Recipe sweep, seeds and release candidate 3 (2026-11-21)

B, 10.3k-frame procedural subset, distill ratio 1, EMA .995, scored every 2k frames. Selection
rule (fixed before the sweep): lowest SO-101 + DROID-val AbsRel. Sweep rows use every 4th DROID
and RoboLab frame.

| Run (best checkpoint by rule) | VAL | DROID test scale / AbsRel / shape | RoboLab |
|---|---|---|---|
| J: 2 encoder blocks @ 4k (= candidate 2) | 0.589 | 0.92 / 0.245 / 0.098 | - |
| K1: 4 blocks @ 4k | **0.532** | 0.96 / 0.250 / 0.090 | 1.10 / 0.308 / 0.165 |
| K2: 4 -> 2 blocks, lr 5e-5 @ 4k | 0.598 | 0.90 / 0.255 / 0.100 | 1.17 / 0.394 / 0.206 |
| K3: distill weight 5 @ 4k | 0.581 | 0.91 / 0.246 / 0.097 | 1.13 / 0.382 / 0.215 |
| K4: 8 blocks @ 4k | 0.567 | 1.00 / 0.268 / 0.085 | 1.04 / 0.270 / 0.150 |
| K5: full encoder @ 4k | 0.565 | 1.01 / 0.266 / 0.085 | 1.00 / 0.250 / 0.145 |
| K1 seed 2 @ 4k | 0.551 | 0.97 / 0.247 / 0.090 | 1.05 / 0.289 / 0.172 |
| K1 seed 3 @ 4k | 0.533 | 0.97 / 0.244 / 0.090 | 1.05 / 0.289 / 0.172 |
| Soup of the three K1 seeds (full sets) | 0.554 | 1.01 / 0.256 / 0.091 | 1.09 / 0.307 / 0.166 |

- More trainable encoder blocks help the held-out sets; beyond 4 they cost SO-101 shape and
  lose on the rule. K1 is reproducible across seeds (VAL 0.532-0.551).
- Every run peaks near 4k frames; longer training drifts toward the synthetic distance prior.
- Release candidate 3 = K1 @ 4k. Full held-out sets: DROID test 1.00 / 0.255 / 0.091 (stock
  2.27 / 1.445 / 0.103); RoboLab 1.11 / 0.314 / 0.162 (stock 2.92 / 2.135 / 0.267); SO-101
  1.09 / 0.224 / 0.087. All ONNX graphs and a TensorRT FP16 engine pass parity (<= 0.0009;
  INT8 0.0037, not gated).

## Depth Pro 2 S variant (not a release candidate)

Same selection rule, full sets (scale / AbsRel / shape):

| Model | VAL | SO-101 | DROID test | RoboLab |
|---|---|---|---|---|
| S base | 5.420 | 4.32 / 3.752 / 0.096 | 2.92 / 2.497 / 0.132 | 3.42 / 3.064 / 0.370 |
| S, G @ 8k (2 blocks, 23.7k data) | **0.834** | 1.18 / 0.340 / 0.117 | 0.85 / 0.298 / 0.135 | 0.89 / 0.349 / 0.258 |
| S, F (2 blocks, 8.9k data) | 0.866 | 1.27 / 0.393 / 0.110 | 0.89 / 0.289 / 0.128 | 0.97 / 0.330 / 0.246 |
| S, K1 recipe (4 blocks, EMA) @ 8k | 0.922 | 1.35 / 0.472 / 0.123 | 1.09 / 0.354 / 0.120 | 1.02 / 0.373 / 0.205 |

S stays well behind B (B candidate 3: VAL 0.540, DROID test 0.255). Exporting S G @ 8k: FP32,
FP16, clip and stream ONNX pass parity, but the TensorRT FP16 engine returns inf on 2 sky
pixels (reference 165 m) and weight-only INT8 on 0.01% of pixels (155 m) with AbsRel 0.029 on
the rest: the fine-tuned S drives far-field inverse depth so close to zero that reduced
precision flips it. Irrelevant inside the 4 m wrist range, but the gate fails as specified, so
S is not staged. Release B only.
