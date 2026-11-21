# 03 Appearance gap: synthetic vs real wrist frames

Date: 2026-11-20. Data, plots and scripts: `$WRIST_DEPTH_DATA/research/appearance/`
(`metrics.csv`, `verify.csv`, `hist_metrics.png`, `hist_verify.png`, `vignetting.png`,
`sheet_*.jpg`, `crops_hf_real_synth.jpg`, `verify_sheet.jpg`, `verify_crops.png`, `scripts/`).
The dev frames are private and stay on that drive.

## Groups

| group | n | source |
|---|---|---|
| hf | 112 (20 datasets) | community SO-101 wrist frames: AV1 video, re-saved as JPEG q73-94 (median 85, read from the DQT) |
| real_raw | 40 | dev wrist cam, raw PNG (distorted; k1 -0.457, k2 0.270, k3 -0.096, f 557 px) |
| real_undist | 40 | the same frames after `cv2.undistort(K, dist, K)`, which is what `eval_gt.py` feeds the model |
| synth | 300 | 200 pilot + 100 pilot2 `.npz`, after the current `camera_effects` |

pilot2 was still being written during this analysis (1297 -> 2837 frames), so the synth samples
differ slightly between runs. Metrics are computed on luma Y (0-255) of the full 640x480 frame.
Each cell is the median [p10-p90].

## Measured statistics

| metric | hf | real_raw | real_undist | synth (current) |
|---|---|---|---|---|
| Laplacian variance | 106 [19-2220] | 440 [144-681] | 119 [46-194] | **57 [13-390]** |
| HF energy >0.25 cyc/px | 0.0088 [0.0024-0.075] | 0.049 [0.024-0.10] | 0.022 [0.011-0.051] | **0.0058 [0.0012-0.031]** |
| HF energy >0.125 cyc/px | 0.041 | 0.118 | 0.077 | **0.025** |
| edge 10-90% width (px) | 2.25 [1.25-4.0] | 1.25 [0.75-1.8] | 1.6 [1.25-2.25] | **3.25 [1.7-4.5]** |
| frames with edge width >3 px | 32% | 0% | 2% | **57%** |
| flat-region noise sigma (DN, mean of RGB) | 0.02 [0-0.12] | 0.48 [0.38-0.59] | 0.28 [0.24-0.32] | 0.28 [0.05-0.51] |
| Y mean | 137 [93-188] | 82 [72-94] | 84 [74-96] | 102 [65-143] |
| Y p1 / p50 / p99 | 12 / 150 / 246 | 12 / 81 / 149 | 19 / 82 / 149 | 18 / 89 / 223 |
| clipped high (any ch >=250) | 1.4% [0-22%] | 0 | 0 | 0.1% [0-7.7%] |
| global RMS contrast (std Y/255) | 0.235 [0.16-0.33] | 0.136 [0.11-0.15] | 0.127 [0.11-0.15] | 0.20 [0.13-0.27] |
| local RMS contrast (sigma 5 px) | 0.014 | 0.014 | 0.011 | 0.014 |
| gray-world R/G | 1.02 [0.96-1.11] | 1.07 | 1.07 | 1.04 [0.92-1.19] |
| gray-world B/G | 1.01 [0.86-1.06] | 1.03 | 1.03 | **0.90 [0.81-1.01]** |
| saturation mean / p90 | 0.17 / 0.55 | 0.13 / 0.20 | 0.13 / 0.20 | **0.25 / 0.44** |
| 8x8 blockiness (1 = none) | 1.85 [1.24-2.38] | 1.00 | 1.00 | 1.53 [1.14-2.06] |
| directional edge-width ratio (max/min) | 1.47, 13% >2 | 1.50, 15% >2 | 1.33 | 1.29, 13% >2 |

Calibrating the metrics against dev PNGs re-encoded as JPEG:

| encoding | blockiness | noise | lapvar | HF>0.25 | edge w |
|---|---|---|---|---|---|
| PNG | 1.00 | 0.48 | 374 | 0.049 | 1.25 |
| JPEG q95 / q85 / q75 / q60 | 1.08 / 1.28 / 1.42 / 1.64 | 0.36 / 0.25 / 0.15 / 0.07 | 373 / 343 / 339 / 329 | 0.050 / 0.050 / 0.047 / 0.043 | 1.25 / 1.25 / 1.25 / 1.5 |
| 5x5 Gaussian blur + q85 | 1.42 | 0.18 | 22 | 0.0023 | 2.75 |

How to read these numbers:
- **Synthetic frames are softer than every real group.** Median Laplacian variance is 2x below
  the dev frames as the model sees them (undistorted) and 8x below the raw dev frames. Median edge
  width is 3.25 px against 1.6 px. The splat is soft before augmentation, and the 50% chance of a
  3/5 px Gaussian blur makes it worse: a single 5x5 blur cuts Laplacian variance by 17x.
- **Synthetic frames are not low in contrast.** RMS contrast is 0.20, above the dev camera (0.13)
  and below the community frames (0.235). The softness makes them look flat. The dev camera is
  dim and flat: p99 is 149 and nothing clips. Community frames are bright: p99 is 246 and 22% of
  pixels clip at p90.
- **Colour is off.** Synthetic frames are 1.5-1.9x more saturated than the real groups and warm
  (B/G 0.90 against about 1.02). The splat inherits the warm room light of the iPhone capture.
- **Noise is not the gap.** The dev camera's ISP denoises (sigma about 0.5 DN in flat regions).
  The AV1 community videos have essentially no noise left in flat regions. The current additive
  sigma of 1-5 DN mostly disappears after JPEG, so the measured synthetic value (0.28) is in range.
- **Compression differs by source.** Dev frames have none. Community frames are heavily
  compressed: blockiness is 1.85 even though the q85 re-save alone only adds about 0.28, so the
  AV1 source behaves like JPEG q50-65. That is an inference: Hugging Face rate-limited us (HTTP 429),
  so we could not pull native PNG frames to check.
- **Sharpness in the community data is bimodal across datasets.** DmitryStrog, pjhun, kmg0620 and
  cagataydev are sharp (Laplacian variance 700-4400, edge width 1-1.75 px). Most others are soft
  (Laplacian variance 15-200, edge width 2.3-3.9 px). Per-dataset medians are produced by
  `scripts/summarize.py metrics.csv --ds`.
- **Undistortion softens frames.** On the dev frames it cuts Laplacian variance from 440 to 119
  and HF energy from 0.049 to 0.022, because peripheral pixels are magnified. Raw-image training
  needs a sharper target than undistorted training.
- **Motion blur is not detectable.** The directional edge-width ratio has the same distribution
  for the community frames and the static, settled dev frames (13% vs 15% above 2). If motion blur
  is present it is rare, and this metric cannot isolate it.

### Vignetting
Radial luminance relative to the image centre, top 55% of the frame only (this excludes the
fingers). The values are the median over frames, with r normalised to the half-diagonal.

| r | hf | real_raw | real_undist | synth |
|---|---|---|---|---|
| 0.6 | 0.93 | 0.92 | 0.95 | 1.03 |
| 0.8 | 0.88 | 0.77 | 0.85 | 1.02 |
| 0.9 | 0.84 | 0.72 | 0.77 | 0.99 |

The real groups fit `V(r) = 1 - k r^4` with k about 0.2 (hf), 0.4 (raw) and 0.3 (undistorted).
The synthetic frames have no vignetting. The scene layout (desk in the centre) contributes to
these curves, but synth has the same layout and stays flat.

### Lens distortion (calibration and visual review of `sheet_*.jpg`)
- **Dev camera:** strong barrel distortion. The polynomial is monotonic only up to normalised
  r = 1.15 (max distorted r 0.742), and the top-left raw corner (r_d 0.80) lies outside the
  model's valid range. To fill a raw 640x480 frame, a pinhole render at f = 557 must cover
  u -190..734 and v -138..594, a canvas of about 924x732 (1.45x by 1.5x). Horizontal FOV is about
  78 deg raw against 60 deg for the 640 px pinhole.
- **Community:** nearly every dataset shows barrel/fisheye curvature of desk edges and
  checkerboards that is as strong as the dev camera's or stronger (the armnet checkerboards are
  extreme). None look rectilinear.
- **Synthetic:** pinhole (rectilinear).
- The LSD long-segment ratio (`straight`) was tried as a numeric proxy. It is dominated by
  content (hf 0.32, raw 0.41, undistorted 0.46, synth 0.33) and is not usable.

### Other visual findings
- Splat holes render black, and stretched Gaussians create streaks. Both show up as the very dark
  synthetic tail (Y p1 = 0 in 10% of frames) and the low-HF tail.
- After brightness scaling to 170-245 and WB jitter, the finger sprites read lavender or pink. The
  sprites themselves are neutral (BGR about 89/87/85).

## Augmentation and rendering targets

Implemented as `scripts/new_effects.py::camera_effects_v2`:

| knob | current | recommended |
|---|---|---|
| Gaussian blur | p 0.5, 3/5 px kernel | **off**. Optionally p 0.15 with sigma 0.5-1.0 or a 3-7 px motion kernel for the soft community tail. |
| sharpening | none | **unsharp mask, sigma 1.0-1.6, amount 0.6-1.3**. Apply it to the splat background *before* compositing objects and fingers, to avoid halos on sprite edges (visible in `verify_crops.png`). |
| white balance | x U(0.92,1.08) per channel | base gain (R,G,B) = (0.98, 1.00, **1.10**), then x U(0.94,1.06) |
| saturation | none | scale U(0.5, 0.95) about luma |
| contrast | none | scale U(0.65, 1.1) about the mean |
| exposure | x U(0.75,1.25) | log-uniform x 0.8-1.6, gamma U(0.85,1.2), clipping allowed (community p90 clips 22%) |
| vignetting | none | `1 - k r^4`, k U(0.15, 0.45), r normalised to the half-diagonal |
| noise | sigma U(1,5) | **sigma U(0.3, 1.2)** before compression (the real ISP denoises) |
| compression | JPEG q60-95 always | p 0.4 none (dev-like), p 0.6 JPEG **q45-80** (stand-in for low-bitrate AV1). Optionally run real AV1 (svt-av1, CRF 35-50) offline on a subset. |
| distortion (raw-image training) | none | Render the pinhole view at about 1.5x canvas (about 960x736 at f 557), then remap through Brown-Conrady with (k1,k2,k3) = s x dev calibration, s U(0.7,1.25), f x U(0.9,1.1), principal point +-15 px. Warp depth and trust with nearest-neighbour sampling on the same map. Raw-mode sharpness targets are those of real_raw. |
| fingers | median brightness 170-245 | Apply the WB and saturation change *after* compositing (as above), so the sprites are not tinted separately. |
| splat holes | black | Fill with an inpaint or a blurred far background, or drop frames with >5% hole pixels. |

## Check: 20 random synthetic frames (already augmented) re-processed with v2

| metric | real_undist | hf | synth old20 | synth new20 |
|---|---|---|---|---|
| Laplacian variance | 119 | 106 | 62 | 224 |
| HF >0.25 | 0.022 | 0.0088 | 0.0070 | **0.024** |
| edge width (px) | 1.6 | 2.25 | 3.25 | **2.1** |
| noise sigma | 0.28 | 0.02 | 0.29 | 0.47 |
| Y mean / p99 | 84 / 149 | 137 / 246 | 100 / 219 | 98 / 211 |
| RMS contrast | 0.127 | 0.235 | 0.202 | 0.183 |
| B/G | 1.03 | 1.01 | 0.89 | **0.99** |
| saturation mean | 0.13 | 0.17 | 0.25 | **0.155** |
| blockiness | 1.00 | 1.85 | 1.46 | 1.69 |

After v2, every synthetic median lies between the dev (undistorted) and community medians. Edge
width and HF energy now match the undistorted dev frames, and colour and saturation are within
range. Laplacian variance overshoots (p90 1350) because these frames were already blurred and
JPEG'd before sharpening, which amplifies block edges. Applied in `compose.py` to the clean splat
render, sharpening should need the lower half of the amount range. Re-measure after the next
render with `scripts/run_all.py` (add the group) and `scripts/summarize.py`.
