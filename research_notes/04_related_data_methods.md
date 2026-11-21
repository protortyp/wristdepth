# 04 Related data and methods for wrist-camera metric depth (3-60 cm)

Compiled 2026-11-20 from web sources. Every claim carries a URL. Tags:
**[V]** checked against the primary source (paper, official page or dataset card).
**[S]** secondary or summarised source only.
**[U]** unverified or my own inference; check before relying on it.
"Derived" means my own arithmetic from cited specs.

---

## 0. TL;DR

1. **The most useful real wrist data with independent depth is DROID wrist stereo (ZED Mini, CC-BY 4.0), but you have to compute the depth yourself.** SpatialBench tried 5 stereo models on DROID wrist video and picked **S2M2** (FoundationStereo was worse on wrist views). It ships a 16-scene / 3,440-frame DROID wrist eval split. DROID-3D (DepthWorld, S2M2 depth for 71k episodes, wrist included) is gated by a form. [V] https://arxiv.org/html/2605.27367 , https://jaibardhan.com/depthworld
2. **FMB is the only large real dataset I found with true wrist RGB-D from a close-range sensor and a permissive license.** It has 2x RealSense **D405** on the Franka wrist (D405 ideal range 7-50 cm), 22,550 trajectories, **CC BY 4.0**, and is in TFDS at 256x256. [V] https://arxiv.org/pdf/2401.08553 , https://functional-manipulation-benchmark.github.io/ , https://www.tensorflow.org/datasets/catalog/fmb
3. **A direct competitor exists: DA-Next (Ropedia, 2026).** It is DA3-Giant fine-tuned on DA-Next-5M (5.5M egocentric + wrist frames, mostly sim wrist: RLBench, RoboTwin, Colosseum, Robolab). It improves AbsRel on wrist/ego views by 47-59% over DA3-Giant. The weights and data are on HF, licenses not stated. Nothing SO-100/LeRobot-specific was found. [V] https://arxiv.org/html/2605.27367 , https://huggingface.co/ropedia-ai/DA-Next
4. **Free test-time metric anchors.** The gripper fingers sit at a kinematically known 3D position in the camera frame. Prompt or completion models (PromptDA, Prior Depth Anything, CDM, LingBot-Depth) and robot-as-ruler methods (KineDepth) show that a few metric anchors fix scale cheaply. [V] links in section 3

---

## 1. Real wrist / close-range datasets with metric depth

### 1a. Top 5 for our use

| # | Dataset | Camera on wrist | Depth source, close-range quality | Size | License | Fingers in view | (a) Train | (b) Eval |
|---|---|---|---|---|---|---|---|---|
| 1 | **DROID** (raw release) | ZED Mini stereo, 63 mm baseline, 1280x720 [V] https://droid-dataset.github.io , https://www.stereolabs.com/en-ee/store/products/zed-mini | Stereo pairs only (raw MP4 stereo + SVO). RLDS has only left 180x320 images and no depth [V] https://droid-dataset.github.io/droid/the-droid-dataset . ZED Mini min range 0.1 m [V] stereolabs link. Learned stereo needed. S2M2 beats FoundationStereo, IGEV, DynamicStereo and WAFT on wrist sequences; SpatialBench keeps pixels at conf > 0.999 and a 0-3 m range [V] https://arxiv.org/html/2605.27367 | 76k traj, 350 h, 564 scenes [V] droid site. Raw stereo 8.7 TB; about 20% of raw episodes are missing from the bucket [V] the-droid-dataset page | CC-BY 4.0 [V] https://arxiv.org/pdf/2403.12945 | Yes, Robotiq-style 2F gripper visible in the wrist view [U] | **High.** Diverse real wrist views at 15-60 cm. Pseudo-GT comes from stereo, so it is not truly independent, and it fails below about 12 cm (see 1c) | **High** for 15-60 cm. SpatialBench DROID split: 16 scenes, 3,440 frames [V] sb html; HF: https://huggingface.co/datasets/ropedia-ai/SpatialBenchmark (license not stated [U]) |
| 2 | **FMB** | 2x RealSense D405 on the end-effector [V] https://arxiv.org/pdf/2401.08553 | Active-stereo D405. Ideal range 7-50 cm, +/-2% at 50 cm [V] https://intelrealsense.com/depth-camera-d405/ . TFDS depth is 256x256 float [V] tfds | 22,550 traj, 356.6 GiB TFDS [V] https://www.tensorflow.org/datasets/catalog/fmb | **CC BY 4.0** [V] https://functional-manipulation-benchmark.github.io/ | Probably [U] | **Medium-high.** The range fits us, the D405 is independent GT, and the license is redistributable. But it has one lab, white curtains, few objects [V] paper, and 256 px | **Medium.** Real D405 GT at our range. Two wrist cams also allow a stereo cross-check [U] |
| 3 | **DROID-3D** (DepthWorld) / **PointWorld-DROID** | same ZED Mini | DROID-3D: S2M2 depth plus recalibrated extrinsics, wrist included, 71.1k episodes; access request form [V] https://jaibardhan.com/depthworld . PointWorld-DROID: 320x180 uint16 mm depth; wrist depth "may exist" [V] https://huggingface.co/datasets/nvidia/PointWorld-DROID | 71.1k ep / 42,935 ep (3.9 TB) | DROID-3D: not stated [U]. PointWorld: NVIDIA License, R&D only [V] | Yes [U] | **High** if access is granted: it saves running S2M2 on 8.7 TB | Medium (same pseudo-GT caveat) |
| 4 | **HAMMER** | Rig moved by a robot arm: D435, L515, Lucid Helios ToF, polarisation RGB [V] https://www.github.com/Junggy/HAMMER-dataset | **Mesh-rendered GT** from scanned objects plus arm forward kinematics, so it is truly independent [V] https://arxiv.org/pdf/2205.04565 | 13 scenes, about 13k frames [V] | MIT [S] github | No | Low (small, no gripper) | **High as a GT-quality sanity eval** for tabletop at short range; exact ranges not checked [U] |
| 5 | **RH20T** | 1-2 in-hand cameras per platform plus 8-10 global RGB-D [V] https://rh20t.github.io | Depth 1280x720 @10 Hz. The 640x360 version has compression artefacts; the 320x180 version is lossless [V]. **Whether the in-hand cams carry depth is unconfirmed** [U] | 110k+ sequences, about 40 TB raw [V] | Scenes 1-5: CC BY-SA 4.0; scenes 6-10: NC [V] | Likely [U] | Medium, if in-hand depth exists | Medium |

### 1b. Others checked (lower value)

| Dataset | Key facts | Verdict |
|---|---|---|
| BridgeData V2 | Wrist cam is RGB only; depth comes only from the fixed over-the-shoulder cam and only in some data; 60,096 traj; CC BY 4.0 [V] https://rail-berkeley.github.io/bridgedata | No wrist depth. Wrist RGB is usable only for unlabelled or self-supervised training |
| Open X-Embodiment | Feature specs are listed per dataset in a sheet [V] https://github.com/google-deepmind/open_x_embodiment . Example: berkeley_autolab_ur5 has `hand_image` RGB but depth only in the main `image_with_depth` [V] https://www.tensorflow.org/datasets/catalog/berkeley_autolab_ur5 . FMB (above) is the OXE subset that has wrist depth | Scan the sheet for more wrist-depth subsets [U] |
| ARMBench | 235K+ picks; 5 MP top-down and transfer RGB cameras; download by request [V] https://arxiv.org/pdf/2303.16382 | No depth or wrist view. Skip |
| HOI4D | Helmet with Kinect v2 + RealSense D455; 2.4M RGB-D frames, 4,000 sequences; **CC BY-NC 4.0** [V] https://arxiv.org/html/2203.01577v4 , https://hoi4d.github.io | Ego hands at about 30-60 cm. Sensor near-range limits are high [U]. Already in DA-Next-5M (739K frames) [V] sb html. NC |
| HOT3D | Aria (RGB + mono) and Quest 3 (2x mono 1280x1024); poses plus scanned meshes; no depth maps; split licenses (hand annotations NC) [V] https://projectaria.com/datasets/hot3D/license , https://arxiv.org/html/2411.19167v2 | Depth could be rendered from the meshes [U]. Ego, not wrist |
| GraspNet-1Billion | 97,280 RGB-D images (D435 + Azure Kinect) **mounted on the arm's end link**, 256 views per scene [V] https://arxiv.org/pdf/1912.13470 ; CC BY-NC-SA, commercial use by email [V] https://graspnet.net/datasets.html | Eye-in-hand viewpoint without fingers. Sensor depth near range is weak [U]. Good extra tabletop training data if NC is acceptable |
| TransCG | 57,715 RGB-D images, 130 scenes, D435 + L515; GT from optical tracking plus object models; CC BY-NC-SA 4.0 [S] https://github.com/Galaxies99/TransCG | Transparent objects; NC |
| ClearGrasp | 50k+ synthetic, 286 real test images (D415), GT from swapped painted objects [V] https://arxiv.org/abs/1910.02550 ; license not found [U] | Tiny real eval |
| DREDS / STD | 130K Blender-rendered images with **simulated D415 IR active stereo**; STD real set has 50 objects; non-commercial data [V] https://github.com/PKU-EPIC/DREDS . Depth range 0.5-2 m per RaSim [S] https://arxiv.org/html/2404.03962v1 | Too far for 3-60 cm. Useful as a sensor-sim recipe |
| OCID | 96 cluttered scenes, 89 objects (ARID/YCB); CC BY 4.0 per repository metadata [S] https://www.acin.tuwien.ac.at/en/?p=9099 ; sensor not confirmed [U] | Probably >= 0.5 m. Skip |
| ByteCameraDepth | 170k+ RGB-depth pairs, 7 sensors incl. D405 and ZED, indoor scenes [V] https://arxiv.org/html/2509.02530v1 , https://huggingface.co/datasets/ByteDance-Seed/ByteCameraDepth | Sensor-noise modelling data, not wrist |
| LeRobot community SO-101 depth sets | e.g. `codenmood/so101_camera_dataset_0048`: wrist + front RGB-D in mm, 1 episode / 300 frames, Apache-2.0 [V] https://huggingface.co/datasets/codenmood/so101_camera_dataset_0048 ; `CarolinePascal/so101_depth_raw`: front depth only, 172 frames [V] https://huggingface.co/datasets/CarolinePascal/so101_depth_raw/blob/main/README.md | Tiny. Shows that LeRobot can record depth |
| DA-Next-5M | 5.5M frames / 22K scenes: Xperience (ego, real) 400K, ADT 86K, HOI4D 739K, and sim wrist data RLBench 1.2M, RoboTwin 2.6M, Colosseum 334K, Robolab 158K. Sim wrist cams were moved so the gripper is visible [V] https://arxiv.org/html/2605.27367 ; https://huggingface.co/datasets/ropedia-ai/DA-Next-5M (license not stated [U]) | **Ready-made sim wrist data with gripper in view.** Inherits upstream licenses (HOI4D NC) [U] |

### 1c. DROID wrist stereo at close range (derived)
- Disparity d = f·B/Z. ZED Mini B = 63 mm [V stereolabs]. I assume fx ≈ 700 px at 1280x720 [U]; read it from the SVO calibration.
  - Derived at 1280 px: Z = 10 cm gives d ≈ 441 px, 20 cm gives 220 px, 30 cm gives 147 px, 60 cm gives 74 px.
  - At FoveaVLA's 640x360 with max disparity 192 [V] https://arxiv.org/pdf/2610.20064 , the derived minimum depth is ≈ 11.5 cm.
- So DROID stereo pseudo-GT covers about 12-60 cm. It cannot give GT for 3-10 cm, our hardest band (derived).
- Lowering the ZED SDK `depth_minimum_distance` is possible but costs memory and FPS [V] https://www.stereolabs.com/docs/depth-sensing/depth-settings . Better to use S2M2 or FoundationStereo with a large max disparity, or run at half resolution.
- **License chain:** S2M2 code is CC-BY-NC-4.0 and the weights are on HF (minimok/s2m2) [S] https://huggingface.co/fastforks/s2m2 , https://arxiv.org/abs/2507.13229 . Depth labels made with an NC model are a grey zone for a public model; FoundationStereo's license is not checked [U].

---

## 2. Synthetic close-range / eye-in-hand data and sim-to-real

### 2a. Generation recipes others use

| Source | What | Numbers | URL |
|---|---|---|---|
| DA-Next-5M sim wrist | Wrist cam re-posed so the gripper is visible, rendered at 1280x720, FOV 60° (RoboTwin). DR: backgrounds, clutter, clean-background rate 0.2, table height ±0.1, lighting, camera pose ±0.05 rad (Colosseum), 5-8 distractors | 4.3M sim wrist frames | [V] https://arxiv.org/html/2605.27367 |
| BlenderProc4BOP PBR | Physically based renders clearly beat OpenGL render-and-paste. Pose estimation trained only on PBR ranked 5th of 26; adding real data was best | 50K images per dataset, 350K total | [V] https://arxiv.org/pdf/2009.07378v2.pdf , https://github.com/DLR-RM/BlenderProc/blob/main/README_BlenderProc4BOP.md |
| DREDS | Blender ray-traced IR-pattern stereo simulation of the D415, plus randomised textures, materials and lighting | 130K | [V] https://github.com/PKU-EPIC/DREDS |
| CDM "Manipulation as in Simulation" (ByteDance, ICLR'26) | Learned per-camera **hole model + value-noise model** applied to clean sim depth. CDM = RGB + raw depth through two ViTs and token fusion into DPT, from DAv2/DINOv2. Weights for D405, D435, L515, Kinect, ZED 2i | 280K+ synthetic (HyperSim, DREDS, HISS, IRS) + 170K real | [V] https://arxiv.org/html/2509.02530v1 ; models CC-BY-NC-4.0 [S] https://huggingface.co/depth-anything/camera-depth-model-d435/blob/main/README.md |
| EndoSynth (surgery, close range) | Synthetic instruments with metric depth. Fine-tuning DA-family models on it alone improves most unseen real sets "by a significant margin" | weights and data released | [V] https://arxiv.org/abs/2509.18642 , https://github.com/TouchSurgery/EndoSynth |
| Underwater DAv2 | Fine-tuned DAv2 on a synthetic underwater Hypersim; lower absolute errors on real SQUID/FLSea | | [V] https://arxiv.org/html/2507.02148 |

### 2b. What worked for fine-tuning depth foundation models

| Model | Data scale | Losses / representation | Freezing / LR | Camera handling | URL |
|---|---|---|---|---|---|
| **Depth Pro** (our base family) | Stage 1: real + synthetic mix. Stage 2: **synthetic only**, inverting the usual synthetic-then-real order | Canonical inverse depth C, with D = f_px/(w·C). Stage 1: MAE on metric sets (top-20% error pixels dropped on real), MAD-normalised (scale/shift-invariant) loss on non-metric sets, SSI gradient loss on synthetic. Stage 2: MAE + MSE + multi-scale 1st/2nd-order derivative losses (MAGE, MALE, MSGE) | Encoder LR 1.28e-5, decoder 1.28e-4 (10x), **pretrained LayerNorm frozen**, Adam, grad-clip 0.2, batch 128; 250 + 100 epochs of 72k samples | CSTM-label metric normalisation; 50% centre-crop **FOV augmentation**; separate FOV head | [V] https://arxiv.org/html/2410.02073 |
| Depth Anything V2 | 595K synthetic (Hypersim 60K, VKITTI2 20K, TartanAir 306K, IRS 103K, BlendedMVS 115K) + 62M pseudo-labelled real; top-10% loss regions ignored | SSI + **gradient-matching loss (key for sharpness on synthetic)**. Metric variants fine-tuned on Hypersim / VKITTI only | Enc 5e-6 / dec 5e-5 | none (metric heads per domain) | [V] https://arxiv.org/html/2406.09414 |
| Metric3D v2 | 16M+ images, thousands of camera models | Canonical camera space transform resolves focal/scale ambiguity | | Canonical focal | [V] https://arxiv.org/abs/2404.15506 |
| UniDepthV2 | | Edge-guided loss, geometric invariance loss, uncertainty output | | Self-promptable dense camera module; can take GT camera | [V] https://arxiv.org/abs/2502.20110 |
| MoGe-2 | Mixed real + synthetic; **real labels refined/filtered using sharp synthetic-trained predictions** | Affine-invariant point map + separate metric-scale MLP head on CLS token (decoupled) | | Point map in camera frame | [V] https://arxiv.org/html/2507.02546v1 |
| Marigold | Synthetic only (Hypersim + VKITTI) | Affine-invariant; diffusion fine-tune | "a couple of days on a single GPU" | | [V] https://arxiv.org/abs/2312.02145 |
| AnchorDepth (Depth Pro LoRA) | KITTI | LoRA rank 8 on attention of both ViT-L encoders, 2.36M params | Base frozen | | [S] https://huggingface.co/dariusan3/AnchorDepth |
| DA-Next | DA-Next-5M + 11 public sets, fixed mixture | Scale tokens for scene metric scale; optional GT camera tokens | 4x H200, 7 days | camera tokens | [V] https://arxiv.org/html/2605.27367 |

**Implications for us**
- **Distance-dependent bias.** Depth Pro's loss is MAE in canonical inverse depth, which weights near pixels strongly [V dp]. A residual bias that grows with distance is consistent with this, plus a narrow synthetic depth histogram. Remedies:
  - add an SILog or log-depth term [U];
  - balance the sampled depth histogram across 3-60 cm [U];
  - use SSI-gradient losses for shape and keep the metric loss only on synthetic frames with exact intrinsics [V dp].
- **Mix ratio.** Synthetic-only stage 2 worked for Depth Pro [V]. MoGe-2 and SpatialBench both say curated pseudo-GT beats volume ("data quality outweighs data volume") [V sb]. Keep the real DROID/FMB pseudo-GT under a scale-invariant loss if its metric scale is doubtful.
- **Freeze LayerNorm** and use a 10x lower encoder LR, as the Depth Pro recipe does [V].
- **Pass the known SO-101 intrinsics** instead of trusting the FOV head on fisheye-ish wrist lenses [U]. Note the 50% FOV-crop augmentation in the Depth Pro recipe [V].

---

## 3. Complementary / baseline metric-depth methods

| Method | Input | Close-range / wrist relevance | License | URL |
|---|---|---|---|---|
| **Prompt Depth Anything (PromptDA)** | RGB + low-res LiDAR prompt fused at multiple decoder scales; trained with simulated LiDAR on synthetic data plus pseudo-GT on real; S 25.1M / L 340M | **Prompt = sparse depth from finger keypoints or a cheap ToF** [U]. PromptDA++ adds pattern-agnostic prompts [S] https://arxiv.org/html/2412.14015v3 | Apache-2.0 [S] | [V] https://arxiv.org/abs/2412.14015 , https://huggingface.co/depth-anything/prompt-depth-anything-vits |
| Prior Depth Anything | Any prior (sparse points, low-res, holes), coarse-to-fine; ICLR'26 | Sparse finger/kinematic points as prior [U] | not found [U] | [V] https://arxiv.org/abs/2505.10565v1 , https://prior-depth-anything.github.io/ |
| CDM (ByteDance) | RGB + raw sensor depth to clean metric depth; D405 model | Use if we add a D405 to the wrist: cleaner GT for eval [U] | CC-BY-NC-4.0 [S] | [V] https://arxiv.org/html/2509.02530v1 |
| LingBot-Depth | Masked depth modelling; sensor holes treated as masks | RGB-D completion for GT cleanup | Apache-2.0 [S] | [V] https://arxiv.org/abs/2601.17895v1 , https://huggingface.co/robbyant/lingbot-depth/blob/main/README.md |
| MOMA (IROS'25) | One-shot scale-rotation-shift alignment from sparse GT points at calibration | Fixed cam only; we need a per-frame variant [V] | | [V] https://arxiv.org/abs/2506.17110 |
| KineDepth (IROS'25) | Robot is the "measuring stick": LSTM metric regressor trained online with a Kalman filter on DAv2 relative depth | **Fixed** camera; 5.3 cm overall error vs Metric3D v2 +1.5 cm; −22.1% error [V] | CC BY-ND (paper) | [V] https://escholarship.org/uc/item/3jm3n4t5 , arXiv 2409.19490 |
| Gripper-camera FK + SLAM | Monocular cam in the gripper; FK motion constraints in a factor graph give metric scale | Directly analogous to a wrist cam on a moving arm | | [S] https://arxiv.org/pdf/2511.05816 |
| AnchorHEC | Targetless hand-eye calibration; FK fixes metric scale of joint reconstruction | Wrist-cam extrinsics for triangulation | | [S] https://arxiv.org/html/2611.02884 |
| Multi-view feed-forward (DA3, VGGT, π3, MapAnything) on DROID wrist | Single-frame AbsRel (scale-aligned [U]) on SpatialBench-DROID: DA3-S 0.198, DA3-B 0.158, DA3-G 0.114, VGGT 0.121, MapAnything 0.100, VGGT-Omega 0.073, DA-Next 0.098 | Baselines for our eval table; wrist view is "the dominant OOD failure mode" | various | [V] https://arxiv.org/html/2605.27367 (Table 15) |
| Stereo wrist camera | ZED Mini (min 0.1 m) [V]; RealSense D405 (7-50 cm ideal, Min-Z 7 cm @480p) [V] https://intelrealsense.com/depth-camera-d405/ | **A D405 on the SO-101 wrist is the cheapest independent GT for 7-50 cm.** Below 7 cm nothing off-the-shelf; use kinematic contact/touch points [U] | | |
| LeRobot depth recording | `RealSenseCameraConfig(use_depth=True)`; `lerobot-record` has a `DepthEncoderConfig` (depth_min, depth_max, use_log quantiser) and `--dataset.depth_output_unit` | Record D405 GT in standard LeRobot format | Apache-2.0 [U] | [V] https://huggingface.co/docs/lerobot/cameras , https://huggingface.co/docs/lerobot/en/video_encoding_parameters.md |

**Test-time scale from known geometry** (my proposal, [U]):
- Fingertip pixel positions (from the URDF + gripper joint, or segmentation) give 2-6 metric depth anchors per frame at about 3-10 cm.
- Contact events (object touched, table touched by the fingertip) give anchors at the object.
- Wrist motion between frames (FK, about 1 mm precision on SO-101 [U]) gives triangulation baselines of 1-5 cm, enough for 10-40 cm.
- Feed these as PromptDA-style prompts, or fit a per-frame scale + shift in inverse depth (MOMA-style, [V] approach).

---

## 4. Existing wrist-depth models / projects (avoid duplication, compare)

| Project | What | SO-100/LeRobot? | URL |
|---|---|---|---|
| **DA-Next** (Ropedia, 2026) | DA3-Giant (1.3B) fine-tuned for ego + wrist; DROID wrist single-frame AbsRel 0.098 | No; general wrist | [V] https://huggingface.co/ropedia-ai/DA-Next , https://github.com/Ropedia/SpatialBench |
| DepthWorld (2026) | Video world model jointly predicting RGB + depth; built DROID-3D with S2M2 wrist depth | No | [V] https://arxiv.org/abs/2610.08780 , https://jaibardhan.com/depthworld |
| FoveaVLA (2026) | Fast-FoundationStereo metric depth on DROID **external** cams; argues for dropping the wrist cam | No | [V] https://arxiv.org/pdf/2610.20064 |
| PointWorld (NVIDIA, 2026) | FoundationStereo depth + VGGT poses on DROID; good poses for about 60% of episodes | No | [S] https://arxiv.org/pdf/2601.03782 |
| OmniWorld-DROID | FoundationStereo wrist depth with original poses; SpatialBench calls it unreliable | No; CC-BY-NC-SA | [V] https://huggingface.co/datasets/InternRobotics/OmniWorld , https://arxiv.org/html/2605.27367 |
| CDM D405 | RGB-D refinement for D405 | No; NC | [V] https://manipulation-as-in-simulation.github.io |
| SO-101 community | Tiny RGB-D recordings only; **no monocular wrist-depth model for SO-100/101 found** (searched GitHub/HF/arXiv 2026-11-20) | — | [V] searches; see 1b |

**Gap we fill:** a small (ViT-S), focal-aware model for the 3-60 cm band on low-cost arms, with the gripper in view. Compare against these baselines:
- DA-Next
- DA3-S/B (SpatialBench numbers above)
- raw Depth Pro
- PromptDA with finger prompts

Use these eval sets:
- SpatialBench-DROID wrist split
- FMB wrist D405
- our own SO-101 + D405 recordings
