# 05 - Assets, licenses, wrist-camera models, BlenderProc (for a public synthetic wrist-depth dataset)

Researched 2026-11-20. Goal: a procedural BlenderProc/Cycles generator whose **rendered RGB + metric depth can be
released publicly**. Each fact has a source. Tags: **[verified]** = checked directly (file, API or repo contents);
**[doc]** = from the linked page; **[inferred]** = my reasoning, check before relying on it.

## TL;DR

- **Clean licenses:** ambientCG (CC0), Poly Haven (CC0), the SO-ARM100 CAD (Apache-2.0), GSO (CC-BY 4.0),
  YCB (CC-BY 4.0), OmniObject3D (CC-BY 4.0), BOP HB (CC0) / T-LESS and LM (CC-BY 4.0), and Objaverse filtered to
  `license in {"by","cc0"}`.
- **Avoid for a public dataset:** HOPE (repo says **CC BY-NC-SA 4.0**; BOP lists BY-SA), ABO (CC BY-NC), ITODD,
  XYZ-IBD and TYO-L (NC), and HOT3D (custom agreement). Also avoid Objaverse NC/SA objects and Printables/
  Thingiverse files without an explicit CC-BY/CC0 tag.
- **Dishware gap:** Scanned datasets handle white glossy and transparent items badly (GSO says so itself). Model
  plates, bowls, mugs and glasses **procedurally** (revolved profiles, CC0 materials). This avoids license problems
  and gives exact metric scale.
- **Camera:** The SO-ARM100 wrist-mount READMEs recommend the **Innomaker U20CAM-1080P**. It is a 32x32 mm UVC
  board with a 1/3" rolling-shutter sensor, a 2.8 mm M12 lens, 130 deg D / 103-106 deg H FOV and TV distortion
  below -17%. An **equidistant-fisheye model fits its specs well** (see section 4).
- **BlenderProc 2.8.0:** It is pinned to Blender 4.2.1 (Linux x64, macOS x64/arm64; **no Linux ARM64**). Its
  built-in distortion is Brown-Conrady only, applied as a post-process. For 100-130 deg lenses, use Cycles'
  native fisheye camera through `bpy`, or render a wide pinhole image and remap it yourself.

---

## 1. Object assets

| Source | Count / content | License (exact) | Format | Metric scale | Material quality | Use? |
|---|---|---|---|---|---|---|
| Google Scanned Objects (GSO) | 1030 objects (Fuel API lists 1033 names [verified]). Shoe 254, Consumer Goods 248, Other toys 147, Bottles/Cans/Cups 53 ([paper](https://ar5iv.labs.arxiv.org/html/2204.11918)) | "Creative Commons Attribution 4.0 International" (per-model Fuel metadata and the header of each `.mtl` [verified]); [blog](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) | SDF + OBJ + PNG texture; about 13 GB | **Yes, meters**: Cole_Hardware_Mug_Classic_Blue bbox is 0.166 x 0.126 x 0.095 [verified] | Diffuse-only texture (`map_Kd`, "beckmann" shader tag [verified]); the paper says "highly specular or transparent objects cannot be represented" | **Yes**, core set. Dishware by name [verified]: about 16 bowls, about 20 plates/saucers (mostly plant saucers), a few mugs/ramekins, a white porcelain pitcher and teapot, 1 "glass" (a magnifier). Add roughness and specular yourself. |
| YCB | 77-80+ objects (food boxes, cans, fruit, mug, bowl, plate, tools) | CC BY 4.0 ([AWS registry](https://registry.opendata.aws/ycb-benchmarks), [ai-habitat/ycb LICENSE](https://www.huggingface.co/datasets/ai-habitat/ycb/blob/main/LICENSE.txt)) | OBJ/PLY (google_16k textured) | ycbbenchmarks meshes are in meters; the BOP YCB-V models are in mm [inferred from BOP format; check] | Scanned diffuse texture | **Yes**: red mug, bowl, plate, banana, foam brick, and so on |
| BOP datasets | Per dataset: HB 33 (17 toy, 8 household, 8 industrial), T-LESS 30, LM 15, YCB-V 21, HOPE 28, HANDAL 40, HOT3D 33 | HB "CC0 1.0 Universal"; LM and T-LESS "CC BY 4.0"; YCB-V "MIT"; LM-O, TUD-L, HANDAL, IPD "CC BY-SA 4.0"; HOPE "CC BY-SA 4.0" on BOP but **"CC BY-NC-SA 4.0" in [the HOPE repo](https://github.com/swtyree/hope-dataset)**; ITODD, XYZ-IBD "CC BY-NC-SA 4.0"; TYO-L "CC BY-NC 4.0"; HOT3D custom ([BOP datasets](https://bop.felk.cvut.cz/datasets/)) | PLY, **mm** units (BOP convention); BlenderProc has `load_bop_objs` ([docs](https://dlr-rm.github.io/BlenderProc/blenderproc.api.loader.html)) | Yes (mm) | Vertex color or texture; HOT3D has PBR | **Use HB, T-LESS, LM, YCB-V.** BY-SA (HANDAL) is usable only if the dataset accepts share-alike. **Skip HOPE**: the license conflict plus NC in the source repo. |
| Objaverse 1.0 | 798,759 UIDs ([docs](https://objaverse.allenai.org/docs/objaverse-1.0)); LVIS subset about 47K objects / 1,156 categories ([paper](https://arxiv.org/abs/2212.08051)) | Dataset as a whole: ODC-By v1.0. Per object: CC-BY 4.0 721K, CC-BY-NC 25K, CC-BY-NC-SA 52K, CC-BY-SA 16K, CC0 3.5K ([HF card](https://huggingface.co/datasets/allenai/objaverse/raw/main/README.md)) | GLB | **No.** Scales are "quite arbitrary" ([HF discussion #44](https://huggingface.co/datasets/allenai/objaverse/discussions/44)). Rescale with per-category size priors. | Very mixed; use [Objaverse++](https://arxiv.org/abs/2504.07334) quality tags to filter | **Filter**: `objaverse.load_annotations()[uid]["license"]` must be `"by"` or `"cc0"` (the docs filter on `"by"`). Keep attribution (author + URL) per object in the release manifest. There is a reputational and NoAI-tag debate around Sketchfab content ([the-decoder](https://the-decoder.com/sketchfab-objaverse-ai-copyright-dispute-enters-third-dimension/)), so use it for diversity, not as the backbone. |
| Poly Haven models | 522 models [verified, API]: dishes 13, food 19, office 12, electronics 32, containers 68, tables 27 | Assets are "CC0" ([license](https://polyhaven.com/license)). Their own example renders are not CC0, but **your renders of CC0 assets are fine**. | glTF/FBX/USD/.blend, 1k-8k PBR (Diffuse, Rough, Metal, nor_gl, AO, ARM [verified]) | **Yes**: the API `dimensions` field is in mm (apple 98 x 96 x 85, wooden bowl 313 x 309 x 94 [verified]) | Best-in-class PBR | **Yes**: tea_set_01, wooden_bowl_01/02, food_*, stationery_supplies, office_notepads, vintage_stapler, wooden_cutting_board, and so on |
| OmniObject3D | 6,000 scans, 190 categories ([CVPR'23](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_OmniObject3D_Large-Vocabulary_3D_Object_Dataset_for_Realistic_Perception_Reconstruction_and_CVPR_2023_paper.html)) | "released under the CC BY 4.0" ([repo README](https://github.com/omniobject3d/omniobject3d)) | Textured mesh | Has rescaled-to-world poses; mesh units unconfirmed [check] | Scanned diffuse | Optional, for many food/fruit/toy categories |
| ABO (Amazon Berkeley Objects) | about 8K artist meshes with PBR | **CC BY-NC 4.0** ([paper](https://arxiv.org/abs/2110.06199)) | glTF | yes | PBR | **No** (NC) |

**Dishware, glasses and cables: generate procedurally.** GSO, YCB and OmniObject3D are scans, so their white
glossy and transparent items are weak or missing. Rotate a randomized 2D profile with Blender's Screw/Spin modifier
to build plates, bowls, mugs (plus a handle sweep) and glasses (Glass BSDF with real wall thickness). Build cables
as bezier curves with bevel plus random control points. Give each a Principled BSDF (white ceramic: base 0.85-0.95,
roughness 0.05-0.3, clearcoat). This has no license burden, exact metric GT, and unlimited variety [inferred].

## 2. Surfaces, materials, HDRIs

| Source | What | License | Counts / notes |
|---|---|---|---|
| [ambientCG](https://ambientcg.com) | PBR materials | "CC0 1.0 Universal", which covers files **and** previews; "copy, modify, distribute ... even for commercial purposes"; no credit required ([license](https://docs.ambientcg.com/license/)) | 2,043 materials [verified, API v2]: wood 295, tiles 222, metal 215, fabric 106, marble 77, concrete 67, leather 50 (desk mats), plastic 28, carpet 25, paper 10. No "laminate" tag (use Wood*/Plastic*). Loaded with `blenderproc download cc_textures` plus `bproc.loader.load_ccmaterials` ([loader docs](https://dlr-rm.github.io/BlenderProc/blenderproc.api.loader.html)) |
| [Poly Haven textures](https://polyhaven.com/textures) | PBR textures | CC0 ([license](https://polyhaven.com/license)) | 871 [verified]: wood 138, plaster-concrete 137, fabric 43, metal 25, "indoor" 76. Loaded with `load_haven_mat` |
| [Poly Haven HDRIs](https://polyhaven.com/hdris) | Lighting / background | CC0 | 998 [verified]: **indoor 301**, artificial light 230, studio 96. Loaded with `get_random_world_background_hdr_img_path_from_haven` |
| Poly Haven API caveat | - | The [API page](https://polyhaven.com/our-api) says it is free for non-commercial and academic use and commercial use needs a license. The [corporate page](https://polyhaven.com/corporate) and [Public-API README](https://github.com/Poly-Haven/Public-API) say it is free for any purpose. ToS 3.2 bans scraping without permission. | Download once through the API (or `blenderproc download haven`) with a unique User-Agent, and cache locally. The CC0 assets themselves are unaffected. |

Suggested table-surface mix: wood/laminate 35%, white/grey plastic or melamine 20%, desk mats (leather, rubber,
fabric) 15%, fabric/tablecloth 10%, concrete/stone 10%, cardboard/paper/tiles 10% [inferred: matches typical
hobby desks].

## 3. Gripper and wrist-camera CAD

| Item | Source | License | Notes |
|---|---|---|---|
| SO-100/SO-101 arm, gripper, moving jaw | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | **Apache-2.0** ([LICENSE](https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/LICENSE)) | STEP and STL for `Moving_Jaw_SO101`, `Wrist_Roll_Follower_SO101`, and the SO100 `Moving_Jaw_08d`. `Simulation/SO101/` has URDF/MJCF, including **`so101_new_calib_camera.xml`** with a wrist camera mount and camera body (Hex-Nut 32x32 UVC form factor) [verified] |
| Official wrist-camera mounts | `Optional/` in the same repo [verified] | Apache-2.0 | `Wrist_Cam_Mount_32x32_UVC_Module` (SO100/SO101, integrated wrist roll), `SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module` (STEP+STL), `Wrist_Cam_Plug_Mount_32x32_UVC_Module` (SO100), `Wrist_Cam_Mount_Vinmooog_Webcam`, RealSense D405 and D435 holders |
| Compliant (Fin-Ray-style) gripper | `Optional/Compliant_Gripper` | Apache-2.0 | TPU 95A; same external geometry as the stock jaw |
| Alternative compliant finger | [horndeer/SO-ARM101-LeRobot](https://github.com/horndeer/SO-ARM101-LeRobot) | Apache-2.0 | TPU finger, PLA base, optional 3M grip tape |
| Robonine parallel gripper | [roboninecom/SO-ARM100-101-Parallel-Gripper](https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper) | **CC-BY-4.0** | 84 mm stroke; interchangeable camera holders (GC2093 2MP, IMX335 5MP, RealSense, Orbbec Gemini 2) |
| XLeRobot / LeKiwi (use SO-101 arms) | [XLeRobot](https://github.com/Vector-Wangel/XLeRobot), [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) | Apache-2.0 | TPU Fin-Ray fingers documented in the [XLeRobot docs](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html) |
| Koch v1.0 / v1.1 | [AlexanderKoch-Koch/low_cost_robot](https://github.com/AlexanderKoch-Koch/low_cost_robot) (MIT), [jess-moss/koch-v1-1](https://github.com/jess-moss/koch-v1-1) (Apache-2.0) | MIT / Apache-2.0 | Follower gripper: `Follower_Gripper_Moving_Part`, `Static_Part`. No official wrist-camera mount. |
| Moss v1 | [jess-moss/moss-robot-arms](https://github.com/jess-moss/moss-robot-arms) | Apache-2.0 | Being deprecated in favour of SO-100 |
| MuJoCo Menagerie SO-ARM100 | [trs_so_arm100](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100) | Apache-2.0 [verified] | Clean MJCF and meshes |
| Printables / Thingiverse / MakerWorld mounts | per item | Often the "Standard Digital File License" or CC BY-NC ([Printables ToS](https://www.prusa3d.com/page/printables-com-terms-of-service_231249/)) | **Do not use** unless tagged CC-BY or CC0. The Apache and CC-BY parts above already cover the shapes. |

**Print colors and materials.** Docs recommend PLA+ for structure and TPU 95A for compliant fingers
([SO-ARM100 README](https://github.com/TheRobotStudio/SO-ARM100)). The SO-101 MJCF uses printed-part
rgba(1, 0.82, 0.12), yellow, with black STS3215 servos and a black camera [verified]. The Seeed printed kit is a
white follower, black leader and grey clamps ([search result](https://rcdrone.top/products/so-arm101-low-cost-ai-arm-3d-printed-skeleton)).
Randomize among yellow, orange, white, black, grey, red and blue PLA (matte to satin, roughness 0.4-0.7, slight
layer-line normal map), plus translucent TPU fingertips (black, white or clear) [inferred].

## 4. Wrist-camera modules

| Module | Used where | Sensor / resolution | FOV | Lens | Source |
|---|---|---|---|---|---|
| **Innomaker U20CAM-1080P** (Amazon B0CNCSFQC1) | "Recommended model" in all three official 32x32 UVC wrist-mount READMEs [verified]; "any 32mm x 32mm USB camera module with min 720p / 30 fps" also works | 1/3", 1920x1080, **rolling shutter**; MJPEG 1080p/960p/720p, YUY2 1080p/720p @30 | **D 130 deg, H 106 deg** (manual table; the feature text says H 103 deg; the vendor page says D 120 / H 102) | **f = 2.8 mm, F2.3, TV distortion < -17%**, M12, CRA 14 deg | [U20CAM-1080P-S1 manual](https://github.com/INNO-MAKER/U20CAM-1080P-S1), [vendor](https://www.inno-maker.com/product/u20cam-1080p/) |
| Seeed RGB USB Camera 100 deg 720P (SKU 100079619) | Sold "for SO-ARM101"; Seeed reBot UVC32 bracket | OV9732, 1MP 1280x720 | 100 deg (diagonal) | manual focus; 38x38 mm board | [OpenELab](https://openelab.io/products/seeed-studio-rgb-usb-camera-100-degree-720p-ov9732-so-arm101), [datasheet](https://files.seeedstudio.com/Bazaar/product_pdf/100079619.pdf) |
| ET-S231 1080P (90 deg / 120 deg) | Seeed UVC32 wrist bracket | 1080p | 90 or 120 deg | M12 | [Seeed bracket](https://www.seeedstudio.com/reBot-UVC-32x32-Wrist-camera-bracket-kit-p-7043.html) |
| Arducam B0385 OV9782 | Community global-shutter option | 1MP 1280x800, global shutter, 100 fps MJPEG | 70 deg H, "low distortion" | M12 | [Arducam](https://www.arducam.com/product/100fps-global-shutter-color-usb-camera-board-1mp-ov9782-uvc-webcam-module-with-low-distortion-m12-lens-without-microphones-for-computer-laptop-android-device-and-raspberry-pi-arducam/) |
| Vinmooog webcam | SO-100 webcam mount | 1080p webcam | about 70-90 deg [inferred] | - | [mount README](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/Wrist_Cam_Mount_Vinmooog_Webcam) |
| RealSense D405 / D435 | Higher-end wrist mounts | D405 has factory intrinsics (pinhole + Brown) | ~87 deg | - | [SO-ARM100 Optional](https://github.com/TheRobotStudio/SO-ARM100) |

No public intrinsics or distortion coefficients exist for the UVC modules (none found). Calibrate a real unit with
OpenCV `fisheye.calibrate` (Kannala-Brandt).

**Modeling the U20CAM [inferred from the datasheet numbers].** A 1/3" 16:9 active area is about 5.23 x 2.94 mm
(6.0 mm diagonal).

| Model | HFOV | DFOV | Fits the datasheet? |
|---|---|---|---|
| Pinhole, f = 2.8 mm | 86 deg | 94 deg | No |
| Equidistant (r = f·θ) | **107 deg** | **123 deg** | Yes: matches H 106 / D 130 within tolerance, consistent with the -17% TV distortion |

At 1920 px width, f ≈ 1030 px. LeRobot usually records at 640x480, which is a scaled and possibly cropped mode, so
use per-mode intrinsics.

Plan:
1. Sample a Kannala-Brandt (equidistant + k1..k4) camera with f/width ≈ 0.50-0.56, about 1.0-1.15 times that for
   the 100-deg OV9732 class, and pinhole + Brown k1 for the 70-90 deg webcam/Arducam class.
2. Add rolling-shutter skew and MJPEG compression noise in augmentation.
3. Also emit undistorted / pinhole-rectified GT so the model can be trained in either space.

## 5. BlenderProc capabilities

| Topic | Finding | Source |
|---|---|---|
| Version / Blender | Latest is **2.8.0 (2024-10-22)**, which "upgrades to Blender 4.2.1". The repo is still maintained (commits Dec 2025 / Jan 2026). License **GPL-3.0**; the tool's license does not apply to renders. | [change_log](https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/change_log.md), [PyPI](https://pypi.org/project/blenderproc/), [repo](https://github.com/DLR-RM/BlenderProc) |
| Platforms | `InstallUtility.py` "only support[s] blender-4.2.1" and downloads `linux-x64`, `macos-x64`, `macos-arm64` or `windows-x64` [verified]. Blender ships **no Linux ARM64 builds** even in 5.1.x (only linux-x64, macos-arm64, windows-arm64/x64) [verified on download.blender.org]. On Linux aarch64 (for example Jetson or DGX Spark), use a self-built or distro Blender with `--custom-blender-path`; this is unsupported. | [InstallUtility](https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/utility/InstallUtility.py), [Blender 5.1 downloads](https://download.blender.org/release/Blender5.1/) |
| Intrinsics + distortion | `set_intrinsics_from_K_matrix`, then `bproc.camera.set_lens_distortion(k1,k2,k3,p1,p2)` (**Brown-Conrady, pinhole, zero skew**). This renders at a larger resolution, then `bproc.postprocessing.apply_lens_distortion` remaps colors (interpolated) and depth/normals/segmentation (not interpolated). The docs advise against k3/p1/p2. **No fisheye model.** | [LensDistortionUtility](https://dlr-rm.github.io/BlenderProc/blenderproc.python.camera.LensDistortionUtility.html), [example](https://dlr-rm.github.io/BlenderProc/examples/advanced/lens_distortion/README.html) |
| Native fisheye alternative | Cycles panoramic cameras include **Fisheye Equidistant** and **Fisheye Lens Polynomial**, θ = k0 + k1·r + ... + k4·r⁴ with r in sensor mm, set through `bpy` inside a BlenderProc script. Users report the k-values do not match Motion Tracker conventions, so fit them yourself. | [Blender 4.2 manual](https://docs.blender.org/manual/en/4.2/render/cycles/object_settings/cameras.html), [commit](https://developer.blender.org/rB24e0165), [devtalk](https://devtalk.blender.org/t/fisheye-lens-polynomial-and-lensfun-database/23457) |
| Recommendation [inferred] | At a 130 deg diagonal a pinhole intermediate needs about tan(65 deg)=2.1x half-width, so many wasted pixels at the edge. Option (a): native Cycles equidistant/polynomial fisheye, with a **ray-distance** GT. Option (b): render pinhole about 115 deg H at high resolution, remap with your own KB model (nearest-neighbour for depth), and accept a cropped DFOV. Validate both against a checkerboard. | - |
| Depth outputs | `enable_depth_output(activate_antialiasing=False)`: **z-depth** ("distance between the camera and the plane parallel to the camera"), non-antialiased EXR. `enable_distance_output`: **Euclidean ray distance**, antialiased via a mist pass with `antialiasing_distance_max`. `convert_to_depth` / `convert_to_distance` flags exist. Use z-depth with `activate_antialiasing=False` for metric GT; antialiased distance blends edges. | [renderer tutorial](https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/docs/tutorials/renderer.md), [API](https://dlr-rm.github.io/BlenderProc/blenderproc.api.renderer.html) |
| Other passes | `enable_normals_output`, `enable_segmentation_output(map_by='category_id' / 'instance' / ...)`, `enable_diffuse_color_output`; the BOP and COCO writers | [API](https://dlr-rm.github.io/BlenderProc/blenderproc.api.renderer.html) |
| Physics placement | `simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=40, ...)` uses Blender rigid body (Bullet). `sample_poses_on_surface` and `sample_poses` place objects without physics. | [object API](https://dlr-rm.github.io/BlenderProc/blenderproc.api.object.html) |
| Material randomization | `load_ccmaterials` (ambientCG), `load_haven_mat`, and direct Principled-BSDF edits through `bproc.material` / `bpy` | [loader API](https://dlr-rm.github.io/BlenderProc/blenderproc.api.loader.html) |
| Render devices / speed | `set_render_devices(use_only_cpu=False, desired_gpu_device_type=None)`, default order `["OPTIX","CUDA","HIP"]` on Linux. `set_denoiser("INTEL" / "OPTIX" / None)`, `set_noise_threshold` (0.1-0.001), `set_max_amount_of_samples`, `set_light_bounces`. There are no official benchmarks. Estimate: OptiX at 64-256 spp plus denoise, 640x480, is about 0.3-2 s/frame on an RTX 40-class GPU; CPU is 10-50x slower. Each scene also pays a Blender startup and asset-load cost, so render many frames per scene. Benchmark locally. | [API](https://dlr-rm.github.io/BlenderProc/blenderproc.api.renderer.html) |

## 6. Recommended starter asset set

| Layer | Assets | License | Attribution needed |
|---|---|---|---|
| Gripper and camera body | SO-ARM100 SO101 `Moving_Jaw`, `Wrist_Roll_Follower`, compliant jaw, Hex-Nut 32x32 UVC mount, the MJCF `wrist_camera_so101_v1` mesh; SO100 jaw; Koch v1.1 follower gripper; Robonine parallel gripper | Apache-2.0 / MIT / CC-BY-4.0 | Ship the Apache NOTICE; credit Robonine (CC-BY) |
| Hero objects | GSO subset of about 300 (bowls, saucers, mugs, porcelain, toys, boxes, bottles); YCB about 60; BOP HB 33 (CC0) + T-LESS 30 + LM 15 | CC-BY 4.0 / CC0 / MIT | Yes: cite GSO, YCB, BOP; list object IDs |
| PBR props | Poly Haven: tea_set_01, wooden_bowl_01/02, carved_wooden_plate, jug_01, food_* (19), stationery_supplies, office_notepads, vintage_stapler, clipboard, circuit_board, classic_laptop, wooden_cutting_board | CC0 | No (credit anyway) |
| Procedural | White and coloured ceramic plates, bowls, mugs; clear and tinted glasses; cutlery; USB, power and servo cables; Lego-like bricks; cubes / ArUco cube | Own (release CC0) | - |
| Long-tail diversity (optional) | Objaverse-LVIS categories (plate, bowl, mug, cup, wineglass, fruit, toy, stapler, pen, scissors, cable, remote...), filtered to `license in {by, cc0}` plus Objaverse++ quality, rescaled by category priors | CC-BY 4.0 / CC0 | Yes: per-object author + URL in the manifest |
| Surfaces | ambientCG Wood*, Plastic*, Leather*, Fabric*, Concrete*, Tiles*, Marble*, Paper*/Cardboard*; Poly Haven wood/fabric | CC0 | No |
| Lighting | Poly Haven indoor + artificial-light + studio HDRIs (about 300-400), plus random area/point lights (desk lamp, ceiling panel) | CC0 | No |

**Release-manifest rule:**
- Store `source`, `asset_id`, `license` and `attribution` for every asset used in every scene.
- Exclude anything tagged NC, SA (unless the dataset itself is released BY-SA), "Standard Digital File License",
  HOT3D or ABO.
- Release the dataset under **CC-BY 4.0**, since CC-BY inputs require attribution and CC0 cannot carry it.
