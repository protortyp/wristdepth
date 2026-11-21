# 01: Census of public LeRobot wrist-camera datasets (Hugging Face)

Date: 2026-11-20. Raw table: `$WRIST_DEPTH_DATA/research/census/census.json`
(per-dataset metadata plus visual annotations for the 60 sampled datasets).
Frames: `$WRIST_DEPTH_DATA/research/census/frames/` (360 jpg).
Contact sheets: `$WRIST_DEPTH_DATA/research/census/sheets/sheet_00..11.jpg`.
Scripts: `$WRIST_DEPTH_DATA/research/census/work/` (`search.py`, `info.py`,
`parse.py`, `grab.py`, `sheet.py`, `ann.py`, `final.py`).

## Method

- HF dataset search (`so101, so100, so-101, so-100, so_101, so_100, lerobot, koch, wrist,
  gripper, piper, lekiwi`, paginated): 26,319 repos, 19,396 tagged `LeRobot`.
- `meta/info.json` fetched for all 19,396: 18,837 parsed, 562 failed (404/403 gated/401).
- Wrist camera = video/image feature whose last key part matches
  `wrist|gripper|hand|ego|grip|eye_in|in_hand|eef|end_effector|arm_cam`.
- "Low-cost arm" = robot_type matches `so|koch|lekiwi|omx|xlerobot|moss`.
- Sim by name = id or robot_type matches `sim|mujoco|isaac|leisaac|genesis|maniskill|libero|gym|...`.
- Visual sample: 60 real low-cost datasets, top by downloads, max 1 per owner;
  6 frames each at 4/20/36/52/68/84 % of the first wrist video file, hand-annotated.

## Totals

| Set | Datasets | Owners | Episodes | Frames | Median eps |
|---|---:|---:|---:|---:|---:|
| All LeRobot with a wrist cam | 9,336 | 1,713 | 2.19 M | 1.48 B | 20 |
| Low-cost arms (SO/Koch/LeKiwi/OMX) | 7,341 | 1,422 | 622 k | 338 M | 15 |
| Low-cost, real (not sim by name) | **6,982** | 1,378 | **436 k** | **218 M** | 10 |

- Sim share (by name), low-cost: 359 / 7,341 datasets (4.9 %) but **120 M / 338 M frames (36 %)**;
  sim sets are few and large. The name filter undercounts: 2 of 60 "real" visual samples
  were renders (`gpudad/so101_pick_cube_chunked`, `szk1ck/so101-ycb-pickplace`),
  plus 1 more that looks rendered (`PersonaAIDev/so101_test_0113_mimic_3`).
- Frame totals are inflated by re-uploads/merges (268 "merged/mix" sets = 24 M frames;
  `LeeAfe/so101-megamix-v1` and `whosricky/so101-megamix-v1` are identical). Long tail:
  median 7.4 k frames per dataset, top 1 % of datasets hold 35 % of frames.
- Robot types (real, low-cost): so_follower 3,379, so101_follower 1,573, lekiwi(_client) 692,
  so100(_follower) 623, so101 219, bi_so_follower 151, omx 34, koch < 30.
- Wrist cams per dataset: 1 = 95.5 %, 2 (bimanual) = 4.4 %. Key name: `wrist` 76 %,
  `left_wrist` 5 %, `gripper` 5 %, `handeye` 3 %, `hand` 2 %.

## Format distribution (6,982 real low-cost)

| Resolution | n | % |  | FPS | n | % |  | Codec | n | % |
|---|---:|---:|---|---|---:|---:|---|---|---:|---:|
| 640x480 | 5,934 | 85.0 | | 30 | 6,131 | 87.8 | | av1 | 5,693 | 81.5 |
| 480x640 (portrait) | 306 | 4.4 | | 10 | 301 | 4.3 | | h264 | 799 | 11.4 |
| 1280x720 | 292 | 4.2 | | 20 | 172 | 2.5 | | unset | 481 | 6.9 |
| 640x360 | 142 | 2.0 | | 15 | 153 | 2.2 | | hevc/mp4v | 7 | 0.1 |
| 1920x1080 | 126 | 1.8 | | 25 | 108 | 1.5 | | | | |
| 320x240 | 92 | 1.3 | | 5 | 66 | 0.9 | | | | |

Codebase: v3.0 79 %, v2.1 20 %, v2.0 1 %. 99.5 % stored as video (yuv420p).

## Visual sample (60 datasets, 360 frames)

Validity: 51 real wrist views, **6 mislabeled** (a static overhead/front cam stored under
the wrist key: `kmg0620/so101_bomb_pick_place_v1`, `pranavsaroha/so100_medicine_0602`,
`Kwj1021/so101_pick_place_20261024_143100`, `ricky0526/so101_pick_mouse_to_box_v1`,
`apayan/so100_ball8`, `miketangmk/so101_table_sort_v2`), 3 rendered. One more
(`Qiu-Xinchuan/so101_total_1.0`) mixes a static cam into the wrist file. Any use of
community data for eval must filter for "fingers visible".

Percentages below are of the 51 valid wrist datasets.

### Finger colour

| Colour | n | % | Examples |
|---|---:|---:|---|
| White / ivory | 17 | 33 | lerobot/svla_so100_pickplace, shilpitha-03/so101_cube_in_box_20261030_164459, IlyaGusev/so101_pick_gum, LeeAfe/so101-megamix-v1, ravediamond/so101-bowls-policy-rollouts |
| Black | 15 | 29 | armnet/armnetbench_v01_lerobot_so101, arash2060/so101-generalist-pretrain-v1, Cache-SCA/SO101-teleop_stack_RGBblock_on_bluedish_100epi_more, hangwu/koch_pick_terminal, Zekai-Chen/bi_so101_fold_towel_4cam |
| Orange / salmon | 5 | 10 | youliangtan/so101-table-cleanup, 5hadytru/so101_bench_real, legalaspro/so101-greenblack-cube-cup-pnp-30hz, maximilianofir/so101-left-rack-balanced-1000-dr-speed05-1-20261005 |
| Purple | 4 | 8 | geonmin-kim/SO101-lv4-3color-cube-mat-to-mat-no-human-reset-v1, DmitryStrog/so101_pnp_merged_500, hi-space/SO-ARM101-PICK-BANANA |
| Grey | 3 | 6 | fport/tic-tac-toe-so101-x-round1, digas99/so101_tray_train_v3, Elvinky/bi-so101-insert-screw-562ep |
| Other saturated (teal 2, green 2, yellow, blue translucent, olive) | 7 | 14 | hungho77/so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube, mkm16207/so101_unplug_ethernet_20261022_025009_20261022_025735, jccj/so100_block_in_cup, hjkso1406/so101-4tasks-100eps, SalZa2004/so101_track_rails_depth_2 |

Shape: 46/51 (90 %) are the stock SO-100/101 3D-printed jaw (tapered, ridged/holed
PLA, often with visible print texture). Non-stock: Koch gripper (large dark body),
OMX gripper, custom lattice fingers (`vraiRobotLab/...`), custom purple-tipped gripper with
black screw body (`DmitryStrog`, `nbirukov`, same rig).

### Fingers visible and mount

| Category | n | % | Examples |
|---|---:|---:|---|
| Both tips visible, bottom of frame (camera above jaw, looking down along fingers) | 44 | 86 | most; e.g. shilpitha-03/..., LeeAfe/so101-megamix-v1, lsalomme/so101-d1v2-cube-pick-place |
| Fingers at top of frame (inverted image / camera under jaw) | 5 | 10 | lerobot/svla_so100_pickplace, LUOSYrrrrr/so101_battery_insertion_v2, est4ever/so101_blue_left_20261025_172708, tttonyalpha/so101_pick_cube_300_task1, Elvinky/bi-so101-insert-screw-562ep |
| Side mount, one jaw side-on at an edge | 2 | 4 | cn0303/so101-strawberry-pluck-v1.1-20260601-1304, KosukeHaketa/so101_pickplace_demo (portrait) |
| Only one jaw clearly visible | 4 | 8 | svla_so100_pickplace, cn0303/..., KosukeHaketa/..., aaronsu11/so100_instrument |
| Only gripper body, no tips | 1 | 2 | Elvinky/bi-so101-insert-screw-562ep |

Typical finger footprint: two tapered jaws entering from the bottom corners, converging
toward the centre, covering roughly 10-40 % of the image (largest on Koch/yellow/custom
grippers). Jaw opening varies from closed to fully open within episodes.
Portrait 480x640 datasets (4.4 % of the census) imply a 90° rotated mount.

### Lens / FOV

| Distortion | n | % | Examples |
|---|---:|---:|---|
| Low (straight lines straight, ~60-75° HFOV) | 5 | 10 | arash2060/so101-generalist-pretrain-v1, vraiRobotLab/so101-pick-and-place-hand-cartesian, aaronsu11/so100_instrument, SalZa2004/so101_track_rails_depth_2 |
| Moderate barrel (~85-110°) | 30 | 59 | armnet/armnetbench_v01_lerobot_so101, shilpitha-03/..., youliangtan/so101-table-cleanup, LeeAfe/so101-megamix-v1, HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1 |
| Strong fisheye (>120°, curved table edges, often dark corner vignette) | 16 | 31 | geonmin-kim/SO101-lv4-..., IlyaGusev/so101_pick_gum, DmitryStrog/so101_pnp_merged_500, LUOSYrrrrr/so101_battery_insertion_v2, Jiamo0912/so101-cube-drawer |

FOV estimates are by eye; no intrinsics are published in any info.json.

### Scene

| Table surface | n | % | Examples |
|---|---:|---:|---|
| White / light grey (incl. 2 glossy) | 19 | 37 | geonmin-kim/..., VoicAndrei/so100_cubes_noaptea_companiilor_2, Project-IRA/TPSoSe2026_..., RoMALab/so101_stacking_without_orange_blue (glossy) |
| Wood, light (incl. salmon-tinted) | 13 | 25 | shilpitha-03/..., IlyaGusev/so101_pick_gum, Cache-SCA/..., digas99/so101_tray_train_v3 |
| Wood, dark / floor | 4 | 8 | HashtagRobotics/..., Jiamo0912/so101-cube-drawer, RoboColosseum/so101-upright-bottle |
| Dark or mid-grey mat (some with printed logos) | 10 | 20 | LeeAfe/so101-megamix-v1, lsalomme/..., armnet/... (printed), legalaspro/... (printed), hangwu/koch_pick_terminal |
| Patterned (checkerboard, green cutting mat grid) | 3 | 6 | DmitryStrog/..., nbirukov/so101_pnp_merged2, ravediamond/so101-bowls-policy-rollouts |
| Dark/unlit | 2 | 4 | cagataydev/so101-tic-tac-toe, 5hadytru/so101_bench_real |

Clutter: low 28 (55 %), medium 19 (37 %), high 4 (8 %: home desks with cables, monitors,
stationery). Objects are mostly coloured cubes, cups, bowls; transparent or glossy items
(bottles, acrylic boxes, plastic drawers, mugs) appear in 8/51 (16 %). Human hands in view:
2/51. Background beyond the table edge (walls, monitors, floor, chairs) is visible in
roughly half of the datasets whenever the camera tilts up.

### Image quality

| Lighting | n | % |  | Motion blur | n | % |
|---|---:|---:|---|---|---:|---:|
| Even indoor | 38 | 75 | | Low | 41 | 80 |
| Overexposed / washed out | 5 | 10 | | Moderate | 4 | 8 |
| Hard arm shadows | 4 | 8 | | Heavy (in some frames) | 6 | 12 |
| Specular hotspots (glossy table) | 3 | 6 | | | | |
| Underexposed | 1 | 2 | | | | |

Per-frame stats over 306 valid frames: mean luma p10/p50/p90 = 96/130/160; 16 % of
frames have > 5 % clipped pixels; 20 % have Laplacian variance < 30 (blurry). Objects
inside the jaws (< 5 cm) are often defocused (fixed-focus webcams). AV1 at 640x480 gives
visible blocking in flat regions.

## Implications for synthetic data (priority order)

1. **Lens model**: render 4:3 640x480 with barrel distortion spanning low to strong
   fisheye (target mix ~10/60/30 %), HFOV ~70-140°, optional circular vignette. Also
   emit 180° flipped (10 %) and 90° rotated (4 %) variants. Depth labels must be
   distorted with the same model.
2. **Finger geometry and pose**: stock SO-100/101 jaw meshes entering from the bottom,
   covering 10-40 % of the frame, jaw opening randomized, camera offset/tilt jittered per
   mount; add Koch/OMX/custom gripper variants at low rate. Fingers are the nearest
   surface (2-8 cm) and must get exact depth.
3. **Finger material**: colour randomized with weights ~33 % white, 35 % black/grey,
   10 % orange, 8 % purple, 14 % other saturated (incl. translucent); matte PLA with
   print-layer texture and holes.
4. **Table surface**: white/light (37 %), wood (33 %), dark/grey mats incl. printed logos
   (20 %), high-contrast patterns (6 %); include glossy tables with specular hotspots.
5. **Exposure and sensor**: auto-exposure errors (over- and under-exposure), hard
   shadows cast by the arm, motion blur on 10-20 % of frames, near-range defocus,
   webcam noise, AV1/H.264 compression at 640x480.
6. **Objects and range**: small coloured cubes/cups/bowls at 3-60 cm, objects held
   inside the jaws at < 5 cm; include transparent and glossy objects (~15 %).
7. **Background**: walls, monitors, cables, chairs and floor visible beyond the table
   edge at > 60 cm; occasional human hands.

## Caveats

- Downloads are bot-inflated; "top by downloads" is not "most representative".
- Visual sample is 1 dataset per owner, 6 frames from the first video file only.
- Wrist key detection is name-based; mislabeled keys (~10 % in the sample) are counted
  as wrist in the metadata totals.
