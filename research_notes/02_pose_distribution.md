# 02 Wrist-camera pose distribution in community SO-100/101 teleoperation

Date: 2026-11-20. Data, plots and scripts: `$WRIST_DEPTH_DATA/research/poses/`
(`out/results.json`, `out/datasets.json`, `out/*.png`, `out_allframes/` = same without the
rest-pose filter, `fk/` per-dataset FK arrays, `scripts/`, `meta/` HF search + info/stats).

## Bottom line

- Real teleop camera-to-table distance along the optical axis (active frames, equal weight per
  dataset): **median 18.3 cm, p10 10.4, p25 13.1, p75 28.4, p90 57**. Pick-place only: median 17.0,
  p90 37. Per-dataset medians range 12.5 to 30.5 cm (p10 to p90 over datasets).
- The pilot synth median (12.8 cm) sits at the community p25; our rig's real eval (21.7 cm) is at
  about the community p60. Both lie inside the real range, so the fix is to widen and reshape the
  synthetic distance prior, not to move it to one value.
- Camera pitch from straight down: **median 34 deg, p10 15, p90 77**. Pick-place: median 33,
  p90 59. About 20% of all active frames (10% of pick-place) are >60 deg, which the current
  sampler (`axis[2] > -0.5` rejects pitch >60 deg) never produces.
- Hard floor: with this mount the camera sits about 9 cm above and behind the fingertip, so the
  axis distance never drops below about 8 cm, even with the fingers on the table.

## Method

1. HF search (`so101`, `so100`, `so-101`, `so_101`): 2017 datasets. Took the 219 most downloaded
   that are not sim, eval or rollout by name, then read `meta/info.json` and `meta/stats.json`. 173
   have a 6-D `observation.state` named `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
   wrist_roll, gripper` (`.pos` suffix). Picked 44 real teleop sets, at most one per owner/task
   family and no duplicates. Each set: up to 12 evenly spaced v3 data files, or 15 v2.1 episodes.
   Every 10th frame (sets over 300k frames: every n/30000-th).
2. Units: LeRobot clips `RANGE_M100_100` to ±100 exactly, so any arm joint above 100.5 means
   `DEGREES`. Sets within ±100 were scored both ways. 26 sets are degrees and 10 are normalized
   (mapped linearly onto the model joint range). Gripper is always 0-100, mapped to the model's
   -10..100 deg.
3. FK: `so101_new_calib.xml`, `qpos[:6]`, `mj_kinematics`. Reads the `gripper` body, the
   `gripperframe` site (fingertip) and the camera from `T_gripper_from_cam_cv` (`calib/real/wrist.json`).
   Our `joint_offsets_rad` map our own servo zero, not LeRobot's, so they are **not** applied.
4. Convention check: the scripts tried 16 hypotheses ({deg, norm} × sign flips of lift, elbow
   and wrist_flex). Penalty = share of frames with the tip below -2 cm or a link below z=0, plus
   tip pointing up in the lowest 10% of frames, plus a 5th-percentile tip height outside [-2, 4] cm
   (grasps touch the table).
   - **The raw model convention (no flips) fits LeRobot:** 36 of 44 sets pass with penalty ≤0.06.
     Their tip p5 lies between -1.3 and +2.5 cm, which is fingers on the table.
   - All 7 "deg?" sets that failed in degrees pass as normalized with no flip.
   - The 8 failing sets were excluded, not "fixed". A single sign flip would fix some of them,
     but there is no physical reason for one: IlyaGusev pick_gum, RoboColosseum stack-cubes and
     upright-bottle, SalZa2004 track_rails, dmowns pick_place_items, kmg0620 bomb_pick_place,
     physicalairi 1200ep, shilpitha-03 cube_in_box. The last three have a tip p5 of -12 to -20 cm,
     so they use a different calibration.
5. Table plane: z=0 in the base frame. The per-dataset 2nd percentile of tip height (median
   -1.0 cm, min -2.4) suggests many robots sit 1-2 cm above the desk on a plate or clamp.
   `*_t` metrics, measured against that estimated plane, come out about 1 cm larger.
6. Rest frames: shoulder_lift < -80 deg and elbow > 75 deg, 19% of frames on average. They are
   dropped in `out/` and kept in `out_allframes/`. The svla_so100 home pose (-87/75 deg) is just
   missed by this filter. It causes the spike at (17 cm, 56 deg) in the 2D plot.
7. Aggregation gives every dataset equal weight (4000 resampled values each), so 5hadytru (2.5M
   frames) does not dominate.

## Distributions (active frames, equal weight per dataset; p10 / p25 / p50 / p75 / p90)

| quantity | all (36 sets) | pick-place (28) | other (8) |
|---|---|---|---|
| gripper body origin height (cm, mount independent) | 9.1 / 10.9 / 13.9 / 17.8 / 21.9 | 9.2 / 11.0 / 13.9 / 17.5 / 21.6 | 8.8 / 10.6 / 14.2 / 19.4 / 22.8 |
| fingertip height (cm, mount independent) | 0.1 / 2.3 / 6.0 / 10.7 / 15.8 | -0.2 / 1.8 / 5.2 / 9.9 / 15.0 | 1.9 / 4.7 / 8.2 / 13.6 / 17.8 |
| camera height (cm) | 8.3 / 10.3 / 13.7 / 18.8 / 23.3 | 8.8 / 10.5 / 13.7 / 18.4 / 22.9 | 4.8 / 9.3 / 13.4 / 20.6 / 24.5 |
| camera-to-table distance along the optical axis (cm) | 10.4 / 13.1 / 18.3 / 28.4 / 57 | 10.2 / 12.6 / 17.0 / 24.2 / 36.9 | 12.4 / 16.9 / 28.3 / 68 / 151 |
| pitch from straight down (deg) | 15 / 26 / 34 / 53 / 77 | 14 / 25 / 33 / 47 / 59 | 21 / 28 / 53 / 85 / 101 |
| gripper opening (0 = closed, of 100) | 2 / 6 / 14 / 24 / 34 | 2 / 6 / 14 / 23 / 32 | 2 / 6 / 15 / 28 / 39 |

- The optical axis misses the table plane (pitch ≥90 deg) in 4.4% of frames.
- Frame-weighted values (no per-dataset balancing, rest frames included): median distance 15.9 cm,
  pitch 34 deg, camera height 11.9 cm.
- **Time near the table (fingertip height):**
  - all sets: <5 cm 44%, <10 cm 72%, >20 cm 3%.
  - pick-place: <5 cm 49%, <10 cm 76%.
  - other: <5 cm 27%, <10 cm 61%.
- SO-101 is mostly operated low: transport is a short, low hop. The gripper body is above 10 cm
  83% of the time only because it sits about 8-10 cm above the fingertip.
- **"Other" tasks** (insertion, unplugging, button pressing, barcode scanning) have the far,
  oblique tail. The gripper is near horizontal, with pitch 70-100 deg and distances of 50 cm or
  more, often running into the background. Pick-place stays compact (p90 37 cm, 59 deg).

Pitch conditional on distance (all sets, active frames):

| distance bin (cm) | weight | pitch p10 / p50 / p90 (deg) |
|---|---|---|
| <10 | 7.6% | 15 / 28 / 46 |
| 10-15 | 26.8% | 13 / 28 / 45 |
| 15-20 | 22.2% | 15 / 32 / 56 |
| 20-25 | 12.9% | 14 / 36 / 59 |
| 25-30 | 7.7% | 15 / 39 / 65 |
| 30-40 | 7.9% | 24 / 48 / 77 |
| 40-60 | 5.7% | 36 / 63 / 93 |
| >60 | 9.3% | 51 / 81 / 104 |

## Caveats

- **Mount dependence.** All camera numbers use our wrist.json mount: camera about 6.6 cm off
  the roll axis, optical axis about 27 deg from the gripper axis. Community mounts differ, both
  the official SO-101 wrist mount and the many printed variants. The gripper and fingertip
  heights do not depend on the mount.
- **Wrist roll.** Roll is user-calibrated, but it matters little for a top-down gripper. Sweeping
  roll through 360 deg at a fixed grasp pose moves pitch between 17 and 37 deg and camera height
  by only ±1 cm. For a horizontal gripper, roll decides whether the camera looks 25-30 deg down or
  up, which is part of the >90 deg tail.
- **Kinematics.** SO-100 sets (svla_so100_*) run through the SO-101 model, which is a close but
  not identical match.
- **Normalized units.** Normalized sets assume the user's calibrated range equals the model's
  range. They still pass the plausibility check.
- **Sampling.** We read only the first v3 file of the very large sets (armnet, 5hadytru, Project-IRA,
  aailabkaist). The scripts work with episodes, not demos, and the task category comes from the
  dataset name.

## Recommended synthetic sampling (replaces `uniform(0.05, 0.42)` + pitch <60 deg)

Draw the distance d along the optical axis to the desk, then pitch θ given d, then solve the
joints as now with ±1.5 cm / ±5 deg tolerance. Use a mixture: about 70% matches real teleop and
30% is flat, so the network sees real-looking views without learning a sharp distance prior.

| component | weight | d (cm) | θ (deg) |
|---|---|---|---|
| grasp / close | 0.30 | U(8, 15) | N(28, 10) clipped to [5, 55] |
| approach / hover | 0.25 | U(15, 25) | N(33, 13) clipped to [5, 65] |
| transport / overview | 0.10 | U(25, 40) | N(45, 15) clipped to [10, 80] |
| oblique / far | 0.05 | U(40, 70) | U(50, 85) |
| flat prior | 0.30 | log-uniform(8, 60) | U(5, 70) |

- The mixture gives distance p10 / p25 / p50 / p75 / p90 = 9.7 / 12.3 / 18.1 / 27.4 / 42 cm (simulated). That
  matches the real community median of 18 cm and keeps our rig (21.7 cm) near the middle.
- Lower `d` to 8 cm and drop anything below 7 cm. Shorter distances are physically impossible
  with this mount, so they are wasted samples.
- Allow pitch up to 85 deg: replace `axis[2] > -0.5` with `axis[2] > -0.09`. Keep views beyond
  85 deg out, because their depth is mostly background, not desk.
- Wrist roll: centre it on the mount's upright pose, ±45 deg. Real roll is user-calibrated and
  only changes pitch by about ±10 deg at top-down poses.
- Gripper opening: Beta-like, skewed toward closed or slightly open (median 14 of 100, p90 34).
  The current sprite range of 0-1.48 rad over-represents wide-open jaws.
- For a pick-place-only model, drop the far component and give its weight to grasp/close.
- Re-check after generation: histogram the synth `d` and θ against `out/hist2d_dist_pitch.png`.
