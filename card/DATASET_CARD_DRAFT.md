---
license: cc-by-4.0
task_categories:
  - depth-estimation
tags:
  - robotics
  - wrist-camera
  - synthetic
  - lerobot
  - so-101
size_categories:
  - 10K<n<100K
---

# {{name}}: procedural wrist-camera frames with exact depth

{{n_frames}} synthetic frames from a gripper-mounted camera, each with exact z-depth, a
category mask and intrinsics. Built to fine-tune metric depth models for robot wrist cameras;
see [{{model}}]({{model_url}}).

## Contents

Per frame (`.npz`): `rgb` (480x640 uint8), `depth` (float32 meters along the optical axis, 0 =
unlabelled), `seg` (1 gripper, 2 table and flat items, 3 objects, 4 room), `K`, and the camera
pose and jaw opening in `meta`. `raw/` is the pinhole render; `wrist/` adds the wrist-camera
look (fisheye, exposure, blur, JPEG) with `f_px` the centre focal length.

## How it is made

BlenderProc/Cycles. Random tables, rooms, objects, procedural dishware, flat prints and lights,
seen through the SO-101 gripper (CAD fitted to real finger photos). Camera pose (5 cm-1.5 m),
lens, finger colour and mount follow a census of ~7,000 community LeRobot wrist datasets.
Depth labels land within 0.15 mm of the table plane (median).

## License and attribution

Renders: CC-BY-4.0. Assets: CC0 ([Poly Haven](https://polyhaven.com/license),
[ambientCG](https://docs.ambientcg.com/license/)); gripper CAD from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
Generator: {{github}} (Apache-2.0).
