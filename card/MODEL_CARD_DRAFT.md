---
license: apple-amlr
license_link: LICENSE
library_name: onnx
pipeline_tag: depth-estimation
base_model: {{base_model}}
base_model_relation: finetune
tags:
  - depth-estimation
  - metric-depth
  - wrist-camera
  - robotics
  - lerobot
  - so-101
  - onnx
---

# {{name}}: metric depth for robot wrist cameras

Depth Pro 2 {{size}} fine-tuned for gripper cameras at 5 cm to 1.5 m: a frame and the focal
length `f_px` in, depth in meters out. Unofficial; not affiliated with or endorsed by Apple.

> **Research use only.** A Model Derivative under the [Apple Machine Learning Research Model
> License](LICENSE): non-commercial research and academic use only. Apple Machine Learning
> Research Model is licensed under the Apple Machine Learning Research Model License Agreement.
> Modified by fine-tuning on synthetic wrist-camera data.

## Results

Stock Depth Pro 2 {{size}} puts wrist scenes 2-3x too far away. {{name}} cuts the error
{{gain}}x on unseen DROID scenes and keeps the depth shape. Scale = median predicted / true
depth (1.00 is perfect); shape = AbsRel after per-frame scale alignment. None of these frames
were trained on; the checkpoint was picked on SO-101 and DROID val, so those rows are
optimistic.

![Four random DROID test frames: RGB, stereo depth, stock and fine-tuned prediction](droid_examples.png)

| Test set | Model | Scale | AbsRel | delta1 | Shape |
| --- | --- | ---: | ---: | ---: | ---: |
{{results_table}}

- **SO-101**: one desk, 40 frames at 13-32 cm; depth from kinematics, checked against a board.
- **DROID**: 785 SpatialBench wrist frames (Franka + Robotiq, many labs, stereo depth), split by
  scene into val and test.
- **RoboLab**: 748 simulated wrist frames (another robot and renderer).

## Limitations

- Scale still varies by setup ({{so101_scale}}x on SO-101, {{droid_val_scale}}x on DROID val);
  views beyond 50 cm read too close. For best accuracy, calibrate one scale factor per camera
  from a few frames at a known distance.
- Trained with the SO-101 gripper in view; other grippers work but were not trained on.

## Use

Same I/O as [DepthPro2-{{size}}-ONNX](https://huggingface.co/christianengel/DepthPro2-{{size}}-ONNX):
`pixel_values` RGB in [0, 1], `(1, 3, 480, 640)`; `f_px` the horizontal focal length of that
image. Undistorted or raw fisheye frames (with the centre focal length) both work.

Parity against the fine-tuned PyTorch model (FP32):

| File | Mode | Precision | MB | AbsRel | delta1 |
| --- | --- | --- | ---: | ---: | ---: |
{{files_table}}

TensorRT: `./build_tensorrt.sh onnx/model_fp16.onnx`; the engine passes the same gate (AbsRel
{{trt_abs_rel}}). Speed equals [DepthPro2-{{size}}-ONNX](https://huggingface.co/christianengel/DepthPro2-{{size}}-ONNX).

## Training

- **Data**: {{train_frames}} frames sampled from {{n_frames}} procedural BlenderProc renders:
  random tables, rooms, objects, prints and lights, seen through the SO-101 gripper (CAD fitted
  to real finger photos). Pose, lens, finger colour and mount follow a census of 7,000 community
  LeRobot wrist datasets. Assets: CC0 (Poly Haven, ambientCG), Apache-2.0 (SO-ARM100 CAD).
- **Shape**: self-distillation from the frozen stock model on unlabelled community wrist frames
  (scale-invariant loss).
- **Recipe**: decoder, head and last 4 encoder blocks; log-depth L1 + gradient matching; AdamW,
  one-cycle LR 1e-4 (encoder 1e-5); weight EMA.

Code: {{github}} (Apache-2.0).

## Citation

```bibtex
@article{DepthPro2:2026:arxiv,
  title   = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author  = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong and Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal = {arXiv preprint arXiv:2611.26603},
  year    = {2026},
}
```
