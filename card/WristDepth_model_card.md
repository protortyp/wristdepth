---
license: other
license_name: apple-amlr
license_link: LICENSE
pipeline_tag: depth-estimation
library_name: depth-pro-2
base_model: apple/DepthPro2-B
base_model_relation: finetune
datasets:
  - christianengel/WristDepth-procedural
tags:
  - depth-estimation
  - metric-depth
  - wrist-camera
  - robotics
  - lerobot
  - so-101
  - onnx
---

# WristDepth: metric depth for robot wrist cameras

Depth Pro 2 B fine-tuned for gripper cameras at 5 cm to 1.5 m. A frame and its focal length
in, depth in meters out. PyTorch weights plus ONNX graphs for ONNX Runtime and TensorRT.
Unofficial; not affiliated with or endorsed by Apple.

> **Research use only.** A Model Derivative under the [Apple Machine Learning Research Model
> License](LICENSE): non-commercial research and academic use only. Apple Machine Learning
> Research Model is licensed under the Apple Machine Learning Research Model License Agreement.
> Modified by fine-tuning on synthetic wrist-camera data.

![Four random unseen DROID wrist frames: RGB, stereo depth, stock and WristDepth prediction](droid_examples.png)

## Results

Stock Depth Pro 2 B puts wrist scenes 2-3x too far away. WristDepth cuts the error 6x on
unseen DROID scenes, with the depth shape intact on unseen scenes. Scale = median predicted / true depth (1.00 is
perfect); shape = AbsRel after per-frame scale alignment. No test frame was trained on; the
checkpoint was picked on SO-101 and DROID val, so those rows are optimistic.

| Test set | Model | Scale | AbsRel | delta1 | Shape |
| --- | --- | ---: | ---: | ---: | ---: |
| SO-101 | Depth Pro 2 B | 2.21 | 1.530 | 0.00 | 0.068 |
| SO-101 | **WristDepth** | 1.09 | 0.224 | 0.69 | 0.087 |
| DROID val | Depth Pro 2 B | 1.52 | 0.888 | 0.24 | 0.179 |
| DROID val | **WristDepth** | 0.71 | 0.316 | 0.36 | 0.167 |
| DROID test | Depth Pro 2 B | 2.27 | 1.445 | 0.05 | 0.103 |
| DROID test | **WristDepth** | 1.00 | 0.255 | 0.54 | 0.091 |
| RoboLab (sim) | Depth Pro 2 B | 2.92 | 2.135 | 0.06 | 0.267 |
| RoboLab (sim) | **WristDepth** | 1.11 | 0.314 | 0.48 | 0.162 |

- **SO-101**: one desk, 40 frames at 13-32 cm; depth from kinematics, checked against a board.
- **DROID**: 785 SpatialBench wrist frames (Franka + Robotiq, many labs, stereo depth), split by
  scene into val and test.
- **RoboLab**: 748 simulated wrist frames (another robot and renderer).

## How to use

PyTorch, with Apple's package:

```bash
pip install "git+https://github.com/apple-aiml-research/ml-depth-pro-2"
```

```python
from functools import partial

import torch
from depth_pro_2 import DepthPro2, load_rgb
from depth_pro_2.model import to_normalized_tensor

model = DepthPro2.from_pretrained("christianengel/WristDepth", precision=torch.bfloat16)
model = model.to("cuda").eval()
transform = partial(to_normalized_tensor, device=torch.device("cuda"))

image, _ = load_rgb("wrist.jpg")
depth = model.infer(transform(image), f_px=450.0)["depth"]  # meters; f_px = your camera's fx
```

ONNX: [`example.py`](example.py). Graphs take `pixel_values` (RGB in [0, 1], `(1, 3, 480, 640)`)
and `f_px` (horizontal focal length of that image) and return `depth_meters`. Undistorted
frames and raw fisheye frames (with the centre focal length) both work. Parity against the
PyTorch model in FP32:

| File | Mode | Precision | MB | AbsRel | delta1 |
| --- | --- | --- | ---: | ---: | ---: |
| `onnx/model.onnx` | single | FP32 | 387 | 0.0002 | 1.0000 |
| `onnx/model_fp16.onnx` | single | FP16 | 199 | 0.0004 | 1.0000 |
| `onnx/model_int8.onnx` | single | INT8 | 143 | 0.0037 | 1.0000 |
| `onnx/model_clip4.onnx` | clip of 4 | FP32 | 389 | 0.0002 | 1.0000 |
| `onnx/model_clip4_fp16.onnx` | clip of 4 | FP16 | 201 | 0.0002 | 1.0000 |
| `onnx/model_stream4.onnx` | stream, window 4 | FP32 | 387 | 0.0002 | 1.0000 |
| `onnx/model_stream4_fp16.onnx` | stream, window 4 | FP16 | 199 | 0.0004 | 1.0000 |

TensorRT: `./build_tensorrt.sh onnx/model_fp16.onnx`; the engine passes the same gate (AbsRel
0.0009). Speed equals [DepthPro2-B-ONNX](https://huggingface.co/christianengel/DepthPro2-B-ONNX).

## Limitations

- Scale still varies by setup (1.09x on SO-101, 0.71x on DROID val); views beyond 50 cm
  read too close. For best accuracy, calibrate one scale factor per camera from a few frames at
  a known distance.
- Trained with the SO-101 gripper in view; other grippers work but were not trained on.

## Training

- **Data**: 4,000 frames sampled from 36,464 procedural BlenderProc renders
  ([WristDepth-procedural](https://huggingface.co/datasets/christianengel/WristDepth-procedural)):
  random tables, rooms, objects, prints and lights, seen through the SO-101 gripper. Pose, lens,
  finger colour and mount follow a census of ~7,000 community LeRobot wrist datasets.
- **Shape**: self-distillation from the frozen stock model on unlabelled community wrist frames
  (scale-invariant loss).
- **Recipe**: decoder, head and last 4 encoder blocks; log-depth L1 + gradient matching; AdamW,
  one-cycle LR 1e-4 (encoder 1e-5); weight EMA.

Code: [protortyp/wristdepth](https://github.com/protortyp/wristdepth) (Apache-2.0).

## Citation

```bibtex
@article{DepthPro2:2026:arxiv,
  title   = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author  = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong and Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal = {arXiv preprint arXiv:2611.26603},
  year    = {2026},
}
```
