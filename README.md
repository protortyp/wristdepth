# WristDepth

![WristDepth on an SO-101 wrist camera: RGB left of the slider, predicted depth right](assets/wristdepth_slider_center.png)

<sub>[Video](https://huggingface.co/christianengel/WristDepth/blob/main/assets/wristdepth_slider.mp4) (made with [`assets/make_slider_video.py`](assets/make_slider_video.py)). Source clip: [legalaspro/so101-greenblack-cube-cup-pnp-30hz](https://huggingface.co/datasets/legalaspro/so101-greenblack-cube-cup-pnp-30hz) (Apache-2.0).</sub>

Metric depth for robot wrist cameras. Model: [christianengel/WristDepth](https://huggingface.co/christianengel/WristDepth)
(PyTorch + ONNX). Data: [christianengel/WristDepth-procedural](https://huggingface.co/datasets/christianengel/WristDepth-procedural).

Stock monocular depth models misjudge gripper views by 2-4x; this repo renders procedural wrist-camera data and fine-tunes
[Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2) on it, cutting the error
about 6x on unseen DROID scenes while keeping the depth shape.

| Held-out set | Depth Pro 2 B (scale / AbsRel) | Fine-tuned B (scale / AbsRel) |
| --- | --- | --- |
| DROID wrist test (392 frames, other labs) | 2.27 / 1.445 | 1.00 / 0.255 |
| RoboLab wrist (748 simulated frames) | 2.92 / 2.135 | 1.11 / 0.314 |

Details, ablations and caveats: [`research_notes/06_pilot_results.md`](research_notes/06_pilot_results.md).

## Try it on one wrist image

Use a rectified RGB image and the horizontal focal length from its camera calibration, in pixels at that image’s resolution. For a camera with `fx = 557.2` pixels, run this from the repository root.

```sh
uv run examples/predict_image.py wrist.png --fx-px 557.2
```

The first run downloads the model. Inference runs locally on CPU. The example writes `wrist.depth.npy` with camera-axis depth in metres at the original image size, and `wrist.depth.png` with a fixed grayscale range from 5 cm (white) to 1 m (black). It preserves the image aspect ratio and adjusts the focal length when fitting the frame to the model input.

The default model is v1. For DROID-style wrist views, add `--model christianengel/WristDepth-DROID-preview`. Its model card compares both exported graphs on DROID and the SO-101. The public procedural dataset contains 98,180 Blender frames split by scene. [DROID preview and measured limitations](https://huggingface.co/christianengel/WristDepth-DROID-preview) · [Procedural dataset](https://huggingface.co/datasets/christianengel/WristDepth-procedural)

## Pipeline

1. **Census** ([`research_notes/01`](research_notes/01_census.md)-[`05`](research_notes/05_assets_licenses.md)):
   camera, gripper, pose and appearance statistics of ~7,000 community LeRobot wrist datasets.
2. **Gripper** (`gripper/`): fit the SO-101 CAD jaw to real finger photos, export meshes posed
   in the camera frame.
3. **Render** (`procgen/`): BlenderProc/Cycles scenes with CC0 assets (Poly Haven, ambientCG),
   then a wrist-camera post-process (fisheye, exposure, blur, JPEG). `vast_render.sh` shards
   rendering over rented GPUs.
4. **Train** (`train/`): fine-tune decoder, head and the last encoder blocks; self-distil from
   the frozen base model on unlabelled community frames (`community/grab_frames.py`) to keep
   depth shape. `eval_sets.py` scores SO-101, DROID (val/test by scene) and RoboLab.
5. **Release** (`card/`): export with [depthpro2-onnx](https://github.com/protortyp/depthpro2-onnx)
   (`export_onnx.py --weights`), check parity, stage a model card from the results files.
6. **Dataset** (`dataset/build_hf.py`): write the renders as Hugging Face Parquet shards
   (`wrist` and `pinhole` configs, splits by scene).

`twin/`, `synth/` and `real/` hold the first pilot: a Gaussian-splat twin of one desk, MuJoCo
compositing, and real SO-101 captures with kinematic ground truth. The twin overfits to its
desk and is kept for diagnostics only.

## Setup

```sh
export WRIST_DEPTH_DATA=/path/to/data      # renders, runs, eval sets
export SO101_XML=/path/to/SO-ARM100/Simulation/SO101/so101_new_calib.xml
uv run --extra train python train/eval_sets.py base=B mine=B:/path/to/checkpoint.pt
```

Rendering runs in BlenderProc's own Blender (`uvx --from blenderproc blenderproc run
procgen/render.py ...`); see each script's header for its arguments.

## License

Code: Apache-2.0. Depth Pro 2 weights and their fine-tunes are under Apple's research-only
[model license](https://github.com/apple-aiml-research/ml-depth-pro-2). Evaluation data
(SpatialBench, community LeRobot frames) is used locally and not redistributed.
