# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub>=0.27,<2", "numpy>=1.24", "onnxruntime>=1.23,<2", "pillow>=10"]
# ///
"""Estimate camera-axis depth for one wrist-camera image using the published ONNX model."""

import argparse
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image


def predict(image: Image.Image, fx_px: float, session: ort.InferenceSession) -> np.ndarray:
    _, _, height, width = session.get_inputs()[0].shape
    scale = min(width / image.width, height / image.height)
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    left = (width - resized[0]) // 2
    top = (height - resized[1]) // 2
    canvas = Image.new("RGB", (width, height))
    canvas.paste(image.convert("RGB").resize(resized, Image.Resampling.BILINEAR), (left, top))

    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    focal = np.asarray([fx_px * resized[0] / image.width], dtype=np.float32)
    depth = session.run(["depth_meters"], {"pixel_values": pixels, "f_px": focal})[0]
    depth = depth.reshape(height, width)[top : top + resized[1], left : left + resized[0]]
    invalid = ~np.isfinite(depth) | (depth <= 0)
    if invalid.all():
        raise ValueError("The model returned no valid depth values")
    depth = np.where(invalid, np.float32(0), depth)
    original = Image.fromarray(depth).resize(image.size, Image.Resampling.BILINEAR)
    result = np.array(original, dtype=np.float32)
    if invalid.any():
        # A resampled distance needs valid measurements throughout its interpolation footprint.
        missing = Image.fromarray(invalid.astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR)
        result[np.asarray(missing) > 0] = np.nan
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--fx-px",
        required=True,
        type=float,
        help="Horizontal focal length in pixels on the input image",
    )
    parser.add_argument(
        "--model", default="christianengel/WristDepth", help="Public Hugging Face model repository"
    )
    args = parser.parse_args()
    if not math.isfinite(args.fx_px) or args.fx_px <= 0:
        parser.error("--fx-px must be finite and positive")

    with Image.open(args.image) as image:
        graph = hf_hub_download(args.model, "onnx/model_fp16.onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            graph, sess_options=options, providers=["CPUExecutionProvider"]
        )
        depth = predict(image, args.fx_px, session)
    valid = np.isfinite(depth)
    if not valid.any():
        raise ValueError("The output image contains no valid depth values")

    prefix = args.image.with_suffix("")
    metric_path = prefix.with_name(prefix.name + ".depth.npy")
    preview_path = prefix.with_name(prefix.name + ".depth.png")
    np.save(metric_path, depth)
    # The fixed visualization range makes predictions from different images comparable.
    near, far = 0.05, 1.0
    gray = np.zeros_like(depth)
    gray[valid] = np.clip(np.log(far / depth[valid]) / np.log(far / near), 0, 1)
    if not valid.all():
        print(f"{depth.size - valid.sum()} pixels have missing depth (NaN in the metric file).")
    Image.fromarray(np.rint(255 * gray).astype(np.uint8)).save(preview_path)
    print(f"{metric_path} (float32 metres)\n{preview_path} (near is bright)")


if __name__ == "__main__":
    main()
