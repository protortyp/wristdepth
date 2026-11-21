"""Wrist-camera clip with a slider sweeping right to left: RGB left of the line, WristDepth right.

    python make_slider_video.py CLIP.mp4 OUT.mp4 "credit line"

Also writes OUT_center.png, the frame where the line crosses the middle.
"""

import subprocess, sys
from functools import partial
import av, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
from depth_pro_2 import DepthPro2
from depth_pro_2.model import to_normalized_tensor

src, out, credit = sys.argv[1], sys.argv[2], sys.argv[3]
SCALE = 2  # 640x480 -> 1280x960
frames = [f.to_ndarray(format="rgb24") for f in av.open(src).decode(video=0)]
model = (
    DepthPro2.from_pretrained("christianengel/WristDepth", precision=torch.bfloat16)
    .to("cuda")
    .eval()
)
transform = partial(to_normalized_tensor, device=torch.device("cuda"))
with torch.no_grad():
    inv = [
        p["inverse_depth"].float().cpu().numpy()
        for p in model.infer_video(
            (transform(f) for f in frames), window=4, f_px=400.0, mode="unbuffered"
        )
    ]
# One colour range for the whole clip so colours mean the same distance in every frame.
allv = np.concatenate([i[::4, ::4].ravel() for i in inv])
lo, hi = np.percentile(allv, 2), np.percentile(allv, 99.5)
font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
small = ImageFont.truetype("DejaVuSans.ttf", 16)
n = len(frames)
W, H = frames[0].shape[1] * SCALE, frames[0].shape[0] * SCALE
enc = subprocess.Popen(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        "30",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out,
    ],
    stdin=subprocess.PIPE,
)
still = None
for k, (rgb, iv) in enumerate(zip(frames, inv)):
    depth_rgb = (cm.turbo(np.clip((iv - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)
    a = Image.fromarray(rgb).resize((W, H), Image.Resampling.LANCZOS)
    b = Image.fromarray(depth_rgb).resize((W, H), Image.Resampling.LANCZOS)
    # Ease in and out so the sweep starts and ends gently.
    t = k / max(n - 1, 1)
    t = t * t * (3 - 2 * t)
    x = int(round(W * (1 - t)))
    canvas = a.copy()
    canvas.paste(b.crop((x, 0, W, H)), (x, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle([x - 2, 0, x + 2, H], fill=(255, 255, 255))
    d.ellipse([x - 18, H // 2 - 18, x + 18, H // 2 + 18], fill=(255, 255, 255))
    d.polygon([(x - 10, H // 2), (x - 3, H // 2 - 7), (x - 3, H // 2 + 7)], fill=(40, 40, 40))
    d.polygon([(x + 10, H // 2), (x + 3, H // 2 - 7), (x + 3, H // 2 + 7)], fill=(40, 40, 40))
    for text, pos in (
        ("RGB", (24, 20)),
        ("WristDepth", (W - 24 - d.textlength("WristDepth", font=font), 20)),
    ):
        d.text(pos, text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    d.text(
        (24, H - 34),
        credit,
        font=small,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    enc.stdin.write(np.asarray(canvas).tobytes())
    if still is None and x <= W // 2:
        still = canvas
enc.stdin.close()
enc.wait()
still.save(out.replace(".mp4", "_center.png"))
print("frames", n, "->", out)
