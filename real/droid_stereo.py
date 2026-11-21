"""Metric depth labels for DROID wrist frames from the ZED Mini stereo video (DROID: CC-BY 4.0).

    uv run --extra export --with opencv-python-headless python droid_stereo.py OUT \
        --raft /path/to/RAFT-Stereo --episodes 300 --stride 15
    uv run ... droid_stereo.py OUT --raft ... --episode LAB/success/DATE/STAMP   # one episode

Disparity comes from RAFT-Stereo (MIT) with a left-right consistency check. The training target
is inverse depth f / (W d) = disparity / (W B), so it depends on the stereo baseline B (factory
calibration from Stereolabs, by camera serial) but not on the rectified focal length; f_px is
stored only so depth and the model input agree.

Episodes recorded on a day that also appears in the SpatialBench DROID evaluation set are skipped,
so no evaluation scene (or its same-day twin) is used for training.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
FILES = "https://storage.googleapis.com/gresearch/"
ROOT = "robotics/droid_raw/1.0.1/"
EVAL = Path(os.environ.get("WRIST_DEPTH_DATA", "data")) / "eval" / "spatialbench"
OUT_W = 640
DEPTH_MIN, DEPTH_MAX = 0.05, 3.0


def list_prefixes(prefix: str) -> list[str]:
    out, token = [], None
    while True:
        q = {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        d = json.load(urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=60))
        out += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def list_objects(prefix: str) -> list[str]:
    q = urllib.parse.urlencode({"prefix": prefix, "maxResults": 200})
    d = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=60))
    return [o["name"] for o in d.get("items", [])]


def eval_days() -> set[str]:
    """Days ("Jul_14_2023") of SpatialBench DROID episodes, from folder names like
    Fri_Jul_14_15:19:25_2023."""
    days = set()
    for meta in EVAL.rglob("droid/*/*/meta.json"):
        p = meta.parent.parent.name.split("_")
        days.add(f"{p[1]}_{p[2] or p[3]}_{p[-1]}")
    return days


def episode_day(stamp: str) -> str:
    p = stamp.rstrip("/").split("/")[-1].split("_")
    return f"{p[1]}_{p[2] or p[3]}_{p[-1]}"


def baseline_m(serial: str, cache: Path) -> float:
    f = cache / f"{serial}.conf"
    if not f.exists():
        f.write_bytes(urllib.request.urlopen(f"https://calib.stereolabs.com/?SN={serial}").read())
    for line in f.read_text().splitlines():
        if line.startswith("Baseline="):
            return float(line.split("=")[1]) / 1000
    raise ValueError(f"no baseline for {serial}")


def video_frames(path: Path, stride: int):
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        if i % stride == 0:
            yield i, frame[:, :, ::-1]
        i += 1


class Stereo:
    def __init__(self, raft: Path, device="cuda"):
        sys.path.insert(0, str(raft))
        from core.raft_stereo import RAFTStereo

        args = argparse.Namespace(
            hidden_dims=[128] * 3,
            corr_implementation="reg",
            shared_backbone=False,
            corr_levels=4,
            corr_radius=4,
            n_downsample=2,
            context_norm="batch",
            slow_fast_gru=False,
            n_gru_layers=3,
            mixed_precision=True,
        )
        model = torch.nn.DataParallel(RAFTStereo(args))
        model.load_state_dict(torch.load(raft / "models" / "raftstereo-middlebury.pth"))
        self.model = model.module.to(device).eval()
        self.device = device

    @torch.no_grad()
    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        def t(a):
            return (
                torch.from_numpy(np.ascontiguousarray(a))
                .permute(2, 0, 1)[None]
                .float()
                .to(self.device)
            )

        _, flow = self.model(t(left), t(right), iters=32, test_mode=True)
        return -flow[0, 0].float().cpu().numpy()

    def consistent(self, left, right, tol=1.0):
        """Left disparity, with pixels whose right-view disparity disagrees set to 0."""
        dl = self.disparity(left, right)
        # Right-view disparity: match the mirrored pair, then mirror back.
        dr = self.disparity(right[:, ::-1], left[:, ::-1])[:, ::-1]
        h, w = dl.shape
        xs = np.arange(w)[None].repeat(h, 0)
        xr = np.clip(np.round(xs - dl).astype(int), 0, w - 1)
        back = dr[np.arange(h)[:, None], xr]
        ok = (dl > 0.5) & (np.abs(dl - back) < tol) & (xs - dl >= 0)
        return np.where(ok, dl, 0.0)


def label(stereo, frame, base_m, f_full):
    """640-wide left RGB, depth (0 = no label) and f_px, cropped to a multiple of 32 in height."""
    h, w2 = frame.shape[:2]
    w = w2 // 2
    left, right = frame[:, :w], frame[:, w:]
    disp = stereo.consistent(left, right)
    s = OUT_W / w
    rgb = cv2.resize(left, (OUT_W, round(h * s)), interpolation=cv2.INTER_AREA)
    d_small = cv2.resize(disp, (OUT_W, round(h * s)), interpolation=cv2.INTER_NEAREST) * s
    valid = (
        cv2.resize(
            (disp > 0).astype(np.uint8), (OUT_W, round(h * s)), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )
    f = f_full * s
    depth = np.where(valid & (d_small > 0), f * base_m / np.maximum(d_small, 1e-6), 0).astype(
        np.float32
    )
    depth[(depth < DEPTH_MIN) | (depth > DEPTH_MAX)] = 0
    hh = rgb.shape[0] // 32 * 32
    top = (rgb.shape[0] - hh) // 2
    return rgb[top : top + hh], depth[top : top + hh], f


def process(stereo, episode: str, out: Path, stride: int, cache: Path) -> int:
    names = list_objects(episode)
    meta_name = next((n for n in names if n.split("/")[-1].startswith("metadata_")), None)
    if meta_name is None:
        return 0
    meta = json.load(urllib.request.urlopen(FILES + urllib.parse.quote(meta_name)))
    serial = meta["wrist_cam_serial"]
    mp4 = next((n for n in names if n.endswith(f"MP4/{serial}-stereo.mp4")), None)
    if mp4 is None:
        return 0
    base = baseline_m(serial, cache)
    conf = (cache / f"{serial}.conf").read_text()
    # Factory HD720 focal; the video's rectified focal differs by a few percent, which cancels
    # in the inverse-depth target (see module docstring).
    f_full = float(conf.split("[LEFT_CAM_HD]")[1].split("fx=")[1].split()[0])
    tmp = cache / "episode.mp4"
    subprocess.run(["curl", "-sf", "-o", str(tmp), FILES + urllib.parse.quote(mp4)], check=True)
    tag = meta["uuid"].replace("+", "-")
    n = 0
    for i, frame in video_frames(tmp, stride):
        rgb, depth, f = label(stereo, frame, base, f_full)
        if (depth > 0).mean() < 0.3:
            continue
        np.savez_compressed(
            out / f"droid-{tag}_{i:05d}.npz",
            rgb=rgb,
            depth=depth,
            trust=((depth > 0) * 255).astype(np.uint8),
            f_px=np.float32(f),
            meta=json.dumps(
                {"uuid": meta["uuid"], "lab": meta["lab"], "frame": i, "baseline_m": base}
            ),
        )
        n += 1
    return n


def episodes(n: int, seed: int, skip_days: set[str]) -> list[str]:
    """n episodes spread over labs and days, never on a SpatialBench evaluation day."""
    rng = np.random.default_rng(seed)
    labs = list_prefixes(ROOT)
    picked = []
    per_lab = max(1, n // len(labs))
    for lab in labs:
        days = [d for d in list_prefixes(lab + "success/")]
        rng.shuffle(days)
        got = 0
        for day in days:
            if got >= per_lab:
                break
            eps = [e for e in list_prefixes(day) if episode_day(e) not in skip_days]
            if not eps:
                continue
            picked.append(eps[rng.integers(len(eps))])
            got += 1
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--raft", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--episode", help="one episode prefix below droid_raw/1.0.1/")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "_cache"
    cache.mkdir(exist_ok=True)
    stereo = Stereo(args.raft)
    if args.episode:
        todo = [ROOT + args.episode.strip("/") + "/"]
    else:
        todo = episodes(args.episodes, args.seed, eval_days())
        (args.out / "episodes.json").write_text(json.dumps(todo, indent=1))
    done_file = args.out / "_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    for k, ep in enumerate(todo):
        if ep in done:
            continue
        try:
            n = process(stereo, ep, args.out, args.stride, cache)
        except Exception as e:  # noqa: BLE001 - one broken episode should not stop a long run
            print(f"skip {ep}: {e}", flush=True)
            continue
        with done_file.open("a") as fh:
            fh.write(ep + "\n")
        print(f"{k + 1}/{len(todo)} {ep.split('/')[-2]}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
