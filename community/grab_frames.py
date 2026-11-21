"""Sample unlabelled wrist-camera frames from many community LeRobot datasets on the Hub.

    python grab_frames.py CENSUS_ROWS.json OUT_DIR [--datasets 600] [--per 6]

Used as self-distillation inputs: the frozen base model's depth shape on real wrist images
is kept while synthetic data corrects metric scale. One dataset per owner for variety; frames
are spread over the first video file. Frames are third-party data: kept local, used for
training only, never redistributed. Needs ffmpeg and a Hub token (anonymous access is
rate-limited to uselessness).
"""

import argparse
import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOKEN = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
HEADER = f"Authorization: Bearer {TOKEN}\r\n"


def video_url(row) -> str:
    pattern = (
        row["video_path"]
        or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    path = pattern.format(
        video_key=row["wrist_key"], chunk_index=0, file_index=0, episode_chunk=0, episode_index=0
    )
    return f"https://huggingface.co/datasets/{row['id']}/resolve/main/{path}"


def run(cmd):
    for attempt in range(6):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if "429" not in p.stderr:
            return p
        time.sleep(20 * (attempt + 1))
    return p


def grab(row, out: Path, per: int):
    url = video_url(row)
    p = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-headers",
            HEADER,
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            url,
        ]
    )
    try:
        duration = float(p.stdout.strip())
    except ValueError:
        return row["id"], 0
    ok = 0
    for i in range(per):
        fn = out / f"{row['id'].replace('/', '__')}__{i}.jpg"
        if not fn.exists():
            t = duration * (0.03 + 0.94 * i / max(per - 1, 1))
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-headers",
                    HEADER,
                    "-ss",
                    f"{t:.2f}",
                    "-i",
                    url,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(fn),
                ]
            )
        ok += fn.exists()
    return row["id"], ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--datasets", type=int, default=600)
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        r
        for r in json.loads(args.rows.read_text())
        if r["dtype"] == "video" and not r["sim_by_name"]
    ]
    random.Random(args.seed).shuffle(rows)
    by_owner = {}
    for r in rows:
        by_owner.setdefault(r["id"].split("/")[0], r)
    chosen = list(by_owner.values())[: args.datasets]
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda r: grab(r, args.out, args.per), chosen))
    (args.out / "sources.json").write_text(
        json.dumps([{"id": i, "frames": n} for i, n in results], indent=1)
    )
    print(f"{sum(n for _, n in results)} frames from {sum(n > 0 for _, n in results)} datasets")


if __name__ == "__main__":
    main()
