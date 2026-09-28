"""Generate a synthetic detections CSV and matching video for the tracker kit.

Stands in for the real recordings while the S3 restore for `20250920_063942_6C42` and
`20251012_164031_1DAC` is pending (see data/README.md). Three birds on known paths:

  bird-a, bird-b : straight lines through the same point at the same frame, opposite
                   directions -- a crossing. Ground truth is known, so any identity switch
                   `run_tracker.py` produces at the crossing is visible against it.
  bird-c         : fast (40 px/frame) with a 3-frame occlusion gap. The gap covers 120px.
                   With `--euclidean-matching-threshold` at its default (250, config_8k.yaml's
                   own fleet-tuned value), the tracker's Kalman prediction bridges it and holds
                   bird-c's id. Drop the threshold below ~120 and this same tracker -- no second
                   tracker involved -- loses the identity at the gap instead: see the kit
                   README section 4 for the confirmed counts on this clip.

Detections are written in birdwatcher's own CSV schema
(projects/bird-watcher/src/birdwatcher/pipelines/detections_csv_schema.py:23-32):
`frame_number, frame_timestamp, x, y, w, h, area, classifier_name` -- x, y are the
top-left corner. The video shows the same birds as filled circles on a plain background,
scaled up from the report's ~8px-bird regime so the clip is a fair rehearsal for a
real-sized frame (see data/README.md) rather than a postage stamp.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import cv2
import numpy as np

CANVAS_SCALE = 3
WIDTH, HEIGHT = 640 * CANVAS_SCALE, 480 * CANVAS_SCALE
FPS = 10
NUM_FRAMES = 150  # bird-a/bird-b cross around frame 125 on this canvas; see module docstring
BIRD_SIZE = 8 * CANVAS_SCALE  # px, scaled up from the report's ~8px bird regime for legibility
OCCLUSION_GAP = range(5, 8)  # frames bird-c's detections are withheld (3 frames, 120px at its speed)


@dataclasses.dataclass(frozen=True)
class BirdPath:
    name: str
    start: tuple[float, float]
    velocity: tuple[float, float]  # px/frame
    color_bgr: tuple[int, int, int]
    missing_frames: frozenset[int] = frozenset()

    def center(self, frame: int) -> tuple[float, float]:
        x, y = self.start
        vx, vy = self.velocity
        return x + vx * frame, y + vy * frame


BIRDS = [
    BirdPath(
        name="bird-a",
        start=(50.0 * CANVAS_SCALE, 100.0 * CANVAS_SCALE),
        velocity=(6.0, 0.0),
        color_bgr=(60, 180, 250),
    ),
    BirdPath(
        name="bird-b",
        start=(550.0 * CANVAS_SCALE, 100.0 * CANVAS_SCALE),
        velocity=(-6.0, 0.0),
        color_bgr=(250, 120, 60),
    ),
    BirdPath(
        name="bird-c",
        start=(50.0 * CANVAS_SCALE, 300.0 * CANVAS_SCALE),
        velocity=(40.0, 0.0),
        color_bgr=(80, 220, 80),
        missing_frames=frozenset(OCCLUSION_GAP),
    ),
]


def _box_row(frame: int, timestamp_ms: int, center_x: float, center_y: float, classifier_name: str) -> dict:
    x = center_x - BIRD_SIZE / 2
    y = center_y - BIRD_SIZE / 2
    return {
        "frame_number": frame,
        "frame_timestamp": timestamp_ms,
        "x": round(x),
        "y": round(y),
        "w": BIRD_SIZE,
        "h": BIRD_SIZE,
        "area": float(BIRD_SIZE * BIRD_SIZE),
        "classifier_name": classifier_name,
    }


def generate(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "synthetic.mp4"
    detections_path = output_dir / "synthetic_detections.csv"

    rows = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (WIDTH, HEIGHT), True)
    try:
        for frame in range(NUM_FRAMES):
            canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
            timestamp_ms = round(frame * 1000 / FPS)
            for bird in BIRDS:
                center_x, center_y = bird.center(frame)
                margin = BIRD_SIZE
                on_screen = -margin <= center_x <= WIDTH + margin and -margin <= center_y <= HEIGHT + margin
                if not on_screen:
                    continue
                cv2.circle(canvas, (round(center_x), round(center_y)), BIRD_SIZE // 2, bird.color_bgr, -1)
                if frame not in bird.missing_frames:
                    rows.append(_box_row(frame, timestamp_ms, center_x, center_y, "bird"))
            writer.write(canvas)
    finally:
        writer.release()

    rows.sort(key=lambda row: row["frame_number"])
    with detections_path.open("w", newline="") as handle:
        fieldnames = ["frame_number", "frame_timestamp", "x", "y", "w", "h", "area", "classifier_name"]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    return video_path, detections_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()
    video_path, detections_path = generate(args.out_dir)
    print(f"wrote {video_path}")
    print(f"wrote {detections_path}")


if __name__ == "__main__":
    main()
