"""Run the bird-watcher tracker over a video and its detections CSV, and write an annotated
video plus a track CSV.

    python run_tracker.py --video data/X.mp4 --detections data/X_detections.csv \
        --out out/X.mp4 --tracks out/X_tracks.csv [--euclidean-matching-threshold 250]

The detections CSV is birdwatcher's own schema
(projects/bird-watcher/src/birdwatcher/pipelines/detections_csv_schema.py:23-32):
`frame_number, frame_timestamp, x, y, w, h, area, classifier_name`. `x, y` are the box's
top-left corner. Frames absent from the CSV are treated as having no detections that frame
(a detection gap), not as an error.

The CSV's `frame_number` values must exactly span the video's own frame range, 0 through
frame_count - 1: this is how a detection is matched to the video frame it belongs to (see
check_frame_alignment below). A CSV cut with `--start-frame-number`/`--stop-frame-number` and
run against the uncut video (or vice versa) fails this check and refuses to run, rather than
silently tracking the wrong frames.

Per-frame processing, straight from `SimpleSORTTracker.match_and_track`'s own comments and
`tracker_filter.py` (`projects/bird-watcher/src/birdwatcher/pipelines/elements/filters/
tracker_filter.py`, the tracker's one production caller): feed this frame's detections (an
empty list on a frame with none) to `match_and_track`; every existing track predicts forward
by exactly one step *every single call*, matched or not -- there is no elapsed-time or
frame-gap input here, unlike the edge tracker this kit's earlier revision copied. That is why
this loop must call `match_and_track` exactly once per video frame, in the video's own frame
order: skipping a call, or calling it twice for one frame, silently changes how many predict
steps a coasting track gets. See the kit README and tracker/simple_sort_tracker.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
import json

import cv2
from tracker.shim import BoundingBoxXYWH, DetectionNoCrop, xywh_to_xyxy
from tracker.simple_sort_tracker import SimpleSORTTracker
from tracker.tracked_object_state import TrackedObjectState

# libs/configs-manager/src/spoorconfigsmanager/config_files/config_8k.yaml: distance_threshold:
# 250, box_thickness: 3, dot_size: 10, all fleet-tuned for 7680px-wide recordings. Anchoring the
# tracker's default threshold and the rendering scale to that same known-good width is what lets
# this runner default sensibly on an 8K recording and still be legible on a small clip.
REFERENCE_WIDTH_PX = 7680
REFERENCE_EUCLIDEAN_MATCHING_THRESHOLD = 250.0
REFERENCE_BOX_THICKNESS_PX = 3
REFERENCE_DOT_DIAMETER_PX = 10
MINIMUM_BOX_THICKNESS_PX = 2
MINIMUM_DOT_RADIUS_PX = 2
MINIMUM_FONT_SCALE = 0.4

# projects/bird-watcher/src/birdwatcher/pipelines/bird_watcher_tracking_pipeline.py:143
DEFAULT_MAX_AGE = 30
DEFAULT_TENTATIVE_THRESHOLD = 3


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        config = json.load(f)

    return config

def annotation_scale(frame_width: int) -> tuple[int, int, float]:
    """Box thickness, path dot radius and font scale for a frame of the given width.

    Linear in `frame_width`, anchored at `REFERENCE_WIDTH_PX` to config_8k.yaml's own tuned
    box_thickness/dot_size, floored at what is legible on the kit's own small synthetic clip.
    """
    ratio = frame_width / REFERENCE_WIDTH_PX
    box_thickness = max(MINIMUM_BOX_THICKNESS_PX, round(REFERENCE_BOX_THICKNESS_PX * ratio))
    dot_radius = max(MINIMUM_DOT_RADIUS_PX, round(REFERENCE_DOT_DIAMETER_PX * ratio / 2))
    font_scale = max(MINIMUM_FONT_SCALE, REFERENCE_BOX_THICKNESS_PX * ratio * 0.4)
    return box_thickness, dot_radius, font_scale


def load_detections_by_frame(csv_path: Path) -> dict[int, list[BoundingBoxXYWH]]:
    by_frame: dict[int, list[BoundingBoxXYWH]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame_number = int(row["frame_number"])
            by_frame[frame_number].append(
                BoundingBoxXYWH(x=float(row["x"]), y=float(row["y"]), width=float(row["w"]), height=float(row["h"]))
            )
    return by_frame


def build_detections(frame_number: int, frame_timestamp: int, boxes: list[BoundingBoxXYWH]) -> list[DetectionNoCrop]:
    """Turn this frame's boxes into detections the tracker can match against.

    Mirrors `tracker_filter.py`'s own construction exactly: `feature` is `(x, y, area)` taken
    from the box's own top-left corner and area, not a separate appearance embedding -- see
    tracker/shim.py and the kit README for why this is load-bearing rather than a default.
    """
    return [
        DetectionNoCrop(
            bounding_box=xywh_to_xyxy(box),
            frame_number=frame_number,
            frame_timestamp=frame_timestamp,
            feature=(int(box.x), int(box.y), float(box.area)),
        )
        for box in boxes
    ]


def _track_color(track_id: int) -> tuple[int, int, int]:
    digest = hashlib.sha1(str(track_id).encode()).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def check_frame_alignment(
    detections_by_frame: dict[int, list[BoundingBoxXYWH]],
    frame_count: int,
    video_path: Path,
    detections_path: Path,
) -> None:
    """Refuse to run if the CSV's frame numbering does not cover this exact video.

    This tracker has no frame-gap or elapsed-time input at all -- every call to
    `match_and_track` advances every track's Kalman prediction by exactly one step (see
    tracker/kalman_filter.py: `dt = 1.0`, fixed). This loop relies on calling it exactly once
    per real video frame, in order, with that frame's own detections. A CSV cut with
    `--start-frame-number`/`--stop-frame-number` (see `main_detections.py`) is renumbered from
    0 relative to the cut; feeding it against the *other* version of the clip (cut vs. uncut)
    would silently attach every detection to the wrong frame -- some coasting tracks would get
    predict-only steps where they should have gotten a real correction, and vice versa. There is
    no field in either file this code could use to detect or correct that offset, so it refuses
    rather than guessing one.
    """
    if not detections_by_frame:
        return
    lowest_frame, highest_frame = min(detections_by_frame), max(detections_by_frame)
    if lowest_frame == 0 and highest_frame == frame_count - 1:
        return
    raise SystemExit(
        f"{detections_path} covers frame_number {lowest_frame}..{highest_frame}, but "
        f"{video_path} has {frame_count} frames (0..{frame_count - 1}). These do not match, "
        "which is exactly what happens when one of the two was cut with "
        "--start-frame-number/--stop-frame-number and the other was not: a cut CSV's "
        "frame_number is renumbered from 0 relative to the cut, not the original recording. "
        "Feed run_tracker.py the *same cut* of the video that produced this CSV, or don't "
        "cut either one. Refusing rather than guessing an offset, since a wrong guess would "
        "silently attach detections to the wrong frames."
    )


def run(
    video_path: Path,
    detections_path: Path,
    out_video_path: Path,
    out_tracks_path: Path,
    euclidean_matching_threshold: float,
    max_age: int,
    tentative_threshold: int,
    kalman_gating: str,
) -> int:
    detections_by_frame = load_detections_by_frame(detections_path)
    tracker = SimpleSORTTracker(
        euclidean_matching_threshold=euclidean_matching_threshold,
        max_age=max_age,
        tentative_threshold=tentative_threshold,
        kalman_gating=kalman_gating,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    check_frame_alignment(detections_by_frame, frame_count, video_path, detections_path)
    box_thickness, dot_radius, font_scale = annotation_scale(width)

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_tracks_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
    track_rows: list[dict] = []
    frames_processed = 0

    with out_tracks_path.open("w", newline="") as tracks_handle:
        fieldnames = ["frame_number", "track_id", "x", "y", "w", "h"]
        tracks_writer = csv.DictWriter(tracks_handle, fieldnames=fieldnames)
        tracks_writer.writeheader()

        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_timestamp = round(frame_number * 1000 / fps)
            boxes = detections_by_frame.get(frame_number, [])
            detections = build_detections(frame_number, frame_timestamp, boxes)
            tracker.match_and_track(detections)

            for tracked_object in tracker.tracked_objects:
                if tracked_object.state != TrackedObjectState.CONFIRMED:
                    continue
                # Same convention as the fleet's own DisplayTracksSink: the last matched
                # detection's box is the "current" box, not the Kalman filter's own state.
                box = tracked_object.detections[-1].bounding_box
                color = _track_color(tracked_object.id)
                top_left = (round(box.x1), round(box.y1))
                bottom_right = (round(box.x2), round(box.y2))
                cv2.rectangle(frame, top_left, bottom_right, color, box_thickness)
                center = (round((box.x1 + box.x2) / 2), round((box.y1 + box.y2) / 2))
                trails[tracked_object.id].append(center)
                for point in trails[tracked_object.id]:
                    cv2.circle(frame, point, dot_radius, color, -1)
                label = str(tracked_object.id)
                cv2.putText(
                    frame,
                    label,
                    (top_left[0], top_left[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    box_thickness,
                )
                track_rows.append(
                    {
                        "frame_number": frame_number,
                        "track_id": tracked_object.id,
                        "x": box.x1,
                        "y": box.y1,
                        "w": box.width,
                        "h": box.height,
                    }
                )

            frame_counter_baseline_y = round(20 * font_scale / MINIMUM_FONT_SCALE)
            cv2.putText(
                frame,
                f"frame {frame_number}",
                (10, frame_counter_baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                box_thickness,
            )
            writer.write(frame)
            frames_processed += 1
            frame_number += 1
            if frame_count and frame_number >= frame_count:
                break

        tracks_writer.writerows(track_rows)

    tracker.finalize_tracking()
    capture.release()
    writer.release()
    return frames_processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="path to write the annotated video")
    parser.add_argument("--tracks", type=Path, required=True, help="path to write the track CSV")
    parser.add_argument(
        "--euclidean-matching-threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--tentative-threshold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--kalman-gating",
        choices=["position", "full"],
        default=None,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON experiment configuration file",
    )


    args = parser.parse_args()

    config = {}
    
    if args.config is not None:
        config = load_config(args.config)

    euclidean_matching_threshold = (
        args.euclidean_matching_threshold
        if args.euclidean_matching_threshold is not None
        else config.get(
            "euclidean_matching_threshold",
            REFERENCE_EUCLIDEAN_MATCHING_THRESHOLD,
        )
    )

    max_age = (
        args.max_age
        if args.max_age is not None
        else config.get("max_age", DEFAULT_MAX_AGE)
    )

    tentative_threshold = (
        args.tentative_threshold
        if args.tentative_threshold is not None
        else config.get(
            "tentative_threshold",
            DEFAULT_TENTATIVE_THRESHOLD,
        )
    )

    kalman_gating = (
        args.kalman_gating
        if args.kalman_gating is not None
        else config.get("kalman_gating", "position")
    )


    frames_processed = run(
        args.video,
        args.detections,
        args.out,
        args.tracks,
        euclidean_matching_threshold,
        max_age,
        tentative_threshold,
        kalman_gating,
    )

    experiment_name = config.get("experiment_name", "custom")

    print(f"experiment: {experiment_name}")
    print(f"euclidean_matching_threshold: {euclidean_matching_threshold}")
    print(f"max_age: {max_age}")
    print(f"tentative_threshold: {tentative_threshold}")
    print(f"kalman_gating: {kalman_gating}")
    print(f"frames processed: {frames_processed}")
    print(f"wrote {args.out}")
    print(f"wrote {args.tracks}")


if __name__ == "__main__":
    main()
