from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2


def load_cloud_tracks(csv_path: Path) -> dict[int, list[dict]]:
    tracks_by_frame = defaultdict(list)

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            frame_number = int(row["frame_number"])

            tracks_by_frame[frame_number].append(
                {
                    "obj_id": int(row["obj_id"]),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                }
            )

    return tracks_by_frame


def track_color(obj_id: int) -> tuple[int, int, int]:
    """
    Generate a stable color for each obj_id.
    Same ID -> same color every time.
    """
    digest = hashlib.sha1(str(obj_id).encode()).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def draw_cloud_reference(
    video_path: Path,
    tracks_path: Path,
    output_path: Path,
) -> None:

    tracks_by_frame = load_cloud_tracks(tracks_path)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Keep trajectory history for each cloud ID
    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    frame_number = 0

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        frame_tracks = tracks_by_frame.get(frame_number, [])

        for track in frame_tracks:
            obj_id = track["obj_id"]

            x1 = round(track["x1"])
            y1 = round(track["y1"])
            x2 = round(track["x2"])
            y2 = round(track["y2"])

            color = track_color(obj_id)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # ID
            cv2.putText(
                frame,
                f"cloud {obj_id}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # Track trail
            center = (
                round((x1 + x2) / 2),
                round((y1 + y2) / 2),
            )

            trails[obj_id].append(center)

            for point in trails[obj_id]:
                cv2.circle(
                    frame,
                    point,
                    2,
                    color,
                    -1,
                )

        # Frame number
        cv2.putText(
            frame,
            f"frame {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        writer.write(frame)

        frame_number += 1

        if frame_count and frame_number >= frame_count:
            break

    capture.release()
    writer.release()

    print(f"frames processed: {frame_number}")
    print(f"cloud obj_ids: {len({t['obj_id'] for tracks in tracks_by_frame.values() for t in tracks})}")
    print(f"wrote: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw stored cloud reference tracks on a video."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="cloud reference tracks CSV",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    draw_cloud_reference(
        args.video,
        args.tracks,
        args.out,
    )


if __name__ == "__main__":
    main()