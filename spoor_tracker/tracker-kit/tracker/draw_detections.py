import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def load_detections(csv_path):
    detections_by_frame = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame = int(row["frame_number"])

            detection = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "w": float(row["w"]),
                "h": float(row["h"]),
                "classifier_name": row.get("classifier_name", ""),
            }

            detections_by_frame[frame].append(detection)

    return detections_by_frame


def draw_detections(video_path, csv_path, output_path):

    detections = load_detections(csv_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps:
        fps = 10

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_number = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_detections = detections.get(frame_number, [])

        for det in frame_detections:

            x1 = int(det["x"])
            y1 = int(det["y"])

            x2 = int(det["x"] + det["w"])
            y2 = int(det["y"] + det["h"])

            # Every detection is deliberately the same color:
            # these are detections, NOT tracks.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2,
            )

            label = det["classifier_name"]

            if label:
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        # Frame number
        cv2.putText(
            frame,
            f"frame {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        writer.write(frame)

        frame_number += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_number} frames")
    print(f"Wrote: {output_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    draw_detections(
        Path(args.video),
        Path(args.detections),
        Path(args.out),
    )


if __name__ == "__main__":
    main()