"""Overlay layers: per-frame boxes and track polylines for rendering over video.

The layer schema (version 1) is producer-agnostic: any producer of per-frame boxes and
tracks may emit it, and any renderer consuming it needs no knowledge of the producer.
Evaluation ``RunResult``s are the first producer. Coordinates are SOURCE video pixels;
renderers scale to their display size using ``source_width``/``source_height``.
"""

from __future__ import annotations

from spoorpredictioneval.contracts import FrameDetections, RunResult
from spoorpredictioneval.metrics import iou

OVERLAY_SCHEMA_VERSION = 1


def overlay_layer(result: RunResult, label: str) -> dict:
    """Project a run's detections and tracks into one overlay layer (schema v1).

    ``source_width``/``source_height`` are included when the result's provenance carries
    them (``extra["source_width"]``/``extra["source_height"]``) and omitted otherwise.
    Frames with no detections are omitted; frame keys are strings (JSON object keys).
    """
    layer: dict = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "label": label,
        "frames": {
            str(frame): [
                {
                    "box": list(detection.box_xywh),
                    "confidence": detection.confidence,
                    "track_id": detection.track_id,
                }
                for detection in detections
            ]
            for frame, detections in sorted(result.correctness.items())
            if detections
        },
        "tracks": [
            {
                "track_id": track.track_id,
                "points": [[frame, x, y] for frame, x, y in track.trajectory],
            }
            for track in result.tracks
        ],
    }
    extra = result.provenance.extra
    if "source_width" in extra and "source_height" in extra:
        layer["source_width"] = extra["source_width"]
        layer["source_height"] = extra["source_height"]
    return layer


def disagreement_frames(
    candidate: FrameDetections,
    reference: FrameDetections,
    iou_threshold: float = 0.5,
    confidence_floor: float = 0.7,
) -> list[dict]:
    """Frames where the candidate and reference disagree, with the reason per frame.

    Reference-side reasons follow the ``recall_decomposition`` vocabulary
    (``no_overlap``, ``under_localized``, ``under_confident``); ``unmatched_candidate``
    marks candidate detections at or above the floor overlapping no reference box.
    One entry per (frame, reason) with the count of affected boxes, ordered by frame.
    """
    reasons: dict[tuple[int, str], int] = {}

    for frame, reference_detections in reference.items():
        candidates = candidate.get(frame, [])
        for reference_detection in reference_detections:
            overlaps = [
                (iou(candidate_detection.box_xywh, reference_detection.box_xywh), candidate_detection)
                for candidate_detection in candidates
            ]
            best_iou, _ = max(overlaps, key=lambda entry: entry[0], default=(0.0, None))
            if best_iou == 0.0:
                reason = "no_overlap"
            elif best_iou < iou_threshold:
                reason = "under_localized"
            else:
                best_confidence = max(
                    candidate_detection.confidence
                    for overlap, candidate_detection in overlaps
                    if overlap >= iou_threshold
                )
                if best_confidence >= confidence_floor:
                    continue
                reason = "under_confident"
            reasons[(frame, reason)] = reasons.get((frame, reason), 0) + 1

    for frame, candidate_detections in candidate.items():
        references = reference.get(frame, [])
        for candidate_detection in candidate_detections:
            if candidate_detection.confidence < confidence_floor:
                continue
            best_iou = max(
                (iou(candidate_detection.box_xywh, r.box_xywh) for r in references),
                default=0.0,
            )
            if best_iou == 0.0:
                reasons[(frame, "unmatched_candidate")] = reasons.get((frame, "unmatched_candidate"), 0) + 1

    return [{"frame": frame, "reason": reason, "count": count} for (frame, reason), count in sorted(reasons.items())]
