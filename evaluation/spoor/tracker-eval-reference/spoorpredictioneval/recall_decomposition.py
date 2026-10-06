"""Attribution of recall loss: for each reference detection, why was it not found.

Attribution is per-reference best-overlap, deliberately
not one-to-one: the question answered is "was there any candidate box on this object,
and what failed", so one candidate may attribute to several reference boxes. Counts may
therefore differ slightly from the one-to-one matched recall in ``banded``.
"""

from __future__ import annotations

from dataclasses import dataclass

from spoorpredictioneval.contracts import FrameDetections
from spoorpredictioneval.metrics import iou


@dataclass(frozen=True)
class RecallDecomposition:
    """Per-reference-detection attribution, one bucket each.

    ``found``: a candidate overlaps at or above the IoU threshold with confidence at or
    above the floor. ``under_confident``: overlap qualifies but no qualifying candidate
    reaches the confidence floor. ``under_localized``: some overlap exists but the best
    IoU is below the threshold. ``no_overlap``: no candidate touches the reference box at
    all (the detector produced nothing there). Distributions carry the evidence for the
    two intermediate buckets.
    """

    found: int
    under_confident: int
    under_localized: int
    no_overlap: int
    under_confident_confidences: tuple[float, ...]
    under_localized_best_ious: tuple[float, ...]
    under_localized_scale_ratios: tuple[float, ...]
    """Candidate area / reference area for each under-localized pair: <1 means the
    candidate box is smaller than the reference box."""
    under_localized_center_offsets: tuple[float, ...]
    """Center distance normalized by the reference box diagonal: how far off-target the
    candidate box sits, independent of size."""

    @property
    def total(self) -> int:
        return self.found + self.under_confident + self.under_localized + self.no_overlap

    def as_table(self) -> str:
        total = self.total or 1
        rows = [
            ("found", self.found),
            ("under_confident", self.under_confident),
            ("under_localized", self.under_localized),
            ("no_overlap", self.no_overlap),
        ]
        lines = [f"{'reason':<18}{'boxes':>8}{'share':>8}"]
        lines += [f"{name:<18}{count:>8}{count / total:>8.1%}" for name, count in rows]
        return "\n".join(lines)


def decompose_recall(
    candidate: FrameDetections,
    reference: FrameDetections,
    iou_threshold: float = 0.5,
    confidence_floor: float = 0.7,
) -> RecallDecomposition:
    """Attribute every reference detection to found / under_confident / under_localized /
    no_overlap against the candidate run, frame by frame."""
    found = 0
    under_confident: list[float] = []
    under_localized: list[float] = []
    scale_ratios: list[float] = []
    center_offsets: list[float] = []
    no_overlap = 0

    for frame, reference_detections in reference.items():
        candidates = candidate.get(frame, [])
        for reference_detection in reference_detections:
            overlaps = [
                (iou(candidate_detection.box_xywh, reference_detection.box_xywh), candidate_detection)
                for candidate_detection in candidates
            ]
            best_iou, best_candidate = max(overlaps, key=lambda entry: entry[0], default=(0.0, None))
            if best_candidate is None or best_iou == 0.0:
                no_overlap += 1
                continue
            if best_iou < iou_threshold:
                under_localized.append(best_iou)
                scale_ratios.append(_scale_ratio(best_candidate.box_xywh, reference_detection.box_xywh))
                center_offsets.append(_center_offset(best_candidate.box_xywh, reference_detection.box_xywh))
                continue
            best_confidence = max(
                candidate_detection.confidence for overlap, candidate_detection in overlaps if overlap >= iou_threshold
            )
            if best_confidence < confidence_floor:
                under_confident.append(best_confidence)
            else:
                found += 1

    return RecallDecomposition(
        found=found,
        under_confident=len(under_confident),
        under_localized=len(under_localized),
        no_overlap=no_overlap,
        under_confident_confidences=tuple(under_confident),
        under_localized_best_ious=tuple(under_localized),
        under_localized_scale_ratios=tuple(scale_ratios),
        under_localized_center_offsets=tuple(center_offsets),
    )


def _scale_ratio(candidate_box, reference_box) -> float:
    _, _, cw, ch = candidate_box
    _, _, rw, rh = reference_box
    reference_area = rw * rh
    return (cw * ch) / reference_area if reference_area else 0.0


def _center_offset(candidate_box, reference_box) -> float:
    cx, cy, cw, ch = candidate_box
    rx, ry, rw, rh = reference_box
    dx = (cx + cw / 2) - (rx + rw / 2)
    dy = (cy + ch / 2) - (ry + rh / 2)
    diagonal = (rw**2 + rh**2) ** 0.5
    return ((dx**2 + dy**2) ** 0.5) / diagonal if diagonal else 0.0
