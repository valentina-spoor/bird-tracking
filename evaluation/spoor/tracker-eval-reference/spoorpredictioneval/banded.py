"""Confidence-banded per-frame detection comparison (design principle P3: never a single threshold).

Compares a candidate ``y'`` against a reference ``y'`` with the reference treated as truth. For each
frame the boxes are matched by IoU; matched candidates count as found, unmatched candidates as
spurious, unmatched references as missed. Findings are bucketed by confidence: candidate-side counts
by candidate confidence (precision), reference-side counts by reference confidence (recall).

Identical candidate and reference therefore yield precision == recall == 1.0 in every non-empty band
(perfect regression, no drift). Any drop localises to a band and a side.
"""

from __future__ import annotations

from dataclasses import dataclass

from spoorpredictioneval.contracts import FrameDetections
from spoorpredictioneval.metrics import match_boxes_by_iou

BAND_EDGES: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 1.01)
"""Left-closed, right-open confidence band edges. Last edge > 1.0 so confidence 1.0 lands inside."""


def _band_index(confidence: float, edges: tuple[float, ...]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= confidence < edges[index + 1]:
            return index
    return len(edges) - 2  # clamp anything outside the range into the top band


@dataclass(frozen=True)
class BandResult:
    """Per-band tallies. Empty band (no candidate and no reference boxes) reads as perfect."""

    low: float
    high: float
    matched: int = 0  # candidate boxes that found a reference (banded by candidate confidence)
    spurious: int = 0  # candidate boxes with no reference (banded by candidate confidence)
    reference_matched: int = 0  # reference boxes that were found (banded by reference confidence)
    missed: int = 0  # reference boxes with no candidate (banded by reference confidence)

    @property
    def precision(self) -> float:
        denominator = self.matched + self.spurious
        return self.matched / denominator if denominator > 0 else 1.0

    @property
    def recall(self) -> float:
        denominator = self.reference_matched + self.missed
        return self.reference_matched / denominator if denominator > 0 else 1.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def banded_detection(
    candidate: FrameDetections,
    reference: FrameDetections,
    iou_threshold: float,
    band_edges: tuple[float, ...] = BAND_EDGES,
) -> list[BandResult]:
    """Compare two per-frame detection sets, returning one BandResult per confidence band."""
    band_count = len(band_edges) - 1
    matched = [0] * band_count
    spurious = [0] * band_count
    reference_matched = [0] * band_count
    missed = [0] * band_count

    for frame_number in sorted(set(candidate) | set(reference)):
        candidate_detections = candidate.get(frame_number, [])
        reference_detections = reference.get(frame_number, [])
        result = match_boxes_by_iou(
            [detection.box_xywh for detection in candidate_detections],
            [detection.box_xywh for detection in reference_detections],
            iou_threshold,
        )
        for candidate_index, reference_index in result.matched:
            matched[_band_index(candidate_detections[candidate_index].confidence, band_edges)] += 1
            reference_matched[_band_index(reference_detections[reference_index].confidence, band_edges)] += 1
        for candidate_index in result.unmatched_candidates:
            spurious[_band_index(candidate_detections[candidate_index].confidence, band_edges)] += 1
        for reference_index in result.unmatched_references:
            missed[_band_index(reference_detections[reference_index].confidence, band_edges)] += 1

    return [
        BandResult(
            low=band_edges[index],
            high=band_edges[index + 1],
            matched=matched[index],
            spurious=spurious[index],
            reference_matched=reference_matched[index],
            missed=missed[index],
        )
        for index in range(band_count)
    ]
