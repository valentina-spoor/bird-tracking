"""Fresh, dependency-free detection-matching primitives.

IoU and center-distance on xywh boxes, greedy one-to-one matching between two box lists under
either measure, and a small TP/FP/FN accumulator. Written here on purpose so the evaluator consumes
no legacy metric code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from spoorpredictioneval.contracts import Box


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``(x, y, w, h)`` boxes. 0.0 when they do not overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(a: Box, b: Box) -> float:
    """Euclidean distance in pixels between the centers of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return math.hypot((ax + aw / 2.0) - (bx + bw / 2.0), (ay + ah / 2.0) - (by + bh / 2.0))


@dataclass(frozen=True)
class Matching:
    """Result of matching candidate boxes to reference boxes.

    ``matched`` holds ``(candidate_index, reference_index)`` pairs; the unmatched lists hold the
    indices with no partner at or above the IoU threshold.
    """

    matched: list[tuple[int, int]]
    unmatched_candidates: list[int]
    unmatched_references: list[int]


def match_boxes_by_iou(candidate: list[Box], reference: list[Box], iou_threshold: float) -> Matching:
    """Greedy one-to-one matching, highest IoU first.

    Deterministic: candidate pairs are ranked by ``(iou, candidate_index, reference_index)`` so ties
    resolve the same way every run. A box matches at most once.
    """
    scored = [
        (iou(candidate_box, reference_box), candidate_index, reference_index)
        for candidate_index, candidate_box in enumerate(candidate)
        for reference_index, reference_box in enumerate(reference)
        if iou(candidate_box, reference_box) >= iou_threshold
    ]
    # Highest IoU first; ties broken by lowest candidate then reference index (deterministic).
    scored.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))

    used_candidates: set[int] = set()
    used_references: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _score, candidate_index, reference_index in scored:
        if candidate_index in used_candidates or reference_index in used_references:
            continue
        matched.append((candidate_index, reference_index))
        used_candidates.add(candidate_index)
        used_references.add(reference_index)

    unmatched_candidates = [i for i in range(len(candidate)) if i not in used_candidates]
    unmatched_references = [i for i in range(len(reference)) if i not in used_references]
    return Matching(matched, unmatched_candidates, unmatched_references)


def match_boxes_by_center_distance(candidate: list[Box], reference: list[Box], max_center_distance: float) -> Matching:
    """Greedy one-to-one matching, nearest box-center first.

    A pair is eligible when its center-to-center distance is at or below ``max_center_distance``
    pixels; the nearest eligible pair is bound first and each box binds at most once. Deterministic:
    pairs are ranked by ``(distance, candidate_index, reference_index)`` so ties resolve the same way
    every run. Unlike IoU, this does not require the boxes to overlap, so a few-pixel size or position
    disagreement between two runs of the same small object does not break the match.
    """
    scored = [
        (center_distance(candidate_box, reference_box), candidate_index, reference_index)
        for candidate_index, candidate_box in enumerate(candidate)
        for reference_index, reference_box in enumerate(reference)
        if center_distance(candidate_box, reference_box) <= max_center_distance
    ]
    # Nearest first; ties broken by lowest candidate then reference index (deterministic).
    scored.sort(key=lambda entry: (entry[0], entry[1], entry[2]))

    used_candidates: set[int] = set()
    used_references: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _distance, candidate_index, reference_index in scored:
        if candidate_index in used_candidates or reference_index in used_references:
            continue
        matched.append((candidate_index, reference_index))
        used_candidates.add(candidate_index)
        used_references.add(reference_index)

    unmatched_candidates = [i for i in range(len(candidate)) if i not in used_candidates]
    unmatched_references = [i for i in range(len(reference)) if i not in used_references]
    return Matching(matched, unmatched_candidates, unmatched_references)


@dataclass(frozen=True)
class Counts:
    """TP/FP/FN accumulator with derived precision/recall/f1. Empty -> all metrics 0.0."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
