"""Identity-continuity metrics for a tracker run, plus reference-free track signals.

Two vantage points:

- **Against a reference** (``evaluate_track_quality``): IDF1, identity switches (IDSW), and
  fragmentation (FM), computed from a per-frame IoU association between a candidate run and a
  reference run whose detections carry trajectory ``track_id``s.
- **From a single run** (``track_signals``): counts and distributions that need no reference at
  all: how many tracks, how long they are, and how gappy their observations are.

Association pairs each reference box to at most one candidate box per frame, one-to-one and greedy,
under one of two measures selected by the caller:

- ``"center_distance"`` (default): nearest box-center first, eligible at or below a pixel threshold
  (``metrics.match_boxes_by_center_distance``). This isolates *identity* continuity for small,
  point-like targets, where a few-pixel box-tightness disagreement between two runs of the same
  object would otherwise be misread as a lost/reacquired identity.
- ``"iou"``: highest intersection-over-union first, eligible at or above an IoU threshold
  (``metrics.match_boxes_by_iou``). Appropriate only when box overlap is itself the thing being
  judged.

Both break ties by lowest candidate index then lowest reference index. A detection lacking a
``track_id`` carries no identity: it may still cover a reference frame (so it can suppress a
fragmentation) but it never contributes an identity match, and reference detections without a
``track_id`` are ignored entirely because they define no trajectory.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal, Optional

from spoorpredictioneval.contracts import Box, RunResult
from spoorpredictioneval.metrics import (
    Matching,
    match_boxes_by_center_distance,
    match_boxes_by_iou,
)

AssociationMode = Literal["center_distance", "iou"]

ASSOCIATION_CENTER_DISTANCE: AssociationMode = "center_distance"
ASSOCIATION_IOU: AssociationMode = "iou"
DEFAULT_ASSOCIATION: AssociationMode = ASSOCIATION_CENTER_DISTANCE

DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_CENTER_DISTANCE_THRESHOLD = 20.0
DEFAULT_LENGTH_HISTOGRAM_EDGES = (2, 5, 10, 25)


@dataclass(frozen=True)
class TrackSignals:
    """Reference-free description of the tracks in one run.

    ``track_count`` is the number of distinct non-null ``track_id``s. A track's *length* is the
    number of frames in which it was observed (its detection count), not its first-to-last span.
    ``length_histogram`` has ``len(length_histogram_edges) + 1`` buckets: bucket ``i`` counts
    tracks whose length is at least ``length_histogram_edges[i - 1]`` and below
    ``length_histogram_edges[i]``, with virtual bounds of zero below the first edge and infinity
    above the last. A *gap* is a run of missing frames between two consecutive observations of the
    same track; ``gap_count`` totals such gaps across all tracks, ``max_gap`` is the largest single
    gap in frames, and ``total_gap_frames`` sums every gap's length. An empty run yields zero
    counts and zero-valued statistics.
    """

    track_count: int
    min_length: int
    median_length: float
    max_length: int
    length_histogram: tuple[int, ...]
    length_histogram_edges: tuple[int, ...]
    gap_count: int
    tracks_with_gaps: int
    max_gap: int
    total_gap_frames: int


@dataclass(frozen=True)
class TrackQualityReport:
    """Identity-continuity metrics of a candidate run against a reference run.

    ``idf1`` is the identity F1 in ``[0.0, 1.0]``: ``2 * IDTP / (2 * IDTP + IDFP + IDFN)`` over the
    globally optimal one-to-one assignment of candidate ``track_id``s to reference trajectories,
    where IDTP is the number of correctly-identified detections under that assignment. It is
    ``0.0`` when there are no identified detections on either side (an empty candidate never
    raises). ``identity_switches`` (IDSW) counts, per reference trajectory, how many times the
    candidate id assigned to it changes across its observed frames; a gap does not reset the
    remembered id, so a drop-and-reacquire with the same id is not a switch, and a mutual swap of
    two objects' ids counts as two switches. ``fragmentation`` (FM) counts, per reference
    trajectory, how many times coverage is interrupted and then resumes, independent of which
    candidate id does the covering. ``candidate_signals`` are the reference-free signals of the
    candidate run. ``association`` records the association measure used and ``association_threshold``
    its threshold (pixels for center-distance, an IoU fraction for IoU).
    """

    idf1: float
    identity_switches: int
    fragmentation: int
    id_true_positives: int
    id_false_positives: int
    id_false_negatives: int
    association: AssociationMode
    association_threshold: float
    candidate_signals: TrackSignals

    @property
    def id_precision(self) -> float:
        denominator = self.id_true_positives + self.id_false_positives
        return self.id_true_positives / denominator if denominator > 0 else 0.0

    @property
    def id_recall(self) -> float:
        denominator = self.id_true_positives + self.id_false_negatives
        return self.id_true_positives / denominator if denominator > 0 else 0.0


@dataclass(frozen=True)
class _ReferenceFrame:
    """One observed frame of a reference trajectory: whether a candidate covered it, and with which id."""

    frame_number: int
    covered: bool
    candidate_id: Optional[str]


def track_signals(
    result: RunResult,
    length_histogram_edges: tuple[int, ...] = DEFAULT_LENGTH_HISTOGRAM_EDGES,
) -> TrackSignals:
    """Summarize the tracks of ``result`` without any reference.

    Tracks are reconstructed from ``result.correctness`` by grouping detections on their
    ``track_id``; detections whose ``track_id`` is ``None`` are ignored. See ``TrackSignals`` for
    the meaning of every field. ``length_histogram_edges`` must be ascending positive integers.
    """
    frames_by_track = _frames_by_track(result)
    lengths = [len(frames) for frames in frames_by_track.values()]

    gaps: list[int] = []
    tracks_with_gaps = 0
    for frames in frames_by_track.values():
        track_gaps = [later - earlier - 1 for earlier, later in zip(frames, frames[1:]) if later - earlier - 1 > 0]
        if track_gaps:
            tracks_with_gaps += 1
        gaps.extend(track_gaps)

    return TrackSignals(
        track_count=len(frames_by_track),
        min_length=min(lengths) if lengths else 0,
        median_length=statistics.median(lengths) if lengths else 0.0,
        max_length=max(lengths) if lengths else 0,
        length_histogram=_histogram(lengths, length_histogram_edges),
        length_histogram_edges=tuple(length_histogram_edges),
        gap_count=len(gaps),
        tracks_with_gaps=tracks_with_gaps,
        max_gap=max(gaps) if gaps else 0,
        total_gap_frames=sum(gaps),
    )


def evaluate_track_quality(
    candidate: RunResult,
    reference: RunResult,
    association: AssociationMode = DEFAULT_ASSOCIATION,
    threshold: Optional[float] = None,
    length_histogram_edges: tuple[int, ...] = DEFAULT_LENGTH_HISTOGRAM_EDGES,
) -> TrackQualityReport:
    """Score ``candidate``'s identity continuity against ``reference``.

    Every reference detection with a ``track_id`` is associated to at most one candidate detection
    per frame under ``association``: ``"center_distance"`` (default) matches nearest box-center first
    and treats ``threshold`` as a maximum center-to-center distance in pixels; ``"iou"`` matches
    highest overlap first and treats ``threshold`` as a minimum IoU fraction. When ``threshold`` is
    ``None`` the measure's default is used (``DEFAULT_CENTER_DISTANCE_THRESHOLD`` /
    ``DEFAULT_IOU_THRESHOLD``). From that association it computes IDF1, IDSW, and FM as defined on
    ``TrackQualityReport``. Never raises on an empty candidate: with no candidate detections IDF1 is
    ``0.0`` and IDSW/FM are ``0``. Raises ``ValueError`` for an unknown ``association``.
    """
    if threshold is None:
        threshold = (
            DEFAULT_CENTER_DISTANCE_THRESHOLD if association == ASSOCIATION_CENTER_DISTANCE else DEFAULT_IOU_THRESHOLD
        )
    reference_trajectories = _associate(candidate, reference, association, threshold)

    cooccurrence = _identity_cooccurrence(reference_trajectories)
    id_true_positives = _optimal_identity_agreement(cooccurrence)
    total_reference = sum(len(frames) for frames in reference_trajectories.values())
    total_candidate = sum(
        1 for detections in candidate.correctness.values() for detection in detections if detection.track_id is not None
    )
    id_false_negatives = total_reference - id_true_positives
    id_false_positives = total_candidate - id_true_positives
    denominator = 2 * id_true_positives + id_false_positives + id_false_negatives
    idf1 = (2 * id_true_positives / denominator) if denominator > 0 else 0.0

    return TrackQualityReport(
        idf1=idf1,
        identity_switches=_identity_switches(reference_trajectories),
        fragmentation=_fragmentation(reference_trajectories),
        id_true_positives=id_true_positives,
        id_false_positives=id_false_positives,
        id_false_negatives=id_false_negatives,
        association=association,
        association_threshold=threshold,
        candidate_signals=track_signals(candidate, length_histogram_edges),
    )


def _frames_by_track(result: RunResult) -> dict[str, list[int]]:
    """Sorted, de-duplicated observation frames per non-null ``track_id`` in ``result.correctness``."""
    frames_by_track: dict[str, set[int]] = {}
    for frame_number, detections in result.correctness.items():
        for detection in detections:
            if detection.track_id is None:
                continue
            frames_by_track.setdefault(detection.track_id, set()).add(frame_number)
    return {track_id: sorted(frames) for track_id, frames in frames_by_track.items()}


def _associate(
    candidate: RunResult,
    reference: RunResult,
    association: AssociationMode,
    threshold: float,
) -> dict[str, list[_ReferenceFrame]]:
    """Per reference trajectory, its observed frames ordered in time with the covering candidate id.

    Reference detections without a ``track_id`` are dropped. Within a frame, reference boxes are
    matched one-to-one against all candidate boxes under ``association`` at ``threshold``; a matched
    reference frame is covered, and its ``candidate_id`` is the matched candidate's ``track_id``
    (possibly ``None``).
    """
    trajectories: dict[str, list[_ReferenceFrame]] = {}
    for frame_number in sorted(reference.correctness):
        reference_detections = [d for d in reference.correctness[frame_number] if d.track_id is not None]
        if not reference_detections:
            continue
        candidate_detections = candidate.correctness.get(frame_number, [])

        matching = _match(
            [detection.box_xywh for detection in candidate_detections],
            [detection.box_xywh for detection in reference_detections],
            association,
            threshold,
        )
        candidate_for_reference = {
            reference_index: candidate_index for candidate_index, reference_index in matching.matched
        }

        for reference_index, reference_detection in enumerate(reference_detections):
            candidate_index = candidate_for_reference.get(reference_index)
            covered = candidate_index is not None
            candidate_id = candidate_detections[candidate_index].track_id if covered else None
            trajectories.setdefault(reference_detection.track_id, []).append(
                _ReferenceFrame(frame_number, covered, candidate_id)
            )
    return trajectories


def _match(
    candidate_boxes: list[Box], reference_boxes: list[Box], association: AssociationMode, threshold: float
) -> Matching:
    """Dispatch a one-to-one candidate-to-reference box matching to the chosen association measure."""
    if association == ASSOCIATION_CENTER_DISTANCE:
        return match_boxes_by_center_distance(candidate_boxes, reference_boxes, threshold)
    if association == ASSOCIATION_IOU:
        return match_boxes_by_iou(candidate_boxes, reference_boxes, threshold)
    raise ValueError(
        f"Unknown association mode {association!r}; expected {ASSOCIATION_CENTER_DISTANCE!r} or {ASSOCIATION_IOU!r}"
    )


def _identity_cooccurrence(trajectories: dict[str, list[_ReferenceFrame]]) -> dict[tuple[str, str], int]:
    """Count of frames where reference id ``r`` was covered by candidate id ``c``, keyed by ``(r, c)``.

    Frames covered by a null candidate id contribute nothing.
    """
    cooccurrence: dict[tuple[str, str], int] = {}
    for reference_id, frames in trajectories.items():
        for frame in frames:
            if frame.candidate_id is None:
                continue
            key = (reference_id, frame.candidate_id)
            cooccurrence[key] = cooccurrence.get(key, 0) + 1
    return cooccurrence


def _identity_switches(trajectories: dict[str, list[_ReferenceFrame]]) -> int:
    """Times the remembered candidate id of a reference trajectory changes across its covered frames."""
    switches = 0
    for frames in trajectories.values():
        last_identity: Optional[str] = None
        for frame in frames:
            if frame.candidate_id is None:
                continue
            if last_identity is not None and frame.candidate_id != last_identity:
                switches += 1
            last_identity = frame.candidate_id
    return switches


def _fragmentation(trajectories: dict[str, list[_ReferenceFrame]]) -> int:
    """Coverage interruptions that later resume, summed over trajectories: covered-run count minus one."""
    fragments = 0
    for frames in trajectories.values():
        covered_runs = 0
        previously_covered = False
        for frame in frames:
            if frame.covered and not previously_covered:
                covered_runs += 1
            previously_covered = frame.covered
        fragments += max(0, covered_runs - 1)
    return fragments


def _optimal_identity_agreement(cooccurrence: dict[tuple[str, str], int]) -> int:
    """Max total co-occurrence over a one-to-one reference-id-to-candidate-id assignment (IDTP)."""
    reference_ids = sorted({reference_id for reference_id, _ in cooccurrence})
    candidate_ids = sorted({candidate_id for _, candidate_id in cooccurrence})
    if not reference_ids or not candidate_ids:
        return 0

    weights = [
        [cooccurrence.get((reference_id, candidate_id), 0) for candidate_id in candidate_ids]
        for reference_id in reference_ids
    ]
    assignment = _max_weight_assignment(weights)
    return sum(
        weights[row][column]
        for row, column in enumerate(assignment)
        if row < len(reference_ids) and column < len(candidate_ids)
    )


def _max_weight_assignment(weights: list[list[int]]) -> list[int]:
    """Assignment maximizing summed weight (Hungarian algorithm on a padded square matrix).

    Returns ``assignment`` where row ``i`` is paired with column ``assignment[i]``. Rows or columns
    added to square the matrix carry zero weight; callers ignore pairs that fall outside the real
    dimensions.
    """
    row_count = len(weights)
    column_count = len(weights[0]) if weights else 0
    size = max(row_count, column_count)
    infinity = float("inf")
    cost = [[-weights[i][j] if i < row_count and j < column_count else 0 for j in range(size)] for i in range(size)]

    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    column_match = [0] * (size + 1)
    parent_column = [0] * (size + 1)

    for row in range(1, size + 1):
        column_match[0] = row
        current_column = 0
        min_slack = [infinity] * (size + 1)
        visited = [False] * (size + 1)
        while True:
            visited[current_column] = True
            matched_row = column_match[current_column]
            delta = infinity
            next_column = -1
            for column in range(1, size + 1):
                if visited[column]:
                    continue
                slack = cost[matched_row - 1][column - 1] - row_potential[matched_row] - column_potential[column]
                if slack < min_slack[column]:
                    min_slack[column] = slack
                    parent_column[column] = current_column
                if min_slack[column] < delta:
                    delta = min_slack[column]
                    next_column = column
            for column in range(size + 1):
                if visited[column]:
                    row_potential[column_match[column]] += delta
                    column_potential[column] -= delta
                else:
                    min_slack[column] -= delta
            current_column = next_column
            if column_match[current_column] == 0:
                break
        while current_column != 0:
            previous_column = parent_column[current_column]
            column_match[current_column] = column_match[previous_column]
            current_column = previous_column

    assignment = [0] * size
    for column in range(1, size + 1):
        if column_match[column] != 0:
            assignment[column_match[column] - 1] = column - 1
    return assignment


def _histogram(lengths: list[int], edges: tuple[int, ...]) -> tuple[int, ...]:
    """Bucket ``lengths`` into ``len(edges) + 1`` counts against ascending ``edges`` (upper-exclusive)."""
    counts = [0] * (len(edges) + 1)
    for length in lengths:
        bucket = sum(1 for edge in edges if length >= edge)
        counts[bucket] += 1
    return tuple(counts)
