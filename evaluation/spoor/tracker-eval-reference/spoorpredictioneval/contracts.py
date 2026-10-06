"""Canonical evaluation inputs and the run container.

These are the shapes an adapter must produce from a predictor's public output ``y``. The evaluator
speaks only these; it never knows how a predictor produced them.

Three terms are kept strictly distinct, because conflating them makes the evaluator lie (a two-bird
clip produced 2528 observations across 198 tracks; the predictor's own logs printed "125 active
tracks", which was neither):

- **observation** — one detection in one frame: a box + confidence, tagged with the ``track_id`` it
  fed. Modelled by ``Detection`` inside ``FrameDetections``.
- **track update** — the tracker's per-frame emission of one track's evolving state (position,
  confidence, age, detection/missed counts). A per-frame event, NOT a track. Modelled by ``TrackUpdate``.
- **track** — a distinct object hypothesis, one stable ``track_id``, spanning the frames it was held.
  Its trajectory / lifespan / confidence are the projection over its track updates. Modelled by ``Track``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Box = tuple[float, float, float, float]
"""A bounding box as ``(x, y, width, height)`` in image pixels."""


@dataclass(frozen=True)
class Detection:
    """One observation: a single detection in a single frame, tagged with the track it fed."""

    box_xywh: Box
    confidence: float
    track_id: Optional[str] = None
    class_name: Optional[str] = None


FrameDetections = dict[int, list[Detection]]
"""``y'`` for correctness: source-frame number -> observations in that frame."""


@dataclass(frozen=True)
class TrackUpdate:
    """One track update: the tracker's state for one track at one frame. A per-frame event, not a track."""

    frame_number: int
    position_xy: tuple[float, float]
    confidence: float
    detection_count: int
    missed_frame_count: int
    age_frames: int


@dataclass(frozen=True)
class Track:
    """A distinct object hypothesis (one ``track_id``) as the projection over its ordered track updates."""

    track_id: str
    updates: tuple[TrackUpdate, ...]

    @property
    def first_frame(self) -> int:
        return self.updates[0].frame_number

    @property
    def last_frame(self) -> int:
        return self.updates[-1].frame_number

    @property
    def lifespan(self) -> int:
        """Frames spanned from first to last update, inclusive (held life, coasting included)."""
        return self.last_frame - self.first_frame + 1

    @property
    def observation_count(self) -> int:
        """Detections the tracker assigned to this track (its own running count)."""
        return max((update.detection_count for update in self.updates), default=0)

    @property
    def mean_confidence(self) -> float:
        return sum(update.confidence for update in self.updates) / len(self.updates) if self.updates else 0.0

    @property
    def max_confidence(self) -> float:
        return max((update.confidence for update in self.updates), default=0.0)

    @property
    def trajectory(self) -> tuple[tuple[int, float, float], ...]:
        return tuple((update.frame_number, *update.position_xy) for update in self.updates)


@dataclass(frozen=True)
class StageStat:
    """Accumulated timing for one pipeline stage over a run."""

    calls: int
    total_seconds: float

    @property
    def ms_per_call(self) -> float:
        return (self.total_seconds / self.calls * 1000.0) if self.calls > 0 else 0.0


StageTiming = dict[str, StageStat]
"""``y''`` for performance: stage name -> accumulated timing."""


@dataclass(frozen=True)
class Provenance:
    """What produced a run. Travels with the result so a comparison can be trusted."""

    config_hash: str
    git_commit: str
    model_hash: str
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LocalizationSummary:
    """Center-error distribution over matched candidate-reference detection pairs.

    Each matched pair contributes one pixel offset (euclidean distance between the two box centers,
    in source pixels) and one normalized offset (that pixel offset divided by the reference box
    diagonal). ``count`` is the number of matched pairs; the ``*_pixels`` and ``*_normalized`` fields
    summarize the two distributions. An empty matched set yields ``count == 0`` and 0.0 for every
    statistic.
    """

    count: int
    median_pixels: float
    p95_pixels: float
    max_pixels: float
    median_normalized: float
    p95_normalized: float
    max_normalized: float


@dataclass(frozen=True)
class RunResult:
    """A single predictor run's evaluation inputs plus its provenance.

    ``correctness`` holds observations (``y'``); ``tracks`` holds the reconstructed tracks (distinct
    ``track_id``s); ``performance`` holds per-stage timing (``y''``).
    """

    correctness: FrameDetections
    performance: StageTiming
    provenance: Provenance
    tracks: tuple[Track, ...] = ()
