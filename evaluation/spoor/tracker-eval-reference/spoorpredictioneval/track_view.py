"""Reconstruct distinct tracks from track updates and summarise a run into a sane, readable view.

``reconstruct_tracks`` is the projection over the track-update stream (one ``Track`` per ``track_id``).
``summarize_run`` produces the descriptive instrument: how many observations, how many distinct
tracks, how they distribute across confidence bands and lifespans, and how many survive an explicit
salience lens. It editorialises nothing; it reports structure and the filter it was given.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from spoorpredictioneval.banded import BAND_EDGES
from spoorpredictioneval.contracts import RunResult, Track, TrackUpdate

_LIFESPAN_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-29", 10, 29),
    ("30+", 30, 10**9),
)


def reconstruct_tracks(track_updates: Iterable[tuple[str, TrackUpdate]]) -> list[Track]:
    """Group ``(track_id, update)`` pairs into ``Track``s, updates ordered by frame.

    Deterministic: tracks are returned ordered by ``(first_frame, track_id)``.
    """
    grouped: dict[str, list[TrackUpdate]] = defaultdict(list)
    for track_id, update in track_updates:
        grouped[track_id].append(update)
    tracks = [
        Track(track_id=track_id, updates=tuple(sorted(updates, key=lambda update: update.frame_number)))
        for track_id, updates in grouped.items()
    ]
    return sorted(tracks, key=lambda track: (track.first_frame, track.track_id))


@dataclass(frozen=True)
class SalienceFilter:
    """An explicit lens for "which tracks look real": long-lived AND confident on average."""

    min_lifespan: int = 30
    min_mean_confidence: float = 0.7

    def keeps(self, track: Track) -> bool:
        return track.lifespan >= self.min_lifespan and track.mean_confidence >= self.min_mean_confidence


@dataclass(frozen=True)
class BandPopulation:
    low: float
    high: float
    observations: int  # detections whose confidence falls in this band
    tracks: int  # tracks whose max_confidence falls in this band


@dataclass(frozen=True)
class RunView:
    observation_count: int
    track_count: int
    bands: list[BandPopulation]
    lifespan_histogram: list[tuple[str, int]]
    salience: SalienceFilter
    salient_track_count: int
    salient_track_ids: list[str]


def _band_index(confidence: float, edges: tuple[float, ...]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= confidence < edges[index + 1]:
            return index
    return len(edges) - 2


def summarize_run(
    run: RunResult, salience: SalienceFilter | None = None, band_edges: tuple[float, ...] = BAND_EDGES
) -> RunView:
    salience = salience or SalienceFilter()
    band_count = len(band_edges) - 1

    observation_confidences = [
        detection.confidence for detections in run.correctness.values() for detection in detections
    ]
    observations_per_band = [0] * band_count
    for confidence in observation_confidences:
        observations_per_band[_band_index(confidence, band_edges)] += 1

    tracks_per_band = [0] * band_count
    for track in run.tracks:
        tracks_per_band[_band_index(track.max_confidence, band_edges)] += 1

    bands = [
        BandPopulation(
            low=band_edges[index],
            high=band_edges[index + 1],
            observations=observations_per_band[index],
            tracks=tracks_per_band[index],
        )
        for index in range(band_count)
    ]

    lifespan_histogram = [
        (label, sum(1 for track in run.tracks if low <= track.lifespan <= high))
        for label, low, high in _LIFESPAN_BUCKETS
    ]

    salient = [track for track in run.tracks if salience.keeps(track)]
    return RunView(
        observation_count=len(observation_confidences),
        track_count=len(run.tracks),
        bands=bands,
        lifespan_histogram=lifespan_histogram,
        salience=salience,
        salient_track_count=len(salient),
        salient_track_ids=[track.track_id for track in salient],
    )
