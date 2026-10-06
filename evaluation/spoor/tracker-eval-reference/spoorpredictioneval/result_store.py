"""The persistence port and the content-addressed run id.

The evaluator depends on this ``ResultStore`` Protocol, never on a concrete backend. The first (and
currently only) implementation is ``JsonlResultStore``. A ``dataset-manager``-backed implementation
is the intended productization drop-in and would satisfy this same Protocol; it is not built here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from spoorpredictioneval.contracts import RunResult


class ResultStore(Protocol):
    """Persist a run and resolve one back by id. Ids are opaque to callers."""

    def persist(self, result: RunResult) -> str: ...

    def resolve(self, run_id: str) -> RunResult: ...


def content_run_id(result: RunResult) -> str:
    """A deterministic id derived from provenance plus output content.

    No clock or randomness, so a byte-identical run yields the same id (content addressing). Distinct
    output yields a distinct id. Callers wanting to force distinct ids for identical output can vary
    ``provenance.extra`` (e.g. a label).
    """
    provenance = result.provenance
    payload = {
        "config_hash": provenance.config_hash,
        "git_commit": provenance.git_commit,
        "model_hash": provenance.model_hash,
        "extra": provenance.extra,
        "correctness": [
            [frame, detection.box_xywh, detection.confidence, detection.track_id, detection.class_name]
            for frame in sorted(result.correctness)
            for detection in result.correctness[frame]
        ],
        "performance": sorted((stage, stat.calls, stat.total_seconds) for stage, stat in result.performance.items()),
        "tracks": sorted(
            [
                track.track_id,
                track.first_frame,
                track.last_frame,
                track.observation_count,
                round(track.mean_confidence, 6),
            ]
            for track in result.tracks
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]
