"""JSONL implementation of the ResultStore port. Stdlib only.

One file per run, ``<run_id>.jsonl``: a provenance line, a performance line, then one detection line
per detection. This is the first stone; the format is an implementation detail behind the port, not
a public contract.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from spoorpredictioneval.contracts import (
    Detection,
    FrameDetections,
    Provenance,
    RunResult,
    StageStat,
    StageTiming,
    Track,
    TrackUpdate,
)
from spoorpredictioneval.result_store import content_run_id


def write_run_result_jsonl(path: Path | str, result: RunResult) -> None:
    """Serialize a RunResult to a JSONL file."""
    with Path(path).open("w") as handle:
        handle.write(json.dumps({"kind": "provenance", **_provenance_to_json(result.provenance)}) + "\n")
        handle.write(json.dumps({"kind": "performance", "stages": _performance_to_json(result.performance)}) + "\n")
        for frame_number in sorted(result.correctness):
            for detection in result.correctness[frame_number]:
                handle.write(
                    json.dumps({"kind": "detection", "frame": frame_number, **_detection_to_json(detection)}) + "\n"
                )
        for track in result.tracks:
            handle.write(json.dumps({"kind": "track", **_track_to_json(track)}) + "\n")


def read_run_result_jsonl(path: Path | str) -> RunResult:
    """Rebuild a RunResult from a JSONL file written by ``write_run_result_jsonl``."""
    provenance: Provenance | None = None
    performance: StageTiming = {}
    correctness: FrameDetections = defaultdict(list)
    tracks: list[Track] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record["kind"]
        if kind == "provenance":
            provenance = _provenance_from_json(record)
        elif kind == "performance":
            performance = _performance_from_json(record["stages"])
        elif kind == "detection":
            correctness[record["frame"]].append(_detection_from_json(record))
        elif kind == "track":
            tracks.append(_track_from_json(record))
        else:
            raise ValueError(f"Unknown record kind {kind!r} in {path}")

    if provenance is None:
        raise ValueError(f"No provenance line in {path}")
    return RunResult(
        correctness=dict(correctness),
        performance=performance,
        provenance=provenance,
        tracks=tuple(tracks),
    )


class JsonlResultStore:
    """Persist/resolve runs as JSONL files under ``root``. The offline first stone behind the port."""

    def __init__(self, root: Path | str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def persist(self, result: RunResult) -> str:
        run_id = content_run_id(result)
        write_run_result_jsonl(self._root / f"{run_id}.jsonl", result)
        return run_id

    def resolve(self, run_id: str) -> RunResult:
        path = self._root / f"{run_id}.jsonl"
        if not path.exists():
            raise KeyError(f"No run {run_id!r} in {self._root}")
        return read_run_result_jsonl(path)


def _provenance_to_json(provenance: Provenance) -> dict:
    return {
        "config_hash": provenance.config_hash,
        "git_commit": provenance.git_commit,
        "model_hash": provenance.model_hash,
        "extra": provenance.extra,
    }


def _provenance_from_json(record: dict) -> Provenance:
    return Provenance(
        config_hash=record["config_hash"],
        git_commit=record["git_commit"],
        model_hash=record["model_hash"],
        extra=record.get("extra", {}),
    )


def _performance_to_json(performance: StageTiming) -> dict:
    return {stage: {"calls": stat.calls, "total_seconds": stat.total_seconds} for stage, stat in performance.items()}


def _performance_from_json(stages: dict) -> StageTiming:
    return {
        stage: StageStat(calls=value["calls"], total_seconds=value["total_seconds"]) for stage, value in stages.items()
    }


def _detection_to_json(detection: Detection) -> dict:
    return {
        "box": list(detection.box_xywh),
        "confidence": detection.confidence,
        "track_id": detection.track_id,
        "class_name": detection.class_name,
    }


def _detection_from_json(record: dict) -> Detection:
    box = record["box"]
    return Detection(
        box_xywh=(box[0], box[1], box[2], box[3]),
        confidence=record["confidence"],
        track_id=record.get("track_id"),
        class_name=record.get("class_name"),
    )


def _track_to_json(track: Track) -> dict:
    return {
        "track_id": track.track_id,
        "updates": [
            [
                u.frame_number,
                u.position_xy[0],
                u.position_xy[1],
                u.confidence,
                u.detection_count,
                u.missed_frame_count,
                u.age_frames,
            ]
            for u in track.updates
        ],
    }


def _track_from_json(record: dict) -> Track:
    updates = tuple(
        TrackUpdate(
            frame_number=row[0],
            position_xy=(row[1], row[2]),
            confidence=row[3],
            detection_count=row[4],
            missed_frame_count=row[5],
            age_frames=row[6],
        )
        for row in record["updates"]
    )
    return Track(track_id=record["track_id"], updates=updates)
