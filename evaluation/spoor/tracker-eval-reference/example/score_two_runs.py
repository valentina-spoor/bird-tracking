"""Worked example: build two tracker runs, store them, score one against the other.

    python3 example/score_two_runs.py

Nothing here touches a database, S3 or a real video. Two runs are built in memory from
synthetic detections, written to a local JSONL store, then resolved back by id and
compared. This is the whole evaluation loop in one file; read it alongside
`spoorpredictioneval/compare.py`.

The reference run is a clean two-object sequence. The candidate is the same sequence with
three defects injected, each of which a different part of the report should catch:

  * one missed detection          -> recall drop, in the confidence band it belongs to
  * one spurious detection        -> precision drop
  * one identity swap mid-track   -> identity switches, IDF1, fragmentation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from spoorpredictioneval.compare import evaluate_refs, render_report
from spoorpredictioneval.contracts import Detection, Provenance, RunResult, TrackUpdate
from spoorpredictioneval.jsonl_store import JsonlResultStore
from spoorpredictioneval.regression import Tolerance
from spoorpredictioneval.track_quality import ASSOCIATION_CENTER_DISTANCE
from spoorpredictioneval.track_view import reconstruct_tracks

FRAMES = 40


def _walk(start_x: float, start_y: float, step_x: float, step_y: float):
    """A straight-line trajectory: one (x, y) centre per frame."""
    return [(start_x + step_x * frame, start_y + step_y * frame) for frame in range(FRAMES)]


def _build(paths: dict[str, list[tuple[float, float]]], *, drop=(), spurious=(), swap_at=None) -> RunResult:
    """Assemble a RunResult from per-track trajectories, optionally damaged.

    `drop` and `spurious` are (track_id, frame) pairs. `swap_at` is a frame from which the
    two tracks exchange identities, which is what a real tracker does when association
    fails across a crossing.
    """
    size = 12.0
    correctness: dict[int, list[Detection]] = {}
    updates: list[tuple[str, TrackUpdate]] = []
    seen: dict[str, int] = {}

    for track_id, centres in paths.items():
        for frame, (x, y) in enumerate(centres):
            if (track_id, frame) in drop:
                continue

            # after the swap frame the detections keep their position but change label
            label = track_id
            if swap_at is not None and frame >= swap_at:
                others = [t for t in paths if t != track_id]
                label = others[0]

            correctness.setdefault(frame, []).append(
                Detection(
                    box_xywh=(x - size / 2, y - size / 2, size, size),
                    confidence=0.92,
                    track_id=label,
                    class_name="bird",
                )
            )
            seen[label] = seen.get(label, 0) + 1
            updates.append(
                (
                    label,
                    TrackUpdate(
                        frame_number=frame,
                        position_xy=(x, y),
                        confidence=0.92,
                        detection_count=seen[label],
                        missed_frame_count=0,
                        age_frames=frame,
                    ),
                )
            )

    for track_id, frame in spurious:
        correctness.setdefault(frame, []).append(
            Detection(box_xywh=(300.0, 300.0, size, size), confidence=0.55, track_id=track_id, class_name="bird")
        )

    return RunResult(
        correctness=correctness,
        performance={},
        provenance=Provenance(config_hash="example", git_commit="0000000", model_hash="example"),
        tracks=tuple(reconstruct_tracks(updates)),
    )


def main() -> int:
    paths = {"a": _walk(100.0, 100.0, 4.0, 1.0), "b": _walk(300.0, 120.0, -4.0, 1.0)}

    reference = _build(paths)
    candidate = _build(paths, drop=(("a", 12),), spurious=(("c", 20),), swap_at=25)

    with tempfile.TemporaryDirectory() as root:
        store = JsonlResultStore(Path(root))
        reference_ref = store.persist(reference)
        candidate_ref = store.persist(candidate)

        # center_distance, not IoU: on small fast targets a few pixels of offset drives IoU
        # to zero while the detection is plainly the same object.
        report = evaluate_refs(
            store,
            candidate_ref,
            reference_ref,
            tolerance=Tolerance(iou_threshold=0.5, max_precision_drop=0.02, max_recall_drop=0.02),
            association=ASSOCIATION_CENTER_DISTANCE,
            association_threshold=8.0,
            confidence_floor=0.7,
        )

        print(f"candidate {candidate_ref}  vs  reference {reference_ref}\n")
        print(render_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
