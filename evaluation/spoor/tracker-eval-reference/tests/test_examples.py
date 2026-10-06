"""Five example tests, one per kind of thing worth testing.

These are illustrative, not a suite. Spoor's own library has far more, and this file is
not a substitute for them. The point is the *shapes*: when you write evaluation code for
the tracker, these five patterns will cover most of what you need.

Run them with:

    python3 -m pytest tests -q

Each test below is labelled with the pattern it demonstrates. Read them in order.
"""

from __future__ import annotations

from pathlib import Path

from spoorpredictioneval.contracts import Detection, Provenance, RunResult, TrackUpdate
from spoorpredictioneval.jsonl_store import JsonlResultStore
from spoorpredictioneval.metrics import iou
from spoorpredictioneval.result_store import content_run_id
from spoorpredictioneval.track_quality import ASSOCIATION_CENTER_DISTANCE, evaluate_track_quality
from spoorpredictioneval.track_view import reconstruct_tracks


# ---------------------------------------------------------------------------
# 1. A pure function, at its boundaries.
#
# `iou` is arithmetic with no state. The useful cases are not the middle, they are the
# edges: identical, disjoint, and partial. Test the edges and the middle takes care of
# itself.
# ---------------------------------------------------------------------------


def test_iou_is_one_for_identical_boxes():
    box = (10.0, 10.0, 20.0, 20.0)
    assert iou(box, box) == 1.0


def test_iou_is_zero_for_disjoint_boxes():
    assert iou((0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 10.0, 10.0)) == 0.0


def test_iou_of_half_overlapping_boxes_is_one_third():
    # two 10x10 boxes sharing a 5x10 strip: intersection 50, union 150
    assert iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 10.0, 10.0)) == 50 / 150


# ---------------------------------------------------------------------------
# 2. A transformation, checked by its structure.
#
# `reconstruct_tracks` turns a flat stream of per-frame updates into grouped Track
# objects. Assert the grouping, not the internals: how many tracks, which ids, how long.
# ---------------------------------------------------------------------------


def _update(frame: int, x: float) -> TrackUpdate:
    return TrackUpdate(
        frame_number=frame,
        position_xy=(x, 50.0),
        confidence=0.9,
        detection_count=1,
        missed_frame_count=0,
        age_frames=frame,
    )


def test_reconstruct_tracks_groups_updates_by_track_id():
    updates = [("a", _update(0, 10.0)), ("b", _update(0, 90.0)), ("a", _update(1, 14.0))]

    tracks = {track.track_id: track for track in reconstruct_tracks(updates)}

    assert set(tracks) == {"a", "b"}
    assert len(tracks["a"].updates) == 2
    assert len(tracks["b"].updates) == 1


# ---------------------------------------------------------------------------
# 3. A rule that is not obvious from reading the code.
#
# This is where a test earns its keep. "An identity switch is counted when the candidate
# track covering a reference object changes" is a sentence; this test is what makes it
# checkable. Build the smallest input that exhibits the rule and nothing else.
# ---------------------------------------------------------------------------


def _run(labels_by_frame: dict[int, list[tuple[str, float]]]) -> RunResult:
    """One object per label per frame, placed at the given x, box 10 wide."""
    correctness: dict[int, list[Detection]] = {}
    updates: list[tuple[str, TrackUpdate]] = []
    for frame, entries in labels_by_frame.items():
        for label, x in entries:
            correctness.setdefault(frame, []).append(
                Detection(box_xywh=(x, 50.0, 10.0, 10.0), confidence=0.9, track_id=label, class_name="bird")
            )
            updates.append((label, _update(frame, x + 5.0)))
    return RunResult(
        correctness=correctness,
        performance={},
        provenance=Provenance(config_hash="test", git_commit="0" * 7, model_hash="test"),
        tracks=tuple(reconstruct_tracks(updates)),
    )


def test_relabelling_one_object_midway_is_one_identity_switch():
    frames = {0: [("x", 10.0)], 1: [("x", 20.0)], 2: [("x", 30.0)], 3: [("x", 40.0)]}
    reference = _run(frames)
    # same boxes in the same places; only the identity the tracker assigned changes at frame 2
    candidate = _run({0: [("x", 10.0)], 1: [("x", 20.0)], 2: [("y", 30.0)], 3: [("y", 40.0)]})

    report = evaluate_track_quality(candidate, reference, association=ASSOCIATION_CENTER_DISTANCE, threshold=5.0)

    assert report.identity_switches == 1


def test_a_perfect_candidate_has_no_identity_switches():
    # The companion assertion. A rule you only test in its failing direction can be
    # satisfied by a function that always returns 1.
    frames = {0: [("x", 10.0)], 1: [("x", 20.0)], 2: [("x", 30.0)]}
    run = _run(frames)

    report = evaluate_track_quality(run, run, association=ASSOCIATION_CENTER_DISTANCE, threshold=5.0)

    assert report.identity_switches == 0
    assert report.idf1 == 1.0


# ---------------------------------------------------------------------------
# 4. A round trip through a port implementation.
#
# `JsonlResultStore` satisfies the `ResultStore` protocol. The contract is "what I persist,
# I can resolve". Test that, not the file format: the format is free to change, the
# contract is not. `tmp_path` is a pytest fixture giving a fresh directory per test.
# ---------------------------------------------------------------------------


def test_persisted_run_resolves_back_unchanged(tmp_path: Path):
    store = JsonlResultStore(tmp_path)
    original = _run({0: [("x", 10.0)], 1: [("x", 20.0)]})

    resolved = store.resolve(store.persist(original))

    assert resolved.correctness == original.correctness
    assert resolved.provenance == original.provenance
    assert len(resolved.tracks) == len(original.tracks)


# ---------------------------------------------------------------------------
# 5. A property, rather than a value.
#
# `content_run_id` promises determinism: same content, same id; different content,
# different id. Asserting the literal hash would pin an implementation detail and break on
# any harmless change. Assert the property instead.
# ---------------------------------------------------------------------------


def test_identical_runs_get_the_same_id():
    assert content_run_id(_run({0: [("x", 10.0)]})) == content_run_id(_run({0: [("x", 10.0)]}))


def test_different_runs_get_different_ids():
    assert content_run_id(_run({0: [("x", 10.0)]})) != content_run_id(_run({0: [("x", 11.0)]}))
