import pandas as pd
import pytest

from birdeval.formats import read_labelled_tracks, read_legacy_tracks
from birdeval.variants import BOX_COLUMNS, derive, derive_all


def _labelled(rows):
    table = pd.DataFrame(rows, columns=[*BOX_COLUMNS, "row_type", "track_confirmed"])
    return table.astype({"frame": "int64", "id": "int64", "x": float, "y": float, "w": float, "h": float})


# Track 1: matched from frame 0, confirmed at frame 2, coasts (predicted) in frames 5 and 6.
# Track 2: never confirmed (two pre_confirmation rows).
LABELLED = _labelled([
    (0, 1, 10, 10, 4, 4, "pre_confirmation", True),
    (1, 1, 11, 10, 4, 4, "pre_confirmation", True),
    (2, 1, 12, 10, 4, 4, "matched", True),
    (3, 1, 13, 10, 4, 4, "matched", True),
    (4, 1, 14, 10, 4, 4, "matched", True),
    (5, 1, 15.2, 10.1, 4.1, 3.9, "predicted", True),
    (6, 1, 16.3, 10.2, 4.2, 3.8, "predicted", True),
    (3, 2, 90, 90, 4, 4, "pre_confirmation", False),
    (4, 2, 91, 90, 4, 4, "pre_confirmation", False),
])


def test_variant_row_counts():
    counts = {name: len(table) for name, table in derive_all(LABELLED).items()}
    assert counts == {"matched_online": 3, "matched_final": 5, "matched_final_all": 7, "predicted": 5, "frozen": 5}


def test_predicted_uses_the_kalman_box_and_frozen_repeats_the_last_match():
    pred = derive(LABELLED, "predicted").set_index("frame")
    frozen = derive(LABELLED, "frozen").set_index("frame")
    assert pred.loc[6, "x"] == 16.3
    assert (frozen.loc[5, ["x", "y", "w", "h"]].tolist(), frozen.loc[6, ["x", "y", "w", "h"]].tolist()) == (
        [14, 10, 4, 4], [14, 10, 4, 4])


def test_never_confirmed_tracks_only_in_matched_final_all():
    for name, table in derive_all(LABELLED).items():
        assert (2 in set(table["id"])) == (name == "matched_final_all"), name


def test_frozen_rebuilt_from_new_runner_equals_legacy_output_on_1dac(clip_1dac):
    """The new runner's output, rebuilt as `frozen`, reproduces the old runner's CSV exactly
    (1DAC, GT detections, baseline parameters; legacy = experiments/results/test_original_GT1)."""
    labelled = read_labelled_tracks(clip_1dac["labelled_gt_input"])
    legacy = read_legacy_tracks(clip_1dac["legacy_gt_input"])
    frozen = derive(labelled, "frozen")
    assert len(frozen) == len(legacy) == 10906
    pd.testing.assert_frame_equal(frozen.reset_index(drop=True), legacy.reset_index(drop=True))
    # and matched_online is the legacy output without its frozen rows
    online = derive(labelled, "matched_online")
    merged = legacy.merge(online, on=BOX_COLUMNS, how="left", indicator=True)
    assert (merged["_merge"] == "both").sum() == len(online) == 10906 - 815


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        derive(LABELLED, "legacy")
