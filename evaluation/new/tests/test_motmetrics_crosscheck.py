"""Cross-check TrackEval (through our dataset class) against py-motmetrics, an independent
implementation, on IDF1, MOTA, IDSW, FP and FN.

Two pair definitions, each computed by motmetrics itself (see motmetrics_reference.py):
- `iou`: IoU >= 0.5, against our `iou` similarity;
- `centre`: centre distance <= s * ln 2, against our `dotd` with the `fixed` scale s = 9, since
  exp(-d / s) >= 0.5 exactly when d <= s * ln 2. This validates the DotD path.

Findings (1DAC, recorded below; the toy cases agree exactly):
- IDF1, FP and FN agree exactly. Detection counts and the global identity assignment do not
  depend on how frames are matched over time.
- IDSW, and so MOTA, can differ by a few switches on crowded scenes. This is a definitional
  difference in how each library continues identities, not a bug in either:
  * TrackEval (CLEAR) runs one Hungarian per frame maximising similarity, with a large bonus for
    pairs that were matched in the immediately previous frame.
  * motmetrics first keeps every GT object's last pairing from any earlier frame if it is still
    within the threshold, then runs Hungarian on the rest, minimising distance (1 - IoU, or
    squared centre distance), so ties and near-ties between close birds resolve differently.
  We assert the observed difference is at most 2 switches and do not force agreement.
"""

import pytest

from conftest import FIXTURES, TOY_FRAME_COUNT
from motmetrics_reference import accumulate, summary
from birdeval.formats import read_gt_tracks, read_labelled_tracks, read_legacy_tracks
from birdeval.run import evaluate
from birdeval.trackeval_dataset import SequenceSpec, SimilarityConfig, TrackerSource
from birdeval.variants import derive

SIMILARITY = {"iou": SimilarityConfig("iou"), "centre": SimilarityConfig("dotd", "fixed", 9.0, 4.0)}
TOY_CASES = ["perfect", "fragmentation", "swap", "false_positive", "false_negative", "empty_ends_with_fp"]


def trackeval_summary(gt_path, source, frame_count, measure):
    rows = evaluate({"v": SequenceSpec(gt_path, frame_count)}, {"t": {"v": source}}, [SIMILARITY[measure]])
    r = next(row for row in rows if row["aggregation"] == "video")
    return {"IDF1": r["IDF1"], "MOTA": r["MOTA"], "IDSW": int(r["IDSW"]), "FP": int(r["CLR_FP"]), "FN": int(r["CLR_FN"])}


@pytest.mark.parametrize("measure", sorted(SIMILARITY))
@pytest.mark.parametrize("case", TOY_CASES)
def test_toy_cases_agree_exactly(case, measure):
    gt_path, tracks_path = FIXTURES / f"{case}_gt.csv", FIXTURES / f"{case}_tracks.csv"
    ours = trackeval_summary(gt_path, TrackerSource(tracks_path, "legacy"), TOY_FRAME_COUNT, measure)
    ref = summary(accumulate(read_gt_tracks(gt_path), read_legacy_tracks(tracks_path), TOY_FRAME_COUNT, measure))
    assert ours == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("measure", sorted(SIMILARITY))
@pytest.mark.parametrize("input_name", ["gt", "yolo"])
@pytest.mark.parametrize("variant", ["matched_final", "frozen"])
def test_1dac(clip_1dac, input_name, variant, measure):
    labelled_path = clip_1dac[f"labelled_{input_name}_input"]
    ours = trackeval_summary(clip_1dac["gt"], TrackerSource(labelled_path, "labelled", variant), 1009, measure)
    pred = derive(read_labelled_tracks(labelled_path), variant)
    ref = summary(accumulate(read_gt_tracks(clip_1dac["gt"]), pred, 1009, measure))
    for key in ("IDF1", "FP", "FN"):
        assert ours[key] == pytest.approx(ref[key], abs=1e-12), key
    assert abs(ours["IDSW"] - ref["IDSW"]) <= 2
    num_gt = 10142
    assert ours["MOTA"] - ref["MOTA"] == pytest.approx((ref["IDSW"] - ours["IDSW"]) / num_gt, abs=1e-12)
