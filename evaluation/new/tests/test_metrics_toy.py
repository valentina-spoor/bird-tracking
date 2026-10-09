"""Hand-made toy cases with known answers (fixtures in tests/fixtures/).

Every video has 10 frames (0..9). Boxes are 10x10 px; a bird moves 5 px per frame in x, and
different birds are 300 px apart, so a predicted box either equals a GT box (IoU = DotD = 1) or
is far from every GT box (similarity ~0). Hence every match is exact and HOTA's DetA/AssA reduce
to simple counts, identical for all alpha thresholds and for every similarity.

Formulas used below (TP, FN, FP: detection counts; IDTP, IDFN, IDFP: after the global identity
assignment):
  MOTA = 1 - (FN + FP + IDSW) / num_gt_boxes
  IDF1 = IDTP / (IDTP + 0.5 IDFP + 0.5 IDFN)
  DetA = TP / (TP + FN + FP)
  AssA = mean over TP pairs (gt_id, pred_id) of TPA / (n_gt_id + n_pred_id - TPA), where TPA is
         the number of frames that pair is matched, n_* the number of boxes of that id
  HOTA = sqrt(DetA * AssA)
"""

from math import sqrt

import pytest

from conftest import TOY_FRAME_COUNT, toy_rows

CASES = {
    # GT bird 1 in frames 2..7, predicted exactly by one track; frames 0, 1, 8, 9 are empty.
    "perfect": dict(HOTA=1.0, DetA=1.0, AssA=1.0, LocA=1.0, MOTA=1.0, MOTP=1.0, IDF1=1.0, IDSW=0, CLR_FP=0, CLR_FN=0,
                    CLR_TP=6, MT=1, ML=0, n_gt_boxes=6, n_pred_ids=1, diag_id_changes=0),
    # GT bird 1 in frames 0..9; track 7 covers frames 0..4, track 8 frames 5..9.
    # IDSW = 1 (frame 5). MOTA = 1 - 1/10 = 0.9.
    # Global assignment pairs bird 1 with one track: IDTP 5, IDFN 5, IDFP 5 -> IDF1 = 5 / 10 = 0.5.
    # Pairs (1,7) and (1,8) each have TPA 5, n_gt 10, n_pred 5 -> 5 / 10 = 0.5 -> AssA 0.5.
    # DetA 1 -> HOTA sqrt(0.5). The bird is covered in every frame, so Frag = 0.
    # Each track only ever matches bird 1, so no track changes identity: diag_id_changes = 0.
    "fragmentation": dict(HOTA=sqrt(0.5), DetA=1.0, AssA=0.5, MOTA=0.9, IDF1=0.5, IDSW=1, Frag=0, CLR_FP=0, CLR_FN=0,
                          n_pred_ids=2, id_ratio=2.0, diag_id_changes=0),
    # Birds 1 and 2 in frames 0..9 (300 px apart). Track 7 follows bird 1 in frames 0..4 then bird 2;
    # track 8 the opposite. IDSW = 2 (both birds at frame 5). MOTA = 1 - 2/20 = 0.9.
    # Best global assignment: 1<->7 and 2<->8, IDTP 10, IDFN 10, IDFP 10 -> IDF1 = 10 / 20 = 0.5.
    # Each of the 4 pairs: TPA 5, n_gt 10, n_pred 10 -> 5 / 15 = 1/3 -> AssA 1/3, HOTA sqrt(1/3).
    # Both tracks change bird once: diag_id_changes = 2.
    "swap": dict(HOTA=sqrt(1 / 3), DetA=1.0, AssA=1 / 3, MOTA=0.9, IDF1=0.5, IDSW=2, CLR_FP=0, CLR_FN=0,
                 diag_id_changes=2),
    # Bird 1 in frames 0..9 predicted exactly; track 9 has 5 boxes far from any bird (frames 0..4).
    # FP 5 -> MOTA = 1 - 5/10 = 0.5. IDTP 10, IDFP 5 -> IDF1 = 10 / 12.5 = 0.8.
    # DetA = 10 / 15 = 2/3, AssA 1 -> HOTA sqrt(2/3).
    "false_positive": dict(HOTA=sqrt(2 / 3), DetA=2 / 3, AssA=1.0, MOTA=0.5, IDF1=0.8, IDSW=0, CLR_FP=5, CLR_FN=0,
                           n_pred_ids=2, n_gt_ids=1),
    # Bird 1 in frames 0..9 predicted exactly; bird 2 (frames 0..4) never predicted.
    # FN 5 -> MOTA = 1 - 5/15 = 2/3. IDTP 10, IDFN 5 -> IDF1 = 10 / 12.5 = 0.8.
    # DetA = 10 / 15, AssA 1 -> HOTA sqrt(2/3). Bird 1 mostly tracked, bird 2 mostly lost.
    "false_negative": dict(HOTA=sqrt(2 / 3), DetA=2 / 3, AssA=1.0, MOTA=2 / 3, IDF1=0.8, IDSW=0, CLR_FP=0, CLR_FN=5,
                           MT=1, ML=1, n_gt_ids=2),
    # Like `perfect` (bird 1 in frames 2..7), plus one predicted box in frame 0 and one in frame 9,
    # where there is no GT at all. Those frames must still be evaluated: FP 2.
    # MOTA = 1 - 2/6 = 2/3. IDTP 6, IDFP 2 -> IDF1 = 6 / 7. DetA = 6 / 8, AssA 1 -> HOTA sqrt(0.75).
    "empty_ends_with_fp": dict(HOTA=sqrt(0.75), DetA=0.75, AssA=1.0, MOTA=2 / 3, IDF1=6 / 7, CLR_FP=2, CLR_FN=0,
                               CLR_TP=6, n_pred_boxes=8),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_toy_case(case):
    rows = toy_rows(case)
    assert len(rows) == 4  # dotd x 3 scale schemes + iou: exact boxes make them all agree
    for row in rows:
        assert row["n_frames"] == TOY_FRAME_COUNT
        for metric, expected in CASES[case].items():
            assert row[metric] == pytest.approx(expected, abs=1e-9), (case, row["similarity"], row["s_scheme"], metric)
