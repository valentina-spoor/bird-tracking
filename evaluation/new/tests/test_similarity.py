import numpy as np
import pandas as pd
import pytest

from birdeval.similarity import dotd, gt_box_scales, iou, per_track_scales


def test_iou_small_box_shifted_two_pixels():
    # TASK finding 6: a 9x7 box shifted 2 px in x and y. Overlap 7 x 5 = 35, union 63 + 63 - 35 = 91.
    assert iou([[0, 0, 9, 7]], [[2, 2, 9, 7]])[0, 0] == pytest.approx(35 / 91)


def test_iou_identity_disjoint_and_degenerate():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 5, 5]])
    m = iou(boxes, np.vstack([boxes, [[0, 0, -1, -1]]]))
    assert m.shape == (2, 3)
    np.testing.assert_allclose(m, [[1, 0, 0], [0, 1, 0]])  # a negative-size box has zero area


def test_dotd_values_and_row_scales():
    gt = [[0, 0, 10, 10], [0, 0, 10, 10]]  # centre (5, 5)
    pred = [[3, 4, 10, 10]]  # centre (8, 9): distance 5
    m = dotd(gt, pred, gt_scales=[5.0, 10.0])
    np.testing.assert_allclose(m[:, 0], [np.exp(-1.0), np.exp(-0.5)])
    # centre-only: a different box size with the same centre scores 1
    assert dotd([[0, 0, 10, 10]], [[2, 2, 6, 6]], [9.0])[0, 0] == pytest.approx(1.0)


def test_per_track_scale_uses_gt_only_and_respects_s_min():
    gt = pd.DataFrame({
        "frame": [0, 1, 2, 0, 1],
        "id": [1, 1, 1, 2, 2],
        "x": 0.0, "y": 0.0,
        "w": [4.0, 16.0, 9.0, 2.0, 3.0],
        "h": [4.0, 16.0, 9.0, 2.0, 3.0],
    })
    scales = per_track_scales(gt, s_min=4.0)
    assert scales == {1: 9.0, 2: 4.0}  # track 1: median of sqrt areas 4, 16, 9; track 2: median 2.5 -> floor 4
    per_row = gt_box_scales(gt, "per_track", s_fixed=9.0, s_min=4.0)
    np.testing.assert_allclose(per_row, [9, 9, 9, 4, 4])
    # the scale depends only on GT: the function takes no tracker input, and changing GT changes it
    gt2 = gt.assign(w=gt["w"] * 2, h=gt["h"] * 2)
    assert per_track_scales(gt2, s_min=4.0) == {1: 18.0, 2: 5.0}


def test_per_box_and_fixed_scales():
    gt = pd.DataFrame({"frame": [0, 0], "id": [1, 2], "x": 0.0, "y": 0.0, "w": [1.0, 36.0], "h": [1.0, 1.0]})
    np.testing.assert_allclose(gt_box_scales(gt, "per_box", 9.0, 4.0), [4.0, 6.0])
    np.testing.assert_allclose(gt_box_scales(gt, "fixed", 9.0, 4.0), [9.0, 9.0])
