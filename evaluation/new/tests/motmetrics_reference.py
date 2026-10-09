"""py-motmetrics reference computation for the cross-check (an independent implementation).

motmetrics computes its own distance matrices here: `iou_matrix(max_iou=0.5)` (distance
1 - IoU, a pair is allowed if IoU >= 0.5) and squared centre distance with
`max_d2 = (s * ln 2)^2`, which allows exactly the pairs with DotD = exp(-d / s) >= 0.5 for a
fixed scale s.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# py-motmetrics 1.4.0 (latest release) still calls np.asfarray, removed in numpy 2.0;
# np.asfarray(x) was exactly np.asarray(x, dtype=float).
if "asfarray" not in np.__dict__:
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402

from birdeval.similarity import centres


def accumulate(gt: pd.DataFrame, pred: pd.DataFrame, frame_count: int, measure: str, s_fixed: float = 9.0):
    acc = mm.MOTAccumulator(auto_id=False)
    gt_by_frame = dict(tuple(gt.groupby("frame")))
    pred_by_frame = dict(tuple(pred.groupby("frame")))
    empty = pd.DataFrame(columns=["id", "x", "y", "w", "h"])
    for frame in range(frame_count):
        g = gt_by_frame.get(frame, empty)
        p = pred_by_frame.get(frame, empty)
        gb = g[["x", "y", "w", "h"]].to_numpy(dtype=float).reshape(-1, 4)
        pb = p[["x", "y", "w", "h"]].to_numpy(dtype=float).reshape(-1, 4)
        if measure == "iou":
            dist = mm.distances.iou_matrix(gb, np.clip(pb, [-np.inf, -np.inf, 0, 0], None), max_iou=0.5)
        elif measure == "centre":
            dist = mm.distances.norm2squared_matrix(centres(gb), centres(pb), max_d2=(s_fixed * math.log(2)) ** 2)
        else:
            raise ValueError(measure)
        acc.update(g["id"].tolist(), p["id"].tolist(), dist, frameid=frame)
    return acc


def summary(acc) -> dict:
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "mota", "num_switches", "num_false_positives", "num_misses"], name="x")
    return {"IDF1": float(s["idf1"].iloc[0]), "MOTA": float(s["mota"].iloc[0]), "IDSW": int(s["num_switches"].iloc[0]),
            "FP": int(s["num_false_positives"].iloc[0]), "FN": int(s["num_misses"].iloc[0])}
