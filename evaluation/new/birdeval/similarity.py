"""Similarity between the GT boxes and the predicted boxes of one frame, as a |G| x |P| matrix
in [0, 1]. Boxes are (x, y, w, h) with (x, y) the top-left corner.

- `iou`: intersection over union. A box with non-positive width or height (a degenerate Kalman
  prediction) has zero area, so its IoU with anything is 0.
- `dotd`: exp(-d / s), with d the distance between box centres and s a scale in pixels, chosen
  per GT box by one of three schemes:
  - `fixed`: one constant (`s_fixed`).
  - `per_track`: for each GT track, max(s_min, median over its boxes of sqrt(w * h)). Computed
    from GT only, once per video (`per_track_scales`).
  - `per_box`: max(s_min, sqrt(w * h)) of that GT box.

DotD inside HOTA follows SMOT4SB's SO-HOTA. Only `fixed` (with one s for the whole dataset) is
comparable to published SO-HOTA numbers; `per_track` and `per_box` are our own adaptive scales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MEASURES = ("dotd", "iou")
S_SCHEMES = ("per_track", "fixed", "per_box")


def iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    pred = np.asarray(pred, dtype=float).reshape(-1, 4)
    gw, gh = np.clip(gt[:, 2], 0, None), np.clip(gt[:, 3], 0, None)
    pw, ph = np.clip(pred[:, 2], 0, None), np.clip(pred[:, 3], 0, None)
    ix = np.minimum((gt[:, 0] + gw)[:, None], (pred[:, 0] + pw)[None, :]) - np.maximum(gt[:, 0][:, None], pred[:, 0][None, :])
    iy = np.minimum((gt[:, 1] + gh)[:, None], (pred[:, 1] + ph)[None, :]) - np.maximum(gt[:, 1][:, None], pred[:, 1][None, :])
    intersection = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    union = (gw * gh)[:, None] + (pw * ph)[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, intersection / union, 0.0)
    return np.clip(out, 0.0, 1.0)


def centres(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return boxes[:, :2] + boxes[:, 2:] / 2


def dotd(gt: np.ndarray, pred: np.ndarray, gt_scales: np.ndarray) -> np.ndarray:
    """exp(-d / s); row i uses gt_scales[i]."""
    scales = np.asarray(gt_scales, dtype=float).reshape(-1)
    if np.any(scales <= 0):
        raise ValueError("DotD scales must be positive")
    distance = np.linalg.norm(centres(gt)[:, None, :] - centres(pred)[None, :, :], axis=2)
    return np.exp(-distance / scales[:, None])


def per_track_scales(gt: pd.DataFrame, s_min: float) -> dict[int, float]:
    """GT track id -> max(s_min, median over the track's boxes of sqrt(w * h)). GT only."""
    root_area = np.sqrt(gt["w"].to_numpy() * gt["h"].to_numpy())
    medians = pd.Series(root_area, index=gt["id"].to_numpy()).groupby(level=0).median()
    return {int(track): float(max(s_min, value)) for track, value in medians.items()}


def gt_box_scales(gt: pd.DataFrame, scheme: str, s_fixed: float, s_min: float) -> np.ndarray:
    """The DotD scale of every GT row, in the table's row order."""
    if scheme == "fixed":
        return np.full(len(gt), float(s_fixed))
    if scheme == "per_box":
        return np.maximum(s_min, np.sqrt(gt["w"].to_numpy() * gt["h"].to_numpy()))
    if scheme == "per_track":
        scales = per_track_scales(gt, s_min)
        return gt["id"].map(scales).to_numpy(dtype=float)
    raise ValueError(f"unknown s scheme {scheme!r}; expected one of {S_SCHEMES}")


def similarity_matrix(measure: str, gt: np.ndarray, pred: np.ndarray, gt_scales: np.ndarray | None = None) -> np.ndarray:
    if measure == "iou":
        return iou(gt, pred)
    if measure == "dotd":
        if gt_scales is None:
            raise ValueError("dotd needs gt_scales")
        return dotd(gt, pred, gt_scales)
    raise ValueError(f"unknown similarity {measure!r}; expected one of {MEASURES}")
