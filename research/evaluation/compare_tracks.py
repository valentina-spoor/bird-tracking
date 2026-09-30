"""Compare tracker outputs against a reference track CSV and with each other.

    python research/evaluation/compare_tracks.py \
        --reference spoor_tracker/tracker-kit/data/20251012_164031_1DAC_cloud_reference_tracks.csv \
        --tracks experiments/results/original/1DAC_tracks.csv experiments/results/fused/1DAC_tracks.csv \
        --names original fused

IMPORTANT: the reference is the stored *cloud tracker* output, not human-labelled ground truth. The
numbers measure agreement with that tracker, so treat them as a relative yardstick between methods
on the same clip, not as absolute accuracy. Boxes are matched per frame with IoU >= --iou.

Columns
  ids            distinct track ids emitted (fewer is better when the true bird count is small)
  med_len        median frames per track
  IDF1/IDP/IDR   standard identity metrics against the reference (global id assignment)
  IDSW           reference tracks whose matched track id changed (identity switches)
  frag           times a reference track was lost and later re-matched (fragmentations)
  ids/ref        emitted ids per reference identity (1.0 is ideal)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "frame": df["frame_number"],
            "id": df["obj_id"],
            "x1": df["x1"], "y1": df["y1"], "x2": df["x2"], "y2": df["y2"],
        }
    )


def load_tracks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "frame": df["frame_number"],
            "id": df["track_id"],
            "x1": df["x"], "y1": df["y"], "x2": df["x"] + df["w"], "y2": df["y"] + df["h"],
        }
    )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def evaluate(reference: pd.DataFrame, tracks: pd.DataFrame, iou_threshold: float) -> dict:
    ref_frames = {f: g for f, g in reference.groupby("frame")}
    trk_frames = {f: g for f, g in tracks.groupby("frame")}
    coords = ["x1", "y1", "x2", "y2"]

    pair_counts: dict[tuple[int, int], int] = {}
    last_pred: dict[int, int] = {}
    was_matched: dict[int, bool] = {}
    idsw = frag = matched_rows = 0

    for frame in sorted(set(ref_frames) | set(trk_frames)):
        r, t = ref_frames.get(frame), trk_frames.get(frame)
        matched_ref: dict[int, int] = {}
        if r is not None and t is not None:
            iou = iou_matrix(r[coords].to_numpy(float), t[coords].to_numpy(float))
            for i, j in zip(*linear_sum_assignment(-iou)):
                if iou[i, j] >= iou_threshold:
                    rid, tid = int(r["id"].iloc[i]), int(t["id"].iloc[j])
                    matched_ref[rid] = tid
                    pair_counts[(rid, tid)] = pair_counts.get((rid, tid), 0) + 1
                    matched_rows += 1
        if r is not None:
            for rid in r["id"].astype(int):
                if rid in matched_ref:
                    if rid in last_pred and last_pred[rid] != matched_ref[rid]:
                        idsw += 1
                    if was_matched.get(rid) is False:
                        frag += 1
                    last_pred[rid] = matched_ref[rid]
                    was_matched[rid] = True
                elif rid in was_matched:
                    was_matched[rid] = False

    ref_ids = sorted(reference["id"].astype(int).unique())
    trk_ids = sorted(tracks["id"].astype(int).unique())
    idtp = 0
    if pair_counts:
        ri = {v: k for k, v in enumerate(ref_ids)}
        ti = {v: k for k, v in enumerate(trk_ids)}
        gain = np.zeros((len(ref_ids), len(trk_ids)))
        for (rid, tid), c in pair_counts.items():
            gain[ri[rid], ti[tid]] = c
        rows, cols = linear_sum_assignment(-gain)
        idtp = int(gain[rows, cols].sum())
    n_ref, n_trk = len(reference), len(tracks)
    idfn, idfp = n_ref - idtp, n_trk - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
    lengths = tracks.groupby("id").size()
    return {
        "ids": len(trk_ids),
        "rows": n_trk,
        "med_len": float(lengths.median()) if len(lengths) else 0.0,
        "IDF1": idf1,
        "IDP": idtp / max(n_trk, 1),
        "IDR": idtp / max(n_ref, 1),
        "IDSW": idsw,
        "frag": frag,
        "ids/ref": len(trk_ids) / max(len(ref_ids), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--csv", type=Path, default=None, help="also write the table to this CSV")
    args = parser.parse_args()

    names = args.names or [p.parent.name for p in args.tracks]
    if len(names) != len(args.tracks):
        parser.error("--names must match --tracks in length")

    reference = load_reference(args.reference)
    rows = {name: evaluate(reference, load_tracks(path), args.iou) for name, path in zip(names, args.tracks)}
    table = pd.DataFrame(rows).T
    print(f"reference: {args.reference.name}  ({reference['id'].nunique()} identities, {len(reference)} boxes)")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    if args.csv:
        table.to_csv(args.csv)


if __name__ == "__main__":
    main()
