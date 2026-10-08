"""Readers for the CSVs this module evaluates. Everything comes out as an internal table with
0-based integer `frame`, integer `id`, and a float top-left box `x, y, w, h`.

- GT tracks (`<video>_reference_tracks_corrected.csv`): `frame_number, obj_id, x1, y1, x2, y2,
  bbox_size`, corner format.
- Tracker CSV, labelled (new runner): `frame_number, track_id, x, y, w, h, row_type,
  track_confirmed`. Keeps `row_type` and `track_confirmed` so variants.py can derive variants.
- Tracker CSV, legacy (old runner, `--legacy-output`): `frame_number, track_id, x, y, w, h`.
  Its rows include frozen boxes that cannot be told apart from real matches, so it is only ever
  evaluated as the `legacy` variant, as it is.
- External reference (cloud tracker): `frame_number, obj_id, x1, y1, x2, y2, ...`, corner
  format, evaluated as it is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .variants import BOX_COLUMNS

LABELLED = "labelled"
LEGACY = "legacy"
CORNERS = "corners"

_LEGACY_COLUMNS = {"frame_number", "track_id", "x", "y", "w", "h"}
_LABELLED_COLUMNS = _LEGACY_COLUMNS | {"row_type", "track_confirmed"}
_CORNER_COLUMNS = {"frame_number", "obj_id", "x1", "y1", "x2", "y2"}


def _read(path: Path) -> pd.DataFrame:
    # Tolerates Windows line endings (the GT exports have them).
    table = pd.read_csv(path)
    table.columns = [c.strip() for c in table.columns]
    return table


def _finish(table: pd.DataFrame, path: Path, size_checked: pd.Series | None = None) -> pd.DataFrame:
    """Cast and validate. `size_checked` marks the rows whose boxes must have positive size
    (default: all rows)."""
    table["frame"] = table["frame"].astype("int64")
    table["id"] = table["id"].astype("int64")
    for column in ("x", "y", "w", "h"):
        table[column] = table[column].astype("float64")
    if (table["frame"] < 0).any():
        raise ValueError(f"{path}: negative frame numbers")
    degenerate = (table["w"] <= 0) | (table["h"] <= 0)
    if size_checked is not None:
        degenerate &= size_checked
    if degenerate.any():
        raise ValueError(f"{path}: boxes with non-positive width or height")
    if table.duplicated(["frame", "id"]).any():
        raise ValueError(f"{path}: an id appears more than once in the same frame")
    return table.sort_values(["frame", "id"], kind="stable").reset_index(drop=True)


def _from_corners(table: pd.DataFrame, id_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": table["frame_number"],
            "id": table[id_column],
            "x": table["x1"].astype("float64"),
            "y": table["y1"].astype("float64"),
            "w": table["x2"].astype("float64") - table["x1"].astype("float64"),
            "h": table["y2"].astype("float64") - table["y1"].astype("float64"),
        }
    )


def read_gt_tracks(path: Path) -> pd.DataFrame:
    table = _read(path)
    missing = _CORNER_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{path}: GT file is missing columns {sorted(missing)}")
    return _finish(_from_corners(table, "obj_id"), path)


def tracker_format(path: Path) -> str:
    """`labelled`, `legacy` or `corners`, from the header alone."""
    columns = {c.strip() for c in pd.read_csv(path, nrows=0).columns}
    if _LABELLED_COLUMNS <= columns:
        return LABELLED
    if _LEGACY_COLUMNS <= columns:
        return LEGACY
    if _CORNER_COLUMNS <= columns:
        return CORNERS
    raise ValueError(f"{path}: unrecognised tracker CSV header {sorted(columns)}")


def _parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError(f"unparseable track_confirmed values {sorted(set(series[parsed.isna()]))}")
    return parsed.astype(bool)


def read_labelled_tracks(path: Path) -> pd.DataFrame:
    """New-format tracker CSV; keeps `row_type` and `track_confirmed`."""
    table = _read(path)
    if not _LABELLED_COLUMNS <= set(table.columns):
        raise ValueError(f"{path}: not a labelled tracker CSV")
    out = table.rename(columns={"frame_number": "frame", "track_id": "id"})
    out["track_confirmed"] = _parse_bool(out["track_confirmed"])
    out["row_type"] = out["row_type"].astype(str).str.strip()
    # A `predicted` box is the Kalman state, which can reach zero or negative size while a track
    # coasts (its height velocity keeps shrinking it). Matched boxes are detections and must not.
    is_detection = out["row_type"] != "predicted"
    return _finish(out[[*BOX_COLUMNS, "row_type", "track_confirmed"]].copy(), path, size_checked=is_detection)


def read_legacy_tracks(path: Path) -> pd.DataFrame:
    table = _read(path)
    if not _LEGACY_COLUMNS <= set(table.columns):
        raise ValueError(f"{path}: not a legacy tracker CSV")
    out = table.rename(columns={"frame_number": "frame", "track_id": "id"})
    return _finish(out[BOX_COLUMNS].copy(), path)


def read_corner_tracks(path: Path) -> pd.DataFrame:
    """External reference tracks in corner format (cloud tracker output)."""
    table = _read(path)
    missing = _CORNER_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return _finish(_from_corners(table, "obj_id"), path)
