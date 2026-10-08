"""Derive evaluation variants from one labelled (new-format) tracker CSV.

The runner records every row it can account for, each with a `row_type` (`matched`,
`pre_confirmation`, `predicted`) and a `track_confirmed` flag. A variant is a choice of which of
those rows a tracker "reports". Each variant is a plain table with columns
`frame, id, x, y, w, h` (0-based frames, top-left corner boxes), sorted by (frame, id).

| variant             | rows                                                                  |
|---------------------|-----------------------------------------------------------------------|
| matched_online      | `matched` rows of confirmed tracks                                    |
| matched_final       | `matched` + `pre_confirmation` rows of confirmed tracks (default)    |
| matched_final_all   | matched_final + all rows of never-confirmed tracks                    |
| predicted           | matched_online + `predicted` rows                                     |
| frozen              | matched_online + each `predicted` row's box replaced by the track's   |
|                     | last matched box (reproduces the runner's `--legacy-output`)          |

`legacy` is not derived here: an old-format CSV is evaluated as it is (see formats.py).
"""

from __future__ import annotations

import pandas as pd

BOX_COLUMNS = ["frame", "id", "x", "y", "w", "h"]

DEFAULT_VARIANT = "matched_final"
DERIVED_VARIANTS = ("matched_online", "matched_final", "matched_final_all", "predicted", "frozen")
LEGACY_VARIANT = "legacy"
ROW_TYPES = ("matched", "pre_confirmation", "predicted")


def _boxes(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[BOX_COLUMNS].sort_values(["frame", "id"], kind="stable").reset_index(drop=True)


def _check(labelled: pd.DataFrame) -> None:
    missing = {*BOX_COLUMNS, "row_type", "track_confirmed"} - set(labelled.columns)
    if missing:
        raise ValueError(f"labelled tracker table is missing columns {sorted(missing)}")
    unknown = set(labelled["row_type"]) - set(ROW_TYPES)
    if unknown:
        raise ValueError(f"unknown row_type values {sorted(unknown)}")
    if labelled.duplicated(["frame", "id"]).any():
        raise ValueError("a track has more than one row in the same frame")
    bad_predicted = labelled[(labelled["row_type"] == "predicted") & ~labelled["track_confirmed"]]
    if len(bad_predicted):
        raise ValueError("`predicted` rows must belong to confirmed tracks")


def matched_online(labelled: pd.DataFrame) -> pd.DataFrame:
    return _boxes(labelled[(labelled["row_type"] == "matched") & labelled["track_confirmed"]])


def matched_final(labelled: pd.DataFrame) -> pd.DataFrame:
    is_match = labelled["row_type"].isin(["matched", "pre_confirmation"])
    return _boxes(labelled[is_match & labelled["track_confirmed"]])


def matched_final_all(labelled: pd.DataFrame) -> pd.DataFrame:
    return _boxes(labelled[labelled["row_type"].isin(["matched", "pre_confirmation"])])


def predicted(labelled: pd.DataFrame) -> pd.DataFrame:
    is_predicted = labelled["row_type"] == "predicted"
    online = (labelled["row_type"] == "matched") & labelled["track_confirmed"]
    return _boxes(labelled[online | is_predicted])


def frozen(labelled: pd.DataFrame) -> pd.DataFrame:
    """matched_online plus, in every `predicted` frame, the track's last matched box repeated."""
    online = matched_online(labelled)
    coasting = _boxes(labelled[labelled["row_type"] == "predicted"])[["frame", "id"]]
    if coasting.empty:
        return online
    # For each coasting (frame, id), take the same track's most recent matched row before it.
    last_match = pd.merge_asof(
        coasting.sort_values("frame"),
        online.rename(columns={"frame": "matched_frame"}).sort_values("matched_frame"),
        left_on="frame",
        right_on="matched_frame",
        by="id",
        allow_exact_matches=False,
        direction="backward",
    )
    if last_match["matched_frame"].isna().any():
        raise ValueError("a `predicted` row has no earlier `matched` row in the same track")
    return _boxes(pd.concat([online, last_match[BOX_COLUMNS]], ignore_index=True))


_DERIVERS = {
    "matched_online": matched_online,
    "matched_final": matched_final,
    "matched_final_all": matched_final_all,
    "predicted": predicted,
    "frozen": frozen,
}


def derive(labelled: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant not in _DERIVERS:
        raise ValueError(f"unknown variant {variant!r}; derived variants are {DERIVED_VARIANTS}")
    _check(labelled)
    return _DERIVERS[variant](labelled)


def derive_all(labelled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _check(labelled)
    return {name: fn(labelled) for name, fn in _DERIVERS.items()}
