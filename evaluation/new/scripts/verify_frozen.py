"""Check that a labelled (new-format) tracker CSV reproduces a legacy tracker CSV exactly.

    python evaluation/new/scripts/verify_frozen.py --labelled NEW.csv --legacy OLD.csv [--json OUT.json]

Checks, all on exact (frame, id, x, y, w, h) values:
1. the `frozen` variant rebuilt from NEW equals OLD;
2. `matched_online` from NEW equals OLD with its frozen rows removed, where a legacy row counts
   as frozen when NEW has a `predicted` row for the same (frame, id). Independently of NEW, each
   such legacy row must also repeat the same track's box from the previous frame, which is what a
   frozen row is.
Prints the row count of every variant. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from birdeval.formats import read_labelled_tracks, read_legacy_tracks  # noqa: E402
from birdeval.variants import BOX_COLUMNS, derive_all  # noqa: E402


def _diff(a: pd.DataFrame, b: pd.DataFrame) -> tuple[int, int]:
    merged = a.merge(b, on=BOX_COLUMNS, how="outer", indicator=True)
    return int((merged["_merge"] == "left_only").sum()), int((merged["_merge"] == "right_only").sum())


def verify(labelled_path: Path, legacy_path: Path) -> dict:
    labelled = read_labelled_tracks(labelled_path)
    legacy = read_legacy_tracks(legacy_path)
    variants = derive_all(labelled)

    report = {
        "labelled": str(labelled_path),
        "legacy": str(legacy_path),
        "row_types": {
            f"{row_type} (confirmed={confirmed})": int(n)
            for (row_type, confirmed), n in labelled.groupby(["row_type", "track_confirmed"]).size().items()
        },
        "variant_rows": {name: len(table) for name, table in variants.items()},
        "variant_ids": {name: int(table["id"].nunique()) for name, table in variants.items()},
        "legacy_rows": len(legacy),
        "legacy_ids": int(legacy["id"].nunique()),
    }

    only_frozen, only_legacy = _diff(variants["frozen"], legacy)
    report["frozen_equals_legacy"] = only_frozen == 0 and only_legacy == 0
    report["frozen_only_rows"], report["legacy_only_rows"] = only_frozen, only_legacy

    predicted_keys = labelled.loc[labelled["row_type"] == "predicted", ["frame", "id"]]
    flagged = legacy.merge(predicted_keys, on=["frame", "id"], how="left", indicator=True)
    is_frozen = (flagged["_merge"] == "both").to_numpy()
    legacy_frozen = legacy[is_frozen]
    legacy_unfrozen = legacy[~is_frozen]
    report["legacy_frozen_rows"] = len(legacy_frozen)

    previous = legacy.copy()
    previous["frame"] += 1
    repeats = legacy_frozen.merge(previous, on=BOX_COLUMNS, how="left", indicator=True)
    report["legacy_frozen_rows_repeating_previous_box"] = int((repeats["_merge"] == "both").sum())

    only_online, only_unfrozen = _diff(variants["matched_online"], legacy_unfrozen)
    report["matched_online_equals_legacy_without_frozen"] = only_online == 0 and only_unfrozen == 0
    report["passed"] = (
        report["frozen_equals_legacy"]
        and report["matched_online_equals_legacy_without_frozen"]
        and report["legacy_frozen_rows_repeating_previous_box"] == len(legacy_frozen)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labelled", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = parser.parse_args()

    report = verify(args.labelled, args.legacy)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
