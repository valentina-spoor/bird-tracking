"""Print a results table (metrics_pooled.csv or metrics_per_video.csv) in a readable form.

    python evaluation/new/scripts/show_metrics.py evaluation/new/results/<run>/metrics_pooled.csv
    python evaluation/new/scripts/show_metrics.py <run>/metrics_per_video.csv --experiment baseline motion --variant all
    python evaluation/new/scripts/show_metrics.py <run>/metrics_pooled.csv --columns all --markdown > table.md

Display only: it filters, sorts and formats; it does not interpret anything. One table is printed
per input (`gt`, `yolo`, `cloud` are never mixed) and, for the per-video file, per video.
By default it shows the default variant and the primary similarity; see --help for the filters.
Column meanings: evaluation/new/METRICS.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMN_SETS = {
    "summary": ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "diag_id_changes", "CLR_FP", "CLR_FN", "n_pred_ids", "n_gt_ids"],
    "hota": ["HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr"],
    "identity": ["IDF1", "IDP", "IDR", "IDTP", "IDFP", "IDFN", "IDSW", "diag_id_changes", "n_pred_ids", "n_gt_ids", "id_ratio"],
    "clear": ["MOTA", "MOTP", "MODA", "IDSW", "Frag", "MT", "PT", "ML", "CLR_TP", "CLR_FP", "CLR_FN", "CLR_Re", "CLR_Pr"],
}
LABEL_COLUMNS = ["experiment", "aliases", "variant"]


def _filter(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.variant == "default":
        df = df[df["is_default_variant"] | (df["variant"].isin(["legacy", "as-is"]) & args.with_legacy)]
    elif args.variant != "all":
        df = df[df["variant"].isin(args.variant.split(","))]
    if args.similarity != "all":
        df = df[df["similarity"] == args.similarity]
        if args.similarity == "dotd" and args.s_scheme != "all":
            df = df[df["s_scheme"] == args.s_scheme]
    if args.input:
        df = df[df["input"].isin(args.input)]
    if args.experiment:
        df = df[df["experiment"].isin(args.experiment)]
    if "aggregation" in df.columns and args.aggregation != "all":
        df = df[df["aggregation"] == args.aggregation]
    if "video" in df.columns and args.video:
        df = df[df["video"].str.contains("|".join(args.video))]
    return df


def _markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--input", nargs="*", help="inputs to show (default: all, one table each)")
    parser.add_argument("--experiment", nargs="*", help="experiments to show (default: all)")
    parser.add_argument("--variant", default="default", help="'default', 'all', or comma-separated names")
    parser.add_argument("--with-legacy", action="store_true", help="with --variant default, also show legacy and cloud rows")
    parser.add_argument("--similarity", default="dotd", help="'dotd', 'iou' or 'all' (default dotd)")
    parser.add_argument("--s-scheme", default="per_track", help="DotD scale: per_track, fixed, per_box or all")
    parser.add_argument("--aggregation", default="pooled", help="pooled file: 'pooled', 'per_video_mean' or 'all'")
    parser.add_argument("--video", nargs="*", help="per-video file: video names or parts of them (e.g. 1DAC)")
    parser.add_argument("--columns", default="summary", help=f"{', '.join(COLUMN_SETS)}, 'all', or comma-separated names")
    parser.add_argument("--sort", default="HOTA", help="column to sort by (descending); default HOTA")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--markdown", action="store_true", help="print Markdown tables")
    args = parser.parse_args()

    df = _filter(pd.read_csv(args.csv), args)
    if df.empty:
        raise SystemExit("no rows match these filters")

    if args.columns == "all":
        skip = {*LABEL_COLUMNS, "input", "is_default_variant", "is_primary_similarity", "video", "videos", "tracker_csv"}
        metrics = [c for c in df.columns if c not in skip]
    else:
        metrics = COLUMN_SETS.get(args.columns, args.columns.split(","))
    missing = [c for c in metrics if c not in df.columns]
    if missing:
        raise SystemExit(f"unknown columns: {missing}")
    setting_columns = [c for c in ("similarity", "s_scheme", "aggregation") if c in df.columns and df[c].nunique() > 1]
    shown = [*LABEL_COLUMNS, *setting_columns, *[c for c in metrics if c not in setting_columns]]

    group_by = ["input", "video"] if "video" in df.columns else ["input"]
    filters = f"variant={args.variant} similarity={args.similarity}"
    if args.similarity == "dotd":
        filters += f" s_scheme={args.s_scheme}"
    if "aggregation" in df.columns:
        filters += f" aggregation={args.aggregation}"
    print(f"{args.csv}  ({filters})\n")
    for keys, group in df.groupby(group_by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        title = "  ".join(f"{k} = {v}" for k, v in zip(group_by, keys))
        table = group[shown].copy()
        if args.sort in table.columns:
            table = table.sort_values(args.sort, ascending=False, kind="stable")
        table["aliases"] = table["aliases"].fillna("")
        table = table.round(args.digits)
        for column in table.select_dtypes("number").columns:  # show counts as whole numbers
            values = table[column].dropna()
            if len(values) and (values == values.round()).all() and column not in ("HOTA", "MOTA", "IDF1"):
                table[column] = table[column].astype("Int64")
        if args.markdown:
            print(f"### {title}\n\n{_markdown(table)}\n")
        else:
            print(f"== {title}  ({len(table)} rows)")
            print(table.to_string(index=False))
            print()


if __name__ == "__main__":
    main()
