"""Evaluate every discovered tracker result against the hand-labelled GT.

    python evaluation/new/scripts/evaluate.py --config evaluation/new/configs/local.yaml \
        --out evaluation/new/results/<timestamp>/ \
        [--gt GT_DIR] [--videos VIDEOS_DIR] [--results PATH=LAYOUT[:INPUT] ...] [--parallel N]

Paths in the config are resolved against the repository root. `--gt`, `--videos` and
`--results` override the config; `--results` replaces the config's list, e.g.
`--results experiments/results_v2=v2:gt,yolo /path/to/experiments/results=legacy:yolo`.
Writes metrics_per_video.csv, metrics_pooled.csv, config_used.yaml, run_info.json and
evaluate.log into --out. See evaluation/new/README.md for how to read them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "evaluation/new"))

from birdeval.batch import run_batch  # noqa: E402


def parse_results(specs: list[str]) -> list[dict]:
    out = []
    for spec in specs:
        path, _, rest = spec.rpartition("=")
        layout, _, input_name = rest.partition(":")
        if not path or layout not in ("v2", "legacy"):
            raise SystemExit(f"--results expects PATH=v2:INPUTS or PATH=legacy:INPUT, got {spec!r}")
        entry = {"path": path, "layout": layout}
        if input_name:
            # v2: comma-separated input folders to read; legacy: the input of every folder
            entry["inputs" if layout == "v2" else "input"] = input_name.split(",") if layout == "v2" else input_name
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: evaluation/new/results/<UTC timestamp>/")
    parser.add_argument("--gt", default=None, help="GT directory (overrides paths.gt_dir)")
    parser.add_argument("--videos", default=None, help="videos directory (overrides paths.videos_dir)")
    parser.add_argument("--results", nargs="+", default=None, metavar="PATH=LAYOUT[:INPUT]")
    parser.add_argument("--parallel", type=int, default=None, help="use TrackEval's parallel mode with N cores")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.gt:
        config["paths"]["gt_dir"] = args.gt
    if args.videos:
        config["paths"]["videos_dir"] = args.videos
    if args.results:
        config["paths"]["results"] = parse_results(args.results)
    if args.parallel:
        config.setdefault("trackeval", {}).update({"use_parallel": True, "num_cores": args.parallel})

    out = args.out or REPO / "evaluation/new/results" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out if out.is_absolute() else Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "evaluate.log", mode="w")])
    info = run_batch(config, out, REPO, sys.argv)
    logging.info("wrote %s (%d videos, %d runs, %d skipped)", out, info["n_videos"], info["n_runs"], len(info["skipped"]))


if __name__ == "__main__":
    main()
