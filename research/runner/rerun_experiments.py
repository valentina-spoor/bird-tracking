"""Re-run experiment configs with the labelled-output runner, on one or more detection inputs.

    python research/runner/rerun_experiments.py \
        --videos-dir <dir with <video>.mp4> \
        --input yolo=<dir with <video>_detections.csv>:_detections.csv \
        --input gt=<dir with <video>_detections_corrected.csv>:_detections_corrected.csv \
        --legacy-results <experiments/results dir> \
        --out experiments/results_v2 [--configs name ...] [--workers 6] [--force]

For every config in experiments/configs/ (or the ones named) and every input, the tracker runs
once per video that has both a video file and an input file, writing
`<out>/<input>/<experiment>/<CLIP>_tracks.csv` (CLIP = last `_` token of the video stem) and a
`run_info.json` in each run folder.

Aliases: configs whose full parameter set, after the runner's own defaults are applied
(run_tracker.resolve_parameters), is identical run only once, under the alphabetically first
name. `<out>/aliases.yaml` maps that name to the others.

Legacy check (yolo input only, the legacy runs' input): wherever `<legacy-results>/<name>/`
holds a CSV for the same clip, for the run's name or any alias, the rebuilt `frozen` variant
is compared exactly with it (evaluation/new/scripts/verify_frozen.py). An empty (header-only)
legacy CSV is recorded as a failed legacy run, together with how many rows the re-run wrote.

`--no-video` is passed to every run: it only skips writing the annotated video. The runner
decodes every frame either way, and appearance-based association reads it regardless.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "spoor_tracker/tracker-kit"
CONFIGS = REPO / "experiments/configs"
VERIFY = REPO / "evaluation/new/scripts/verify_frozen.py"
sys.path.insert(0, str(KIT))

from run_tracker import resolve_parameters  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def runner_version() -> dict:
    tracker_files = sorted((KIT / "tracker").glob("*.py"))
    tracker_hash = hashlib.sha256(b"".join(sha256(p).encode() for p in tracker_files)).hexdigest()
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_files": [line[3:] for line in git("status", "--porcelain").splitlines()],
        "run_tracker_sha256": sha256(KIT / "run_tracker.py"),
        "tracker_package_sha256": tracker_hash,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def effective_parameters(config: dict) -> dict:
    """Every tracker parameter as the runner resolves it, in a canonical JSON-able form."""
    params = resolve_parameters(config)
    params["association_params"] = dataclasses.asdict(params["association_params"])

    def canonical(value):
        if isinstance(value, dict):
            return {k: canonical(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [canonical(v) for v in value]
        if isinstance(value, bool) or isinstance(value, str):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    return canonical(params)


def clip_name(video_stem: str) -> str:
    return video_stem.rsplit("_", 1)[-1]


def parse_inputs(specs: list[str]) -> dict[str, tuple[Path, str]]:
    inputs = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        directory, suffix = rest.rsplit(":", 1)
        inputs[name] = (Path(directory).resolve(), suffix)
    return inputs


def legacy_csv(legacy_root: Path | None, experiment: str, clip: str) -> Path | None:
    if legacy_root is None:
        return None
    folder = legacy_root / experiment
    matches = sorted(p for p in folder.glob("*_tracks.csv") if p.name.split("_", 1)[0] == clip) if folder.is_dir() else []
    if len(matches) > 1:
        raise SystemExit(f"{folder}: more than one tracks CSV for clip {clip}: {matches}")
    return matches[0] if matches else None


def video_frame_size(video: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(video))
    try:
        return (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def output_summary(tracks: Path, width: int, height: int) -> dict:
    table = pd.read_csv(tracks)
    counts = table.groupby(["row_type", "track_confirmed"]).size()
    predicted = table[table["row_type"] == "predicted"]
    cx = predicted["x"] + predicted["w"] / 2
    cy = predicted["y"] + predicted["h"] / 2
    return {
        "rows": len(table),
        "track_ids": int(table["track_id"].nunique()),
        "rows_by_type": {f"{t} (confirmed={c})": int(n) for (t, c), n in counts.items()},
        "predicted_nonpositive_size": int(((predicted["w"] <= 0) | (predicted["h"] <= 0)).sum()),
        "predicted_centre_outside_frame": int(((cx < 0) | (cx >= width) | (cy < 0) | (cy >= height)).sum()),
    }


def run_one(job: dict, force: bool) -> dict:
    run_dir: Path = job["run_dir"]
    tracks = run_dir / f"{job['clip']}_tracks.csv"
    command = [
        sys.executable, "run_tracker.py", "--video", str(job["video"]), "--detections", str(job["detections"]),
        "--tracks", str(tracks), "--no-video", "--config", str(job["config_path"]),
    ]
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    if force or not tracks.exists():
        proc = subprocess.run(command, cwd=KIT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"run failed: {' '.join(command)}\n{proc.stdout}\n{proc.stderr}")
        stdout = proc.stdout
    else:
        stdout = "(skipped: output exists; pass --force to re-run)"
    width, height, frame_count = video_frame_size(job["video"])
    return {
        "clip": job["clip"],
        "video": str(job["video"]),
        "video_frame_count": frame_count,
        "video_size": [width, height],
        "input_file": str(job["detections"]),
        "input_sha256": sha256(job["detections"]),
        "tracks": tracks.name,
        "command": command,
        "runner_stdout": stdout.splitlines(),
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **output_summary(tracks, width, height),
    }


def legacy_checks(names: list[str], clip: str, tracks: Path, legacy_root: Path | None, rows: int) -> list[dict]:
    checks = []
    for name in names:
        legacy = legacy_csv(legacy_root, name, clip)
        if legacy is None:
            continue
        if len(pd.read_csv(legacy)) == 0:
            checks.append({"legacy": str(legacy), "experiment": name, "status": "excluded: empty legacy CSV (failed run)",
                           "rerun_rows": rows})
            continue
        report_path = tracks.parent / f"legacy_check_{name}_{clip}.json"
        proc = subprocess.run([sys.executable, str(VERIFY), "--labelled", str(tracks), "--legacy", str(legacy),
                               "--json", str(report_path)], capture_output=True, text=True)
        report = json.loads(report_path.read_text())
        checks.append({"legacy": str(legacy), "experiment": name,
                       "status": "exact match" if proc.returncode == 0 else "MISMATCH",
                       "frozen_only_rows": report["frozen_only_rows"], "legacy_only_rows": report["legacy_only_rows"],
                       "report": report_path.name})
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=DIR:SUFFIX")
    parser.add_argument("--legacy-results", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="*", default=None, help="config names (default: all in experiments/configs)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    # The runner runs with the kit as its working directory, so every path it gets must be absolute.
    args.out = args.out.resolve()
    args.videos_dir = args.videos_dir.resolve()

    config_paths = sorted(CONFIGS.glob("*.json"))
    if args.configs:
        config_paths = [CONFIGS / f"{name}.json" for name in args.configs]

    groups: dict[str, list[Path]] = {}
    for path in config_paths:
        key = json.dumps(effective_parameters(json.loads(path.read_text())), sort_keys=True)
        groups.setdefault(key, []).append(path)
    runs = {paths[0].stem: (json.loads(key), paths) for key, paths in groups.items()}  # paths are sorted by name
    aliases = {name: [p.stem for p in paths[1:]] for name, (_, paths) in runs.items() if len(paths) > 1}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aliases.yaml").write_text(
        "# Configs with identical parameters after the runner's defaults are applied run once, under the\n"
        "# first name (alphabetical). Each entry: run name -> the other config names it stands for.\n"
        + yaml.safe_dump(aliases, sort_keys=True)
    )

    version = runner_version()
    videos = sorted(args.videos_dir.glob("*.mp4"))
    jobs = []
    for input_name, (input_dir, suffix) in parse_inputs(args.input).items():
        for name, (params, paths) in runs.items():
            for video in videos:
                detections = input_dir / f"{video.stem}{suffix}"
                if not detections.exists():
                    continue
                run_dir = args.out / input_name / name
                run_dir.mkdir(parents=True, exist_ok=True)
                jobs.append({"input": input_name, "name": name, "params": params, "config_path": paths[0],
                             "aliases": [p.stem for p in paths[1:]], "clip": clip_name(video.stem),
                             "video": video, "detections": detections, "run_dir": run_dir})
    print(f"{len(runs)} unique parameter sets from {len(config_paths)} configs, {len(jobs)} runs")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: (job, run_one(job, args.force)), jobs))

    by_folder: dict[Path, list] = {}
    for job, result in results:
        if job["input"] == "yolo":
            names = [job["name"], *job["aliases"]]
            tracks = job["run_dir"] / result["tracks"]
            result["legacy_checks"] = legacy_checks(names, job["clip"], tracks, args.legacy_results, result["rows"])
        by_folder.setdefault(job["run_dir"], []).append((job, result))

    for run_dir, items in by_folder.items():
        job = items[0][0]
        config_path: Path = job["config_path"]
        run_info = {
            "experiment": job["name"],
            "aliases": job["aliases"],
            "input": job["input"],
            "config_file": str(config_path.relative_to(REPO)),
            "config_sha256": sha256(config_path),
            "config": json.loads(config_path.read_text()),
            "effective_parameters": job["params"],
            "runner": version,
            "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "clips": {result["clip"]: result for _, result in sorted(items, key=lambda item: item[1]["clip"])},
        }
        (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")

    for job, result in sorted(results, key=lambda r: (r[0]["input"], r[0]["name"], r[1]["clip"])):
        checks = "; ".join(f"{c['experiment']}: {c['status']}" for c in result.get("legacy_checks", []))
        print(f"{job['input']:5s} {job['name']:28s} {result['clip']} rows={result['rows']:6d} ids={result['track_ids']:4d} "
              f"pred_nonpos={result['predicted_nonpositive_size']:3d} pred_outside={result['predicted_centre_outside_frame']:3d}"
              f"{'  legacy: ' + checks if checks else ''}")


if __name__ == "__main__":
    main()
