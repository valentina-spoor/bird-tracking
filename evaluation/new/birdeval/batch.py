"""Discover tracker results, evaluate them against GT, and write the result tables.

Discovery (the mapping agreed at CHECKPOINT 1):
- Videos: every `<gt_dir>/<video><gt_suffix>` whose `<videos_dir>/<video>.mp4` exists. A GT file
  without its video, and a tracker CSV for a video without GT, are logged as skipped.
- A tracker CSV belongs to a video if its file name starts with the video stem, or if its first
  `_`-separated token equals the video's clip code (last `_` token of the stem), as in
  `1DAC_tracks.csv` or `1DAC_age15_tracks.csv`.
- Results roots, layout `v2`: `<root>/<input>/<experiment>/*_tracks.csv` for each listed input,
  labelled runner output, evaluated as every derived variant.
- Results roots, layout `legacy`: `<root>/<experiment>/*_tracks.csv`, old runner output,
  evaluated only as the `legacy` variant, with the root's `input` (per-folder `input` and
  `experiment` overrides allowed). Empty (header-only) CSVs are failed runs: logged, not scored.
- External references: corner-format CSVs, input = their name, variant `as-is`.
- Aliases (`aliases.yaml`, run name -> other config names with identical parameters): an alias
  name is reported under its run name, with the alias names in the `aliases` column. Legacy CSVs
  that end up as the same (input, experiment, video) are kept once if byte-identical.

Evaluation: trackers are grouped by the set of videos they cover; each group is one TrackEval
call per similarity configuration, so TrackEval pools each tracker over exactly its own videos.
Inputs are never pooled or averaged together: `input` is part of every tracker's identity.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .formats import LABELLED, LEGACY, read_gt_tracks, read_labelled_tracks, tracker_format
from .metadata import video_metadata
from .run import METRIC_COLUMNS, POOLED, evaluate
from .trackeval_dataset import SequenceSpec, SimilarityConfig, TrackerSource

log = logging.getLogger("birdeval")

KEY_COLUMNS = ["experiment", "aliases", "input", "variant", "is_default_variant", "similarity", "s_scheme",
               "is_primary_similarity"]
DIAG_RUN_COLUMNS = ["diag_predicted_nonpositive_size", "diag_predicted_centre_outside_frame"]
GT_STAT_COLUMNS = ["gt_sqrt_area_median", "gt_sqrt_area_p5", "gt_sqrt_area_p95"]


@dataclass
class Video:
    stem: str
    clip: str
    video_path: Path
    gt_path: Path
    frame_count: int
    width: int
    height: int


@dataclass
class Run:
    """One tracker output family: an (input, experiment, kind) with one CSV per video."""

    input: str
    experiment: str
    kind: str  # labelled | legacy | corners
    files: dict[str, Path] = field(default_factory=dict)  # video stem -> CSV
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def discover_videos(paths: dict, base: Path, cache_dir: Path) -> tuple[dict[str, Video], list[str]]:
    gt_dir = resolve_path(paths["gt_dir"], base)
    videos_dir = resolve_path(paths["videos_dir"], base)
    suffix = paths["gt_suffix"]
    videos, skipped = {}, []
    for gt_path in sorted(gt_dir.glob(f"*{suffix}")):
        stem = gt_path.name[: -len(suffix)]
        video_path = videos_dir / f"{stem}.mp4"
        if not video_path.exists():
            skipped.append(f"GT without video, skipped: {gt_path} (no {video_path})")
            continue
        meta = video_metadata(video_path, cache_dir)
        videos[stem] = Video(stem, stem.rsplit("_", 1)[-1], video_path, gt_path, meta.frame_count, meta.width, meta.height)
    clips = defaultdict(list)
    for video in videos.values():
        clips[video.clip].append(video.stem)
    for clip, stems in clips.items():
        if len(stems) > 1:
            skipped.append(f"clip code {clip} is shared by {stems}: CSVs named by clip code alone are ambiguous for them")
    return videos, skipped


def match_video(csv_name: str, videos: dict[str, Video]) -> str | None:
    for stem in videos:
        if csv_name.startswith(stem + "_"):
            return stem
    first = csv_name.split("_", 1)[0]
    matches = [v.stem for v in videos.values() if v.clip == first]
    return matches[0] if len(matches) == 1 else None


def load_aliases(path: Path | None) -> dict[str, tuple[str, list[str]]]:
    """name -> (run name, all alias names of that run)."""
    if path is None or not path.exists():
        return {}
    mapping = yaml.safe_load(path.read_text()) or {}
    out = {}
    for run_name, others in mapping.items():
        out[run_name] = (run_name, list(others))
        for other in others:
            out[other] = (run_name, list(others))
    return out


def _add_csv(runs: dict, key: tuple, csv_path: Path, stem: str, kind: str, aliases: list[str], source: str, skipped: list):
    run = runs.setdefault(key, Run(input=key[0], experiment=key[1], kind=kind, aliases=aliases))
    if stem in run.files:
        if _sha256(run.files[stem]) == _sha256(csv_path):
            run.sources.append(f"{source} (identical to {run.files[stem]})")
            return
        skipped.append(f"{csv_path}: a different CSV for {key} / {stem} already exists ({run.files[stem]}), skipped")
        return
    run.files[stem] = csv_path
    run.sources.append(source)


def discover_runs(results: list[dict], externals: list[dict], videos: dict[str, Video], aliases: dict, base: Path):
    runs: dict[tuple, Run] = {}
    skipped: list[str] = []
    for root_cfg in results:
        root = resolve_path(root_cfg["path"], base)
        layout = root_cfg["layout"]
        if not root.is_dir():
            skipped.append(f"results root not found: {root}")
            continue
        if layout == "v2":
            inputs = root_cfg.get("inputs")
            if not inputs:
                raise ValueError(f"v2 results root {root} needs an `inputs` list (the <input> folder names to read)")
            folders = [(i, e) for i in inputs if (root / i).is_dir() for e in sorted((root / i).iterdir()) if e.is_dir()]
            skipped += [f"input folder {root / i} not found" for i in inputs if not (root / i).is_dir()]
        elif layout == "legacy":
            folders = [(None, e) for e in sorted(root.iterdir()) if e.is_dir()]
        else:
            raise ValueError(f"unknown results layout {layout!r}")
        overrides = root_cfg.get("folders") or {}
        for input_name, folder in folders:
            override = overrides.get(folder.name, {})
            input_name = override.get("input", input_name or root_cfg.get("input"))
            if input_name is None:
                raise ValueError(f"legacy results root {root} needs an `input`")
            name = override.get("experiment", folder.name)
            run_name, alias_names = aliases.get(name, (name, []))
            csvs = sorted(folder.glob("*_tracks.csv"))
            if not csvs:
                skipped.append(f"no *_tracks.csv in {folder}, skipped")
            for csv_path in csvs:
                stem = match_video(csv_path.name, videos)
                if stem is None:
                    skipped.append(f"{csv_path}: no GT video matches this file name, skipped")
                    continue
                kind = tracker_format(csv_path)
                if layout == "v2" and kind != LABELLED:
                    skipped.append(f"{csv_path}: expected labelled runner output in a v2 root, found {kind}, skipped")
                    continue
                if layout == "legacy" and kind != LEGACY:
                    skipped.append(f"{csv_path}: expected legacy output in a legacy root, found {kind}, skipped")
                    continue
                if len(pd.read_csv(csv_path, nrows=1)) == 0:
                    skipped.append(f"{csv_path}: empty (header only), treated as a failed run, not scored")
                    continue
                _add_csv(runs, (input_name, run_name, kind), csv_path, stem, kind, alias_names, str(folder), skipped)
    for ext in externals:
        directory = resolve_path(ext["dir"], base)
        for stem in videos:
            csv_path = directory / f"{stem}{ext['suffix']}"
            if csv_path.exists():
                _add_csv(runs, (ext["name"], f"{ext['name']}_reference", "corners"), csv_path, stem, "corners", [], str(directory), skipped)
    return runs, skipped


def gt_size_stats(gt: pd.DataFrame) -> dict:
    root_area = np.sqrt(gt["w"].to_numpy() * gt["h"].to_numpy())
    if len(root_area) == 0:
        return {c: float("nan") for c in GT_STAT_COLUMNS}
    return {"gt_sqrt_area_median": float(np.median(root_area)), "gt_sqrt_area_p5": float(np.percentile(root_area, 5)),
            "gt_sqrt_area_p95": float(np.percentile(root_area, 95))}


def predicted_diagnostics(csv_path: Path, video: Video) -> dict:
    table = read_labelled_tracks(csv_path)
    predicted = table[table["row_type"] == "predicted"]
    cx = predicted["x"] + predicted["w"] / 2
    cy = predicted["y"] + predicted["h"] / 2
    outside = (cx < 0) | (cx >= video.width) | (cy < 0) | (cy >= video.height)
    return {"diag_predicted_nonpositive_size": int(((predicted["w"] <= 0) | (predicted["h"] <= 0)).sum()),
            "diag_predicted_centre_outside_frame": int(outside.sum())}


def similarity_configs(cfg: dict) -> list[SimilarityConfig]:
    out = []
    for measure in cfg["measures"]:
        if measure == "dotd":
            d = cfg["dotd"]
            out += [SimilarityConfig("dotd", s, float(d["s_fixed"]), float(d["s_min"])) for s in d["s_schemes"]]
        else:
            out.append(SimilarityConfig(measure))
    return out


def run_info(command: list[str], repo: Path) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        direct = json.loads(importlib_metadata.distribution("trackeval").read_text("direct_url.json") or "{}")
    except importlib_metadata.PackageNotFoundError:
        direct = {}
    dirty = git("status", "--porcelain")
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_files": [line[3:] for line in dirty.splitlines()] if dirty else [],
        "trackeval_version": importlib_metadata.version("trackeval"),
        "trackeval_source": direct.get("url"),
        "trackeval_commit": direct.get("vcs_info", {}).get("commit_id"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def run_batch(config: dict, out_dir: Path, base: Path, command: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = config["paths"]
    cache_dir = resolve_path(paths.get("cache_dir", "evaluation/new/cache"), base)
    videos, skipped = discover_videos(paths, base, cache_dir)
    aliases_path = resolve_path(paths["aliases"], base) if paths.get("aliases") else None
    runs, run_skips = discover_runs(paths["results"], paths.get("external") or [], videos, load_aliases(aliases_path), base)
    skipped += run_skips
    for line in skipped:
        log.warning(line)
    log.info("%d videos with GT, %d runs", len(videos), len(runs))

    variants_cfg = config["variants"]
    default_variant = variants_cfg["default"]
    sims = similarity_configs(config["similarity"])
    primary = (config["similarity"]["primary"], config["similarity"]["dotd"]["primary_s_scheme"])

    # Trackers per run: labelled runs give every derived variant.
    trackers: dict[str, dict[str, TrackerSource]] = {}
    tracker_meta: dict[str, tuple[Run, str]] = {}
    for run in runs.values():
        variant_names = variants_cfg["derived"] if run.kind == LABELLED else ["legacy" if run.kind == LEGACY else "as-is"]
        for variant in variant_names:
            name = f"{run.input}|{run.experiment}|{variant}"
            trackers[name] = {stem: TrackerSource(path, run.kind, variant if run.kind == LABELLED else None)
                              for stem, path in run.files.items()}
            tracker_meta[name] = (run, variant)

    groups: dict[tuple, list[str]] = defaultdict(list)
    for name, per_video in trackers.items():
        groups[tuple(sorted(per_video))].append(name)

    rows = []
    te = config.get("trackeval", {})
    for video_set, names in groups.items():
        sequences = {stem: SequenceSpec(videos[stem].gt_path, videos[stem].frame_count) for stem in video_set}
        log.info("evaluating %d trackers on %d videos", len(names), len(video_set))
        rows += evaluate(sequences, {n: trackers[n] for n in names}, sims,
                         use_parallel=bool(te.get("use_parallel", False)), num_cores=int(te.get("num_cores", 1)))

    # Per-video context, loaded one video at a time.
    video_stats = {}
    for stem, video in videos.items():
        video_stats[stem] = gt_size_stats(read_gt_tracks(video.gt_path))
    run_diag = {}
    for run in runs.values():
        if run.kind == LABELLED:
            for stem, path in run.files.items():
                run_diag[(run.input, run.experiment, run.kind, stem)] = predicted_diagnostics(path, videos[stem])

    per_video, pooled = [], []
    for row in rows:
        run, variant = tracker_meta[row.pop("tracker")]
        keys = {
            "experiment": run.experiment,
            "aliases": ";".join(run.aliases),
            "input": run.input,
            "variant": variant,
            "is_default_variant": variant == default_variant,
            "similarity": row.pop("similarity"),
            "s_scheme": row.pop("s_scheme"),
        }
        keys["is_primary_similarity"] = (keys["similarity"], keys["s_scheme"]) == primary
        stem = row.pop("video")
        aggregation = row.pop("aggregation")
        if aggregation == "video":
            diag = run_diag.get((run.input, run.experiment, run.kind, stem), {c: np.nan for c in DIAG_RUN_COLUMNS})
            per_video.append({**keys, "video": stem, **row, **diag, **video_stats[stem], "tracker_csv": str(run.files[stem])})
        else:
            stems = sorted(run.files)
            diag = {c: (sum(run_diag[(run.input, run.experiment, run.kind, s)][c] for s in stems) if run.kind == LABELLED else np.nan)
                    for c in DIAG_RUN_COLUMNS}
            pooled.append({**keys, "aggregation": "pooled", "n_videos": len(stems), "videos": ";".join(stems), **row, **diag})

    per_video_df = pd.DataFrame(per_video)
    group_cols = KEY_COLUMNS
    value_cols = [*METRIC_COLUMNS, *[c for c in per_video_df.columns if c.startswith("diag_")], *GT_STAT_COLUMNS]
    value_cols = list(dict.fromkeys(c for c in value_cols if c in per_video_df.columns))
    means = per_video_df.groupby(group_cols, dropna=False)[value_cols].mean(numeric_only=True).reset_index()
    counts = per_video_df.groupby(group_cols, dropna=False)["video"].agg(n_videos="count", videos=lambda s: ";".join(sorted(s))).reset_index()
    means = means.merge(counts, on=group_cols)
    means.insert(len(group_cols), "aggregation", "per_video_mean")
    pooled_df = pd.concat([pd.DataFrame(pooled), means], ignore_index=True)

    sort_cols = ["input", "experiment", "variant", "similarity", "s_scheme"]
    per_video_df = per_video_df.sort_values([*sort_cols, "video"]).reset_index(drop=True)
    pooled_df = pooled_df.sort_values([*sort_cols, "aggregation"]).reset_index(drop=True)
    pooled_df = pooled_df[[*KEY_COLUMNS, "aggregation", "n_videos", "videos",
                           *[c for c in pooled_df.columns if c not in (*KEY_COLUMNS, "aggregation", "n_videos", "videos")]]]
    per_video_df.to_csv(out_dir / "metrics_per_video.csv", index=False)
    pooled_df.to_csv(out_dir / "metrics_pooled.csv", index=False)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    info = {**run_info(command, base), "n_videos": len(videos), "videos": sorted(videos), "n_runs": len(runs),
            "runs": [{"input": r.input, "experiment": r.experiment, "aliases": r.aliases, "kind": r.kind,
                      "files": {k: str(v) for k, v in r.files.items()}, "sources": r.sources} for r in runs.values()],
            "skipped": skipped, "default_variant": default_variant,
            "primary_similarity": {"measure": primary[0], "s_scheme": primary[1]}}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
