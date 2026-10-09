"""Shared test helpers.

Toy fixtures live in tests/fixtures/ (committed). Tests on the real clips need the git-ignored
data: they read paths from the config named by $BIRDEVAL_CONFIG (default
evaluation/new/configs/local.yaml) and the labelled re-runs in experiments/results_v2/, and
are skipped when those are not available.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

NEW = Path(__file__).resolve().parents[1]
REPO = NEW.parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(NEW))

from birdeval.run import evaluate  # noqa: E402
from birdeval.trackeval_dataset import SequenceSpec, SimilarityConfig, TrackerSource  # noqa: E402

TOY_FRAME_COUNT = 10
ALL_SIMILARITIES = [SimilarityConfig("dotd", "per_track"), SimilarityConfig("dotd", "fixed"),
                    SimilarityConfig("dotd", "per_box"), SimilarityConfig("iou")]


def toy_rows(case: str, similarities=None) -> list[dict]:
    """Per-video rows for one toy case (one sequence, one legacy-format tracker)."""
    seqs = {case: SequenceSpec(FIXTURES / f"{case}_gt.csv", TOY_FRAME_COUNT)}
    trackers = {case: {case: TrackerSource(FIXTURES / f"{case}_tracks.csv", "legacy")}}
    rows = evaluate(seqs, trackers, similarities or ALL_SIMILARITIES)
    return [r for r in rows if r["aggregation"] == "video"]


def _local_paths() -> dict | None:
    path = Path(os.environ.get("BIRDEVAL_CONFIG", NEW / "configs/local.yaml"))
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())["paths"]


@pytest.fixture(scope="session")
def clip_1dac():
    """Paths for the 1DAC clip, or skip if the data is not on this machine."""
    paths = _local_paths()
    if paths is None:
        pytest.skip("no local config with data paths")
    stem = "20251012_164031_1DAC"
    legacy_root = next((Path(r["path"]) for r in paths["results"] if r["layout"] == "legacy"), None)
    files = {
        "video": Path(paths["videos_dir"]) / f"{stem}.mp4",
        "gt": Path(paths["gt_dir"]) / f"{stem}{paths['gt_suffix']}",
        "labelled_gt_input": REPO / "experiments/results_v2/gt/baseline/1DAC_tracks.csv",
        "labelled_yolo_input": REPO / "experiments/results_v2/yolo/baseline/1DAC_tracks.csv",
        "legacy_gt_input": legacy_root / "test_original_GT1/1DAC_tracks.csv" if legacy_root else None,
    }
    missing = [k for k, v in files.items() if v is None or not Path(v).exists()]
    if missing:
        pytest.skip(f"1DAC data not available: {missing}")
    return files
