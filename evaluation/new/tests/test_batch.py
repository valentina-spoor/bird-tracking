"""Discovery and output tables of the batch run, on a synthetic results tree built in tmp_path."""

import shutil

import cv2
import numpy as np
import pandas as pd
import yaml

from conftest import FIXTURES, TOY_FRAME_COUNT
from birdeval.batch import run_batch

STEM = "20990101_000000_TOYS"


def _video(path, frames=TOY_FRAME_COUNT):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def _labelled_from_legacy(src, dst):
    table = pd.read_csv(src)
    table["row_type"] = "matched"
    table["track_confirmed"] = True
    table.to_csv(dst, index=False)


def test_batch_discovery_aliases_legacy_and_tables(tmp_path):
    videos, gt_dir = tmp_path / "videos", tmp_path / "gt"
    videos.mkdir(), gt_dir.mkdir()
    _video(videos / f"{STEM}.mp4")
    shutil.copy(FIXTURES / "swap_gt.csv", gt_dir / f"{STEM}_reference_tracks_corrected.csv")

    v2 = tmp_path / "results_v2"
    (v2 / "yolo" / "baseline").mkdir(parents=True)
    _labelled_from_legacy(FIXTURES / "swap_tracks.csv", v2 / "yolo" / "baseline" / "TOYS_tracks.csv")
    (v2 / "aliases.yaml").write_text(yaml.safe_dump({"baseline": ["kalman_normal", "original"]}))
    (v2 / "regression").mkdir()  # not a listed input: must be ignored
    shutil.copy(FIXTURES / "perfect_tracks.csv", v2 / "regression" / "TOYS_tracks.csv")

    legacy = tmp_path / "results"
    for folder in ("kalman_normal", "original"):  # aliases of baseline with byte-identical CSVs
        (legacy / folder).mkdir(parents=True)
        shutil.copy(FIXTURES / "swap_tracks.csv", legacy / folder / "TOYS_suffix_tracks.csv")
    (legacy / "failed").mkdir()
    (legacy / "failed" / "TOYS_tracks.csv").write_text("frame_number,track_id,x,y,w,h\n")
    (legacy / "other_video").mkdir()
    shutil.copy(FIXTURES / "swap_tracks.csv", legacy / "other_video" / "ZZZZ_tracks.csv")

    config = {
        "paths": {
            "videos_dir": str(videos), "gt_dir": str(gt_dir), "gt_suffix": "_reference_tracks_corrected.csv",
            "results": [{"path": str(v2), "layout": "v2", "inputs": ["yolo"]},
                        {"path": str(legacy), "layout": "legacy", "input": "yolo"}],
            "aliases": str(v2 / "aliases.yaml"), "external": [], "cache_dir": str(tmp_path / "cache"),
        },
        "similarity": {"measures": ["dotd", "iou"], "primary": "dotd",
                       "dotd": {"s_schemes": ["per_track"], "primary_s_scheme": "per_track", "s_fixed": 9.0, "s_min": 4.0}},
        "variants": {"default": "matched_final", "derived": ["matched_online", "matched_final", "frozen"]},
        "trackeval": {"use_parallel": False, "num_cores": 1},
    }
    out = tmp_path / "out"
    info = run_batch(config, out, tmp_path, ["test"])

    per_video = pd.read_csv(out / "metrics_per_video.csv")
    pooled = pd.read_csv(out / "metrics_pooled.csv")
    assert (out / "config_used.yaml").exists() and (out / "run_info.json").exists()

    # one labelled run (3 variants) + one legacy run (deduplicated) = 4 trackers x 2 similarities
    assert len(per_video) == 8
    assert set(per_video["experiment"]) == {"baseline"}
    assert set(per_video["aliases"]) == {"kalman_normal;original"}
    assert sorted(per_video["variant"].unique()) == ["frozen", "legacy", "matched_final", "matched_online"]
    assert per_video["n_frames"].eq(TOY_FRAME_COUNT).all()
    assert per_video.loc[per_video["variant"] == "matched_final", "is_default_variant"].all()
    # the swap case: 2 identity switches whatever the variant (no predicted rows here)
    assert per_video["IDSW"].eq(2).all()
    # predicted-row diagnostics exist for the labelled run only
    assert per_video.loc[per_video["variant"] == "legacy", "diag_predicted_nonpositive_size"].isna().all()
    assert per_video.loc[per_video["variant"] != "legacy", "diag_predicted_nonpositive_size"].eq(0).all()
    # both aggregations, and with one video they agree
    assert sorted(pooled["aggregation"].unique()) == ["per_video_mean", "pooled"]
    wide = pooled.pivot_table(index=["variant", "similarity"], columns="aggregation", values="HOTA")
    np.testing.assert_allclose(wide["pooled"], wide["per_video_mean"])

    skipped = "\n".join(info["skipped"])
    assert "failed/TOYS_tracks.csv: empty" in skipped
    assert "ZZZZ_tracks.csv: no GT video matches" in skipped
    legacy_run = next(r for r in info["runs"] if r["kind"] == "legacy")
    assert any("identical to" in s for s in legacy_run["sources"])
