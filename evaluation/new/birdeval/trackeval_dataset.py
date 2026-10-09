"""TrackEval dataset class over our internal (frame, id, box) tables.

One dataset instance has one similarity (measure + DotD scale scheme), a list of sequences
(videos) and a list of trackers (experiment / input / variant combinations), so one
`Evaluator.evaluate` call gives every per-sequence result and TrackEval's pooled `COMBINED_SEQ`.

Tables are loaded lazily, one (tracker, sequence) at a time, from paths held in picklable specs,
so TrackEval's parallel mode works and only one video's tables are in memory per worker. Each
sequence has exactly `frame_count` timesteps (from the video): frames with no GT and no
prediction are counted as empty frames. GT boxes carry their DotD scale as a fifth column so
`_calculate_similarities`, which only sees boxes, can apply the per-track or per-box scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import _numpy_compat  # noqa: F401  (must run before trackeval is imported)
from trackeval.datasets._base_dataset import _BaseDataset  # noqa: E402
from trackeval.utils import TrackEvalException  # noqa: E402

from .formats import read_corner_tracks, read_gt_tracks, read_labelled_tracks, read_legacy_tracks  # noqa: E402
from .similarity import gt_box_scales, similarity_matrix  # noqa: E402
from .variants import DERIVED_VARIANTS, derive  # noqa: E402

CLASS_NAME = "bird"


@dataclass(frozen=True)
class SequenceSpec:
    gt_path: Path
    frame_count: int


@dataclass(frozen=True)
class TrackerSource:
    """Where one tracker's boxes for one sequence come from.

    kind `labelled` (new runner output; `variant` picks the derivation), `legacy` (old runner
    output, as it is) or `corners` (external reference such as the cloud tracker, as it is).
    """

    path: Path
    kind: str
    variant: str | None = None


@dataclass(frozen=True)
class SimilarityConfig:
    measure: str
    s_scheme: str | None = None
    s_fixed: float = 9.0
    s_min: float = 4.0


def load_tracker_table(source: TrackerSource) -> pd.DataFrame:
    if source.kind == "labelled":
        if source.variant not in DERIVED_VARIANTS:
            raise ValueError(f"labelled source needs a derived variant, got {source.variant!r}")
        return derive(read_labelled_tracks(source.path), source.variant)
    if source.kind == "legacy":
        return read_legacy_tracks(source.path)
    if source.kind == "corners":
        return read_corner_tracks(source.path)
    raise ValueError(f"unknown tracker source kind {source.kind!r}")


def _per_timestep(table: pd.DataFrame, frame_count: int, extra: np.ndarray | None = None):
    """Split a (frame, id, box) table into per-frame id arrays and box arrays."""
    if len(table) and int(table["frame"].max()) >= frame_count:
        raise TrackEvalException(f"frame {int(table['frame'].max())} is outside the video (0..{frame_count - 1})")
    width = 4 if extra is None else 5
    boxes = table[["x", "y", "w", "h"]].to_numpy(dtype=float)
    if extra is not None:
        boxes = np.column_stack([boxes, extra])
    ids_t = [np.empty(0, dtype=int) for _ in range(frame_count)]
    dets_t = [np.empty((0, width)) for _ in range(frame_count)]
    frames = table["frame"].to_numpy()
    ids = table["id"].to_numpy()
    if len(frames):
        order = np.argsort(frames, kind="stable")
        bounds = np.searchsorted(frames[order], np.arange(frame_count + 1))
        for t in range(frame_count):
            rows = order[bounds[t]:bounds[t + 1]]
            if len(rows):
                ids_t[t] = ids[rows].astype(int)
                dets_t[t] = boxes[rows]
    return ids_t, dets_t


class BirdDataset(_BaseDataset):
    def __init__(
        self,
        sequences: dict[str, SequenceSpec],
        trackers: dict[str, dict[str, TrackerSource]],
        similarity: SimilarityConfig,
    ):
        super().__init__()
        for name, per_seq in trackers.items():
            missing = set(sequences) - set(per_seq)
            if missing:
                raise ValueError(f"tracker {name!r} has no source for sequences {sorted(missing)}")
        self.sequences = dict(sequences)
        self.trackers = {name: dict(per_seq) for name, per_seq in trackers.items()}
        self.similarity = similarity
        self.tracker_list = list(self.trackers)
        self.seq_list = list(self.sequences)
        self.class_list = [CLASS_NAME]
        self.should_classes_combine = False
        self.use_super_categories = False
        self.output_fol = None
        self.output_sub_fol = ""

    @staticmethod
    def get_default_dataset_config():
        return {}

    def get_name(self):
        return "BirdDataset"

    def get_display_name(self, tracker):
        return tracker

    def get_output_fol(self, tracker):
        return None

    def _load_raw_file(self, tracker, seq, is_gt):
        spec = self.sequences[seq]
        if is_gt:
            gt = read_gt_tracks(spec.gt_path)
            scales = None
            if self.similarity.measure == "dotd":
                scales = gt_box_scales(gt, self.similarity.s_scheme, self.similarity.s_fixed, self.similarity.s_min)
            ids_t, dets_t = _per_timestep(gt, spec.frame_count, extra=scales)
            return {
                "num_timesteps": spec.frame_count,
                "gt_ids": ids_t,
                "gt_dets": dets_t,
                "gt_classes": [np.zeros(len(ids), dtype=int) for ids in ids_t],
                "gt_crowd_ignore_regions": [np.empty((0, 4)) for _ in ids_t],
                "gt_extras": {},
                "seq": seq,
            }
        table = load_tracker_table(self.trackers[tracker][seq])
        ids_t, dets_t = _per_timestep(table, spec.frame_count)
        return {
            "tracker_ids": ids_t,
            "tracker_dets": dets_t,
            "tracker_classes": [np.zeros(len(ids), dtype=int) for ids in ids_t],
            # Not used by HOTA, CLEAR, Identity or Count; TrackEval only needs the key.
            "tracker_confidences": [np.ones(len(ids)) for ids in ids_t],
        }

    def get_preprocessed_seq_data(self, raw_data, cls):
        gt_ids, tracker_ids = raw_data["gt_ids"], raw_data["tracker_ids"]
        unique_gt = np.unique(np.concatenate(gt_ids)) if any(len(i) for i in gt_ids) else np.empty(0, dtype=int)
        unique_tr = np.unique(np.concatenate(tracker_ids)) if any(len(i) for i in tracker_ids) else np.empty(0, dtype=int)
        gt_map = {int(v): k for k, v in enumerate(unique_gt)}
        tr_map = {int(v): k for k, v in enumerate(unique_tr)}
        data = {
            "num_timesteps": raw_data["num_timesteps"],
            "gt_ids": [np.array([gt_map[int(i)] for i in ids], dtype=int) for ids in gt_ids],
            "tracker_ids": [np.array([tr_map[int(i)] for i in ids], dtype=int) for ids in tracker_ids],
            "gt_dets": raw_data["gt_dets"],
            "tracker_dets": raw_data["tracker_dets"],
            "similarity_scores": raw_data["similarity_scores"],
            "num_gt_ids": len(unique_gt),
            "num_tracker_ids": len(unique_tr),
            "num_gt_dets": int(sum(len(i) for i in gt_ids)),
            "num_tracker_dets": int(sum(len(i) for i in tracker_ids)),
            "seq": raw_data["seq"],
        }
        self._check_unique_ids(data, after_preproc=True)
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if len(gt_dets_t) == 0 or len(tracker_dets_t) == 0:
            return np.zeros((len(gt_dets_t), len(tracker_dets_t)))
        gt_boxes = gt_dets_t[:, :4]
        scales = gt_dets_t[:, 4] if gt_dets_t.shape[1] == 5 else None
        return similarity_matrix(self.similarity.measure, gt_boxes, tracker_dets_t, scales)
