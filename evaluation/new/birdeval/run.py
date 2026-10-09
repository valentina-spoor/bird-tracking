"""Evaluate trackers on sequences with TrackEval and flatten the results into plain rows.

`evaluate()` runs one `Evaluator.evaluate` call per similarity configuration, over every tracker
and every sequence given, and returns one row per (tracker, similarity, sequence) plus one per
(tracker, similarity) for TrackEval's pooled `COMBINED_SEQ`. HOTA-family values are the mean
over TrackEval's alpha thresholds (0.05 to 0.95), as TrackEval itself reports them.
"""

from __future__ import annotations

import numpy as np

from . import _numpy_compat  # noqa: F401
import trackeval  # noqa: E402

from .diagnostics import IdentityChanges  # noqa: E402
from .trackeval_dataset import CLASS_NAME, BirdDataset, SequenceSpec, SimilarityConfig, TrackerSource  # noqa: E402

POOLED = "COMBINED_SEQ"

HOTA_FIELDS = ["HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr"]
IDENTITY_FIELDS = ["IDF1", "IDP", "IDR", "IDTP", "IDFP", "IDFN"]
CLEAR_FIELDS = ["MOTA", "MOTP", "IDSW", "Frag", "MT", "PT", "ML", "CLR_TP", "CLR_FP", "CLR_FN", "MODA", "CLR_Re", "CLR_Pr"]
COUNT_FIELDS = {"Dets": "n_pred_boxes", "GT_Dets": "n_gt_boxes", "IDs": "n_pred_ids", "GT_IDs": "n_gt_ids"}
# Our own diagnostics (not standard metrics), prefixed `diag_`. See diagnostics.py.
DIAGNOSTIC_FIELDS = [IdentityChanges.FIELD]
METRIC_COLUMNS = [*HOTA_FIELDS, *IDENTITY_FIELDS, *CLEAR_FIELDS, *COUNT_FIELDS.values(), "id_ratio", "n_frames", "trackeval_frames", *DIAGNOSTIC_FIELDS]

EVALUATOR_CONFIG = {
    "USE_PARALLEL": False,
    "NUM_PARALLEL_CORES": 1,
    "BREAK_ON_ERROR": True,
    "RETURN_ON_ERROR": False,
    "LOG_ON_ERROR": None,
    "PRINT_RESULTS": False,
    "PRINT_ONLY_COMBINED": False,
    "PRINT_CONFIG": False,
    "TIME_PROGRESS": False,
    "DISPLAY_LESS_PROGRESS": True,
    "OUTPUT_SUMMARY": False,
    "OUTPUT_EMPTY_CLASSES": False,
    "OUTPUT_DETAILED": False,
    "PLOT_CURVES": False,
}


def flatten(metric_results: dict) -> dict:
    hota, clear, identity, count = (metric_results[m] for m in ("HOTA", "CLEAR", "Identity", "Count"))
    row = {name: float(np.mean(hota[name])) for name in HOTA_FIELDS}
    row.update({name: float(identity[name]) for name in IDENTITY_FIELDS})
    row.update({name: float(clear[name]) for name in CLEAR_FIELDS})
    row.update({column: int(count[name]) for name, column in COUNT_FIELDS.items()})
    row["id_ratio"] = row["n_pred_ids"] / row["n_gt_ids"] if row["n_gt_ids"] else float("nan")
    # Timesteps CLEAR iterated over (0 if the tracker has no boxes at all, see CLEAR.eval_sequence).
    row["trackeval_frames"] = int(clear["CLR_Frames"])
    for name, values in metric_results.items():
        if name not in ("HOTA", "CLEAR", "Identity", "Count"):
            row.update({field: int(value) for field, value in values.items()})
    return row


def evaluate(
    sequences: dict[str, SequenceSpec],
    trackers: dict[str, dict[str, TrackerSource]],
    similarities: list[SimilarityConfig],
    use_parallel: bool = False,
    num_cores: int = 1,
    extra_metrics: list | None = None,
) -> list[dict]:
    """One row per (tracker, similarity, sequence or COMBINED_SEQ). `extra_metrics` are further
    TrackEval metric instances whose integer fields are added to each row."""
    config = {**EVALUATOR_CONFIG, "USE_PARALLEL": use_parallel, "NUM_PARALLEL_CORES": num_cores}
    evaluator = trackeval.Evaluator(config)
    # Evaluator.evaluate always appends Count itself.
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        IdentityChanges(),
        *(extra_metrics or []),
    ]
    rows = []
    for similarity in similarities:
        dataset = BirdDataset(sequences, trackers, similarity)
        output, messages = evaluator.evaluate([dataset], metrics)
        for tracker, per_seq in output[dataset.get_name()].items():
            if per_seq is None:
                raise RuntimeError(f"TrackEval failed on {tracker}: {messages[dataset.get_name()][tracker]}")
            for seq, per_class in per_seq.items():
                n_frames = sum(s.frame_count for s in sequences.values()) if seq == POOLED else sequences[seq].frame_count
                rows.append({
                    "tracker": tracker,
                    "similarity": similarity.measure,
                    "s_scheme": similarity.s_scheme if similarity.measure == "dotd" else "none",
                    "video": seq if seq != POOLED else None,
                    "aggregation": "pooled" if seq == POOLED else "video",
                    **flatten(per_class[CLASS_NAME]),
                    "n_frames": n_frames,
                })
    return rows
