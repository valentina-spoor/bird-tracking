"""Our own diagnostics, run as TrackEval metrics so they see the same similarity matrices and are
aggregated per video and pooled like every other metric. They are NOT standard MOT metrics.

`IdentityChanges` (column `diag_id_changes`): in every frame, GT and predicted boxes are paired
by a plain Hungarian matching on the similarity (maximising total similarity, no preference for
keeping last frame's pairs), and a pair counts only if its similarity is at least `threshold`
(0.5, as in CLEAR and Identity). Each time a predicted track is paired with a different GT bird
than the last time it was paired, that is one identity change. Unlike CLEAR's IDSW, which keeps
the previous pairing whenever it is still above threshold, this counts a track that jumps
between two nearly overlapping birds. Caveat: with noisy detections it can over-count, because a
box that sits between two birds, or a merged box covering several, can flip its best pairing
from frame to frame without any real identity error.

`FragAttribution` (fields `Frag_total`, `Frag_after_gt_gap`): reproduces CLEAR's Frag exactly
and counts how many of its restarts happen right after a frame in which that GT bird had no box
(an annotation gap: occluded or not visible) while the bird was matched the last time it was
visible, i.e. restarts that only the gap caused. CLEAR resets "was tracked in the previous frame"
for a bird that is absent from a frame, so a bird tracked perfectly across a gap still adds a
fragment. Used to quantify that effect in the README; not reported in the result tables.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import _numpy_compat  # noqa: F401
from trackeval.metrics._base_metric import _BaseMetric  # noqa: E402

_EPS = np.finfo("float").eps


class _SummedIntegerMetric(_BaseMetric):
    def __init__(self, fields):
        super().__init__()
        self.integer_fields = list(fields)
        self.fields = self.integer_fields
        self.summary_fields = self.fields

    def _sum(self, all_res):
        return {field: self._combine_sum(all_res, field) for field in self.integer_fields}

    def combine_sequences(self, all_res):
        return self._sum(all_res)

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=None):
        return self._sum(all_res)

    def combine_classes_det_averaged(self, all_res):
        return self._sum(all_res)


class IdentityChanges(_SummedIntegerMetric):
    FIELD = "diag_id_changes"

    def __init__(self, config=None):
        super().__init__([self.FIELD])
        self.threshold = float((config or {}).get("THRESHOLD", 0.5))

    def eval_sequence(self, data):
        last_gt = np.full(data["num_tracker_ids"], -1, dtype=int)
        changes = 0
        for gt_ids_t, tracker_ids_t, similarity in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
                continue
            rows, cols = linear_sum_assignment(-similarity)
            keep = similarity[rows, cols] >= self.threshold - _EPS
            for gt_id, tracker_id in zip(gt_ids_t[rows[keep]], tracker_ids_t[cols[keep]]):
                if last_gt[tracker_id] >= 0 and last_gt[tracker_id] != gt_id:
                    changes += 1
                last_gt[tracker_id] = gt_id
        return {self.FIELD: changes}


class FragAttribution(_SummedIntegerMetric):
    def __init__(self, config=None):
        super().__init__(["Frag_total", "Frag_after_gt_gap"])
        self.threshold = float((config or {}).get("THRESHOLD", 0.5))

    def eval_sequence(self, data):
        """Same loop as trackeval.metrics.CLEAR.eval_sequence, keeping only what Frag needs."""
        if data["num_tracker_dets"] == 0 or data["num_gt_dets"] == 0:
            return {"Frag_total": 0, "Frag_after_gt_gap": 0}
        n = data["num_gt_ids"]
        starts = np.zeros(n, dtype=int)
        starts_after_gap = np.zeros(n, dtype=int)
        prev_timestep_tracker_id = np.full(n, np.nan)
        present_previous_frame = np.zeros(n, dtype=bool)
        matched_when_last_present = np.zeros(n, dtype=bool)
        for gt_ids_t, tracker_ids_t, similarity in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
            present_now = np.zeros(n, dtype=bool)
            present_now[gt_ids_t] = True
            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
                matched_when_last_present[gt_ids_t] = False
                present_previous_frame = present_now
                continue
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < self.threshold - _EPS] = 0
            rows, cols = linear_sum_assignment(-score_mat)
            keep = score_mat[rows, cols] > 0 + _EPS
            matched_gt = gt_ids_t[rows[keep]]
            matched_tr = tracker_ids_t[cols[keep]]
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt] = matched_tr
            started = not_previously_tracked & ~np.isnan(prev_timestep_tracker_id)
            starts += started
            starts_after_gap += started & (starts > 1) & ~present_previous_frame & matched_when_last_present
            matched_now = np.zeros(n, dtype=bool)
            matched_now[matched_gt] = True
            matched_when_last_present[gt_ids_t] = matched_now[gt_ids_t]
            present_previous_frame = present_now
        frag_total = int(np.sum(starts[starts > 0] - 1))
        return {"Frag_total": frag_total, "Frag_after_gt_gap": int(starts_after_gap.sum())}
