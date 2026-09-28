# Copied from libs/tracker/src/spoortracker/distance_matching.py at spoor-main commit
# 145fc6696241ec3ea1e1d3b930164bebf226538b (origin/main); imports only. See ../PROVENANCE.md.
# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

import numpy as np
from scipy import spatial

from tracker import linear_assignment
from tracker.shim import Detection, xyxy_to_xywh
from tracker.tracked_object import TrackedObject


def closest_track(bbox, candidates):
    bbox_tl, _ = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    closest_track, index = spatial.KDTree([bbox_tl]).query(candidates_tl)

    return closest_track


def closest_track_cost(
    tracked_objects: list[TrackedObject],
    detections: list[Detection],
    max_age,
    track_indices=None,
    detection_indices=None,
):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracked_objects))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracked_objects[track_idx].age > max_age:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracked_objects[track_idx].kalman_tracker_state.to_tlwh()
        candidates = np.asarray([xyxy_to_xywh(detections[i].bounding_box).values for i in detection_indices])
        cost_matrix[row, :] = closest_track(bbox, candidates)
    return cost_matrix
