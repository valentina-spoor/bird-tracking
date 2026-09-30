# Copied from libs/tracker/src/spoortracker/matching.py at spoor-main commit
# 145fc6696241ec3ea1e1d3b930164bebf226538b (origin/main); imports only. See ../PROVENANCE.md.
import numpy as np

from tracker import linear_assignment, nn_matching
from tracker.association_costs import AssociationParams, make_cost
from tracker.distance_matching import closest_track_cost
from tracker.kalman_filter import KalmanFilter
from tracker.shim import Detection
from tracker.tracked_object import TrackedObject
from tracker.tracked_object_state import TrackedObjectState

MAX_IOU_DISTANCE = 0.7
MAX_AGE = 30


def match_detections_with_tracked_objects(
    tracked_objects: list[TrackedObject],
    detections: list[Detection],
    kalman_filter: KalmanFilter,
    nearest_neighbor_metric: nn_matching.NearestNeighborDistanceMetric,
    kalman_only_position: bool = True,
    association_method: str = "original",
    association_params: AssociationParams | None = None,
):
    new_cost = None
    if association_method != "original":
        new_cost = make_cost(
            association_method,
            association_params or AssociationParams(),
            nearest_neighbor_metric.matching_threshold,
            MAX_AGE,
        )

    def gated_metric(tracked_object: list[TrackedObject], dets, max_age, track_indices, detection_indices):
        
        features = np.array([dets[i].feature for i in detection_indices])
        feature_corresponding_ids = np.array(
            [tracked_object[i].id for i in track_indices]
        )

        if association_method == "original":
            cost_matrix = nearest_neighbor_metric.distance(
                features,
                feature_corresponding_ids,
            )
        else:
            cost_matrix = new_cost(tracked_object, dets, max_age, track_indices, detection_indices)
        
        cost_matrix = linear_assignment.gate_cost_matrix(
            kalman_filter,
            cost_matrix,
            tracked_object,
            dets,
            track_indices,
            detection_indices,
            only_position=kalman_only_position,
        )

        return cost_matrix

    # Split track set into confirmed and unconfirmed tracks.
    confirmed_tracks = []
    unconfirmed_tracks = []
    for i in range(len(tracked_objects)):
        if tracked_objects[i].state == TrackedObjectState.CONFIRMED:
            confirmed_tracks.append(i)
        else:
            unconfirmed_tracks.append(i)

    # Associate confirmed tracks using appearance features.
    (
        matches_a,
        unmatched_tracks_a,
        unmatched_detections,
    ) = linear_assignment.matching_cascade(
        gated_metric,
        nearest_neighbor_metric.matching_threshold,
        MAX_AGE,
        tracked_objects,
        detections,
        confirmed_tracks,
    )

    # Associate remaining tracks together with unconfirmed tracks using IOU.
    iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if tracked_objects[k].age < MAX_AGE]
    unmatched_tracks_a = [k for k in unmatched_tracks_a if tracked_objects[k].age != 1]
    (
        matches_b,
        unmatched_tracks_b,
        unmatched_detections,
    ) = linear_assignment.min_cost_matching(
        closest_track_cost if new_cost is None else new_cost,
        MAX_AGE,
        nearest_neighbor_metric.matching_threshold,
        tracked_objects,
        detections,
        iou_track_candidates,
        unmatched_detections,
    )

    matches = matches_a + matches_b
    unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

    return matches, unmatched_tracks, unmatched_detections
