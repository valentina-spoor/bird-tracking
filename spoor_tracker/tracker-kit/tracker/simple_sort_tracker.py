# Copied from libs/tracker/src/spoortracker/simple_sort_tracker.py at spoor-main commit
# 145fc6696241ec3ea1e1d3b930164bebf226538b (origin/main); imports only. See ../PROVENANCE.md.
import logging
from itertools import count
from typing import Callable

import numpy as np

from tracker.shim import (
    Detection,
    DetectionNoCrop,
    distance_between_two_bounding_boxes,
    xyxy_to_cxcyah,
)

from .kalman_filter import KalmanFilter
from .kalman_tracker_state import KalmanTrackerState
from .matching import match_detections_with_tracked_objects
from .nn_matching import NearestNeighborDistanceMetric
from .tracked_object import (
    TrackedObject,
    get_tracked_object_by_id,
    get_tracked_objects_with_states,
)
from .tracked_object_state import TrackedObjectState

FilterCallable = Callable[[list[TrackedObject]], list[TrackedObject]]

logger = logging.getLogger(__name__)


class SimpleSORTTracker:
    def __init__(
        self,
        euclidean_matching_threshold: float,
        max_age: int,
        tentative_threshold: int = 3,
        kalman_gating: str = "position",
    ):
        self.kalman_filter = KalmanFilter()
        self.nearest_neighbour_metric = NearestNeighborDistanceMetric("euclidean", euclidean_matching_threshold)
        self.tracked_objects: list[TrackedObject] = []
        self.deleted_tracked_objects: list[TrackedObject] = []
        self.max_age = max_age
        self.tentative_threshold = tentative_threshold
        if kalman_gating not in {"position", "full"}:
            raise ValueError(
                "kalman_gating must be either 'position' or 'full'"
            )

        self.kalman_gating = kalman_gating
        self.kalman_only_position = kalman_gating == "position"
        self.id_generator = count()

    def match_and_track(self, new_detections: list[Detection | DetectionNoCrop]):
        # NOTE!
        # This is a big function that could've been split into smaller functions.
        # However, since the tracked object states are being mutated, a split may be be harder follow
        # as you would need to context switch. If you think this is false, feel free to change it.
        # NOTE2!
        # Be aware that the tracked objects will be in the predicted state after this function is called.

        matched_pairs, not_matched_tracked_objects, not_matched_detections = (
            match_detections_with_tracked_objects(
                self.tracked_objects,
                new_detections,
                self.kalman_filter,
                self.nearest_neighbour_metric,
                self.kalman_only_position,
            )
        )

        for tracked_object_idx, detection_idx in matched_pairs:
            tracked_object = self.tracked_objects[tracked_object_idx]
            tracked_object.update(self.kalman_filter, new_detections[detection_idx])
            if tracked_object.state == TrackedObjectState.CONFIRMED:
                continue

            if (
                tracked_object.state == TrackedObjectState.TENTATIVE
                and len(tracked_object.detections) < self.tentative_threshold
            ):
                continue

            tracked_object.state = TrackedObjectState.CONFIRMED

        # Sort from high to low index order to pop items in the back to avoid index errors.
        for idx in sorted(not_matched_tracked_objects, reverse=True):
            is_confirmed = self.tracked_objects[idx].state == TrackedObjectState.CONFIRMED
            if is_confirmed and self.tracked_objects[idx].age < self.max_age:
                continue

            self.tracked_objects[idx].state = TrackedObjectState.DELETED
            self.deleted_tracked_objects.append(self.tracked_objects.pop(idx))

        for idx in not_matched_detections:
            object_id = next(self.id_generator)
            detection = new_detections[idx]
            initial_mean, initial_covariance = self.kalman_filter.initiate(
                xyxy_to_cxcyah(detection.bounding_box).values,
            )
            kalman_tracker_state = KalmanTrackerState(initial_mean, initial_covariance, self.max_age)
            self.tracked_objects.append(TrackedObject(object_id, detection, kalman_tracker_state))

        # NOTE!
        # I really have no idea what this does
        features = []
        features_ids = []
        for tracked_object in get_tracked_objects_with_states(
            self.tracked_objects, set([TrackedObjectState.CONFIRMED])
        ):
            features.extend(tracked_object.kalman_tracker_state.features)
            features_ids.extend([tracked_object.id] * len(tracked_object.kalman_tracker_state.features))

        self.nearest_neighbour_metric.partial_fit(
            np.asarray(features),
            np.asarray(features_ids),
            list(set(features_ids)),
        )

        # Tracked objects states are already predicted for the next frame
        # This means tracked objects are in predicted state in the remaining processing steps of the current frame and
        # at the start of the next frame. This is because no new information about the tracked objects
        # will be supplied, so we can already do a prediction. Staying ahead of the game.
        # It enables access to the predicted location of the tracked objects.
        for tracked_object in self.tracked_objects:
            tracked_object.predict(self.kalman_filter)

    def filter_deleted_tracks(self, filters: list[FilterCallable]) -> None:
        for filter_callable in filters:
            logger.info(f"Filtering deleted tracks with {filter_callable}")
            before = len(self.deleted_tracked_objects)
            logger.info(f"Before: {before}")
            self.deleted_tracked_objects = filter_callable(self.deleted_tracked_objects)
            after = len(self.deleted_tracked_objects)
            logger.info(f"After: {after}")
            logger.info(f"{filter_callable} filtered out {before - after} tracks")

    def concatenate_tracks(self, fps: int, concatenate_distance_in_seconds: float, distance_threshold: float):
        distance_threshold = distance_threshold * concatenate_distance_in_seconds
        temporal_distance = fps * concatenate_distance_in_seconds
        all_tracked_objects = self.tracked_objects + self.deleted_tracked_objects
        all_tracked_objects = sort_tracked_objects_by_last_frame_number(all_tracked_objects)

        while all_tracked_objects:
            tracked_object = all_tracked_objects.pop(0)
            while True:
                best_match = find_concatenation_candidate(
                    tracked_object=tracked_object,
                    all_tracked_objects=all_tracked_objects,
                    distance_threshold=distance_threshold,
                    temporal_distance=temporal_distance,
                )
                if best_match is None:
                    break

                tracked_object += best_match
                all_tracked_objects.remove(best_match)
                if redundant_object := get_tracked_object_by_id(self.tracked_objects, best_match.id):
                    self.tracked_objects.remove(redundant_object)
                if redundant_object := get_tracked_object_by_id(self.deleted_tracked_objects, best_match.id):
                    self.deleted_tracked_objects.remove(redundant_object)

    def finalize_tracking(self):
        # Only call this method when video processing is done.
        while self.tracked_objects:
            tracked_object = self.tracked_objects.pop(0)
            tracked_object.state = TrackedObjectState.DELETED
            self.deleted_tracked_objects.append(tracked_object)


def find_concatenation_candidate(
    tracked_object: TrackedObject,
    all_tracked_objects: list[TrackedObject],
    distance_threshold: float,
    temporal_distance: float,
) -> TrackedObject | None:

    min_distance = 1e5
    best_candidate = None

    for candidate in all_tracked_objects:
        time_diff = np.abs(candidate.detections[0].frame_number - tracked_object.detections[-1].frame_number)
        if (
            candidate.detections[0].frame_number > tracked_object.detections[-1].frame_number
            and time_diff < temporal_distance
        ):
            start_coordinates_candidate = candidate.detections[0].bounding_box
            end_coordinates_current = tracked_object.detections[-1].bounding_box

            pixel_distance = distance_between_two_bounding_boxes(start_coordinates_candidate, end_coordinates_current)

            if pixel_distance < distance_threshold and pixel_distance < min_distance:
                min_distance = pixel_distance
                best_candidate = candidate

    return best_candidate


def sort_tracked_objects_by_last_frame_number(tracked_objects: list[TrackedObject]) -> list[TrackedObject]:
    return sorted(tracked_objects, key=lambda x: x.detections[-1].frame_number)
