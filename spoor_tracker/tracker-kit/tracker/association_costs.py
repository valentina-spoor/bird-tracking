"""Association costs that go beyond the original (x, y, area) nearest-neighbour distance.

The original cost compares ``(x, y, area)`` of a detection against every past sample of a track,
so the only "appearance" it sees is box area, which is a geometric quantity and not a visual one.
The methods here separate the two ingredients:

``motion``      pixel distance from the detection centre to the Kalman-predicted centre.
                Control: isolates "drop the area term / use the prediction" from "add appearance".
``appearance``  blended visual distance (colour, patch NCC, contrast) between the detection and
                the track's template, feasible only inside the motion gate.
``fused``       ``(1 - w) * pixel_distance + w * max_distance * appearance_distance``.

All three stay in pixel units, so ``euclidean_matching_threshold`` keeps its meaning as the
maximum accepted cost, and candidates farther than it from the prediction are never feasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracker.appearance import appearance_distance_matrix
from tracker.linear_assignment import INFTY_COST
from tracker.shim import Detection
from tracker.tracked_object import TrackedObject

ASSOCIATION_METHODS = ("original", "motion", "appearance", "fused")
APPEARANCE_METHODS = ("appearance", "fused")


@dataclass(frozen=True)
class AssociationParams:
    appearance_weight: float = 0.5  # blend weight of the appearance term in ``fused``
    cue_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)  # colour, patch, contrast
    appearance_ema: float = 0.9  # weight kept on the old template when a track is updated
    appearance_gate: float = 1.0  # appearance distance above this is infeasible (1.0 = off)

    def __post_init__(self) -> None:
        if not 0.0 <= self.appearance_weight <= 1.0:
            raise ValueError("appearance_weight must be in [0, 1]")
        if not 0.0 <= self.appearance_ema < 1.0:
            raise ValueError("appearance_ema must be in [0, 1)")
        if len(self.cue_weights) != 3 or min(self.cue_weights) < 0 or sum(self.cue_weights) <= 0:
            raise ValueError("cue_weights must be three non-negative numbers with a positive sum")


def needs_frame(association_method: str) -> bool:
    return association_method in APPEARANCE_METHODS


def motion_distance(
    tracked_objects: list[TrackedObject],
    detections: list[Detection],
    track_indices,
    detection_indices,
) -> np.ndarray:
    predicted = np.array([tracked_objects[i].kalman_tracker_state.mean[:2] for i in track_indices], dtype=float)
    centres = np.array(
        [
            ((detections[j].bounding_box.x1 + detections[j].bounding_box.x2) / 2.0,
             (detections[j].bounding_box.y1 + detections[j].bounding_box.y2) / 2.0)
            for j in detection_indices
        ],
        dtype=float,
    )
    return np.linalg.norm(predicted[:, None, :] - centres[None, :, :], axis=2)


def make_cost(association_method: str, params: AssociationParams, max_distance: float, max_track_age: int):
    """Return ``cost(tracked_objects, detections, max_age, track_indices, detection_indices)``."""
    if association_method not in ASSOCIATION_METHODS or association_method == "original":
        raise ValueError(f"make_cost does not build a cost for {association_method!r}")

    if association_method == "motion":
        weight = 0.0
    elif association_method == "appearance":
        weight = 1.0
    else:
        weight = params.appearance_weight

    def cost(tracked_objects, detections, max_age, track_indices, detection_indices):
        track_indices = list(track_indices)
        detection_indices = list(detection_indices)
        distance = motion_distance(tracked_objects, detections, track_indices, detection_indices)
        result = distance.copy()
        appearance = None

        if weight > 0.0:
            appearance = appearance_distance_matrix(
                [tracked_objects[i].appearance for i in track_indices],
                [detections[j].appearance for j in detection_indices],
                params.cue_weights,
            )
            result = (1.0 - weight) * distance + weight * max_distance * appearance

        infeasible = distance > max_distance
        if appearance is not None and params.appearance_gate < 1.0:
            infeasible |= appearance > params.appearance_gate
        result[infeasible] = INFTY_COST

        # Same rule as the original second stage: tracks coasting longer than the cascade depth are dead.
        for row, track_idx in enumerate(track_indices):
            if tracked_objects[track_idx].age > max_track_age:
                result[row, :] = INFTY_COST
        return result

    return cost
