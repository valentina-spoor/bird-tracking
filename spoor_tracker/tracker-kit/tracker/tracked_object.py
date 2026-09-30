# Copied from libs/tracker/src/spoortracker/tracked_object.py at spoor-main commit
# 145fc6696241ec3ea1e1d3b930164bebf226538b (origin/main); imports only. See ../PROVENANCE.md.
from dataclasses import dataclass

from tracker.shim import DetectionNoCrop

from .kalman_filter import KalmanFilter
from .kalman_tracker_state import KalmanTrackerState
from .tracked_object_state import TrackedObjectState


class TrackedObject:
    def __init__(self, id: int, initial_detection: DetectionNoCrop, initial_kalman_tracker_state: KalmanTrackerState):
        self.__id = id
        self.__detections = [initial_detection]
        self.__kalman_tracker_state = initial_kalman_tracker_state
        self.__state = TrackedObjectState.TENTATIVE
        self.__age = 0
        self.__appearance = getattr(initial_detection, "appearance", None)

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id: int):
        self.__id = id

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state: TrackedObjectState):
        self.__state = state

    @property
    def kalman_tracker_state(self) -> KalmanTrackerState:
        return self.__kalman_tracker_state

    @property
    def detections(self) -> list[DetectionNoCrop]:
        return self.__detections

    @property
    def age(self) -> int:
        return self.__age

    @property
    def appearance(self):
        """Running (EMA) visual template of this track, or None if appearance is not in use."""
        return self.__appearance

    def update_appearance(self, new_appearance, alpha: float):
        if new_appearance is None:
            return
        if self.__appearance is None:
            self.__appearance = new_appearance
        else:
            self.__appearance = self.__appearance.blended(new_appearance, alpha)

    def increment_age(self):
        self.__age += 1

    def reset_age(self):
        self.__age = 0

    def update(self, kalman_filter: KalmanFilter, new_detection: DetectionNoCrop):
        self.__detections.append(new_detection)
        self.__kalman_tracker_state.update(kalman_filter, new_detection)
        self.reset_age()

    def predict(self, kalman_filter: KalmanFilter):
        self.__kalman_tracker_state.predict(kalman_filter)
        self.increment_age()

    def concatenate(self, other: "TrackedObject"):
        if not isinstance(other, TrackedObject):
            raise ValueError("Can only concatenate with another TrackedObject")
        if self.__detections[-1].frame_number > other.detections[0].frame_number:
            raise ValueError("The last detection of base object is older than the first detection of other object")

        self.__detections.extend(other.detections)
        self.__kalman_tracker_state = other.kalman_tracker_state
        self.__appearance = other.appearance
        self.__age = other.age

    def __add__(self, other: "TrackedObject") -> "TrackedObject":
        self.concatenate(other)
        return self

    def __repr__(self):
        return (
            f"TrackedObject(id={self.__id}, "
            f"detections_length={len(self.__detections)}, "
            f"state={self.__state}, "
            f"age={self.__age})"
        )

    def __str__(self):
        return self.__repr__()


@dataclass
class SerializableTrackedObject:
    id: int
    detections: list[DetectionNoCrop]
    state: str
    age: int


def get_tracked_objects_with_states(
    tracked_objects: list[TrackedObject], states: set[TrackedObjectState]
) -> list[TrackedObject]:
    return [t for t in tracked_objects if t.state in states]


def get_tracked_object_by_id(tracked_objects: list[TrackedObject], id: int) -> TrackedObject | None:
    return next((t for t in tracked_objects if t.id == id), None)
