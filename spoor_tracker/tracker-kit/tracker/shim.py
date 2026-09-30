"""Local stand-ins for the `spoorcontainers` bounding-box and detection types the copied
tracker files import.

The production tracker files (see PROVENANCE.md) are written against
`spoorcontainers.bounding_box.{BoundingBoxXYXY, BoundingBoxXYWH, BoundingBoxCXCYAH}`,
`spoorcontainers.bounding_box_transforms.{xyxy_to_cxcyah, xyxy_to_xywh, xywh_to_xyxy}`,
`spoorcontainers.bounding_box_operations.distance_between_two_bounding_boxes`, and
`spoorcontainers.detection.{Detection, DetectionNoCrop}`. `spoorcontainers` itself is a small,
dependency-light package (only `numpy`), unlike the edge tracker's `spoorflow`/`spooredge`
dependencies -- but it is still a monorepo-internal package that would need an editable `uv`
workspace install, defeating "two students `pip install -r requirements.txt` and run it
anywhere." This module reproduces only the surface the nine copied files actually touch, one
property or function at a time, verified by grepping their usage rather than assumed.

This kit uses `DetectionNoCrop`, never `Detection`: the tracker never reads `.crop` (grep
`libs/tracker/src/spoortracker/` for `\\.crop` finds nothing), and this kit never has an image
crop to hand it, so `DetectionNoCrop`'s absence of that field is a correct fit, not a shortcut.

Field-for-field: `bounding_box` and `frame_number` are real and load-bearing (frame_number
orders `concatenate_tracks`'s candidate search and the deleted-track sort; bounding_box drives
every Kalman predict/update and every association cost). `feature` is *also* load-bearing --
unlike the edge tracker's inert `confidence`/`class_id`/`class_name`, this tracker's confirmed-
track matching stage (`matching.py`'s `gated_metric`) computes a nearest-neighbor distance
directly over `feature`, so it cannot be defaulted to a constant; `run_tracker.py` derives it
from the detection's own box exactly as `tracker_filter.py` does. `frame_timestamp`,
`is_predicted_by_tracker`, `video_name_id`, `frame_timestamp_unit` and `frame_rate` are carried
on every `DetectionNoCrop` the tracker touches but never read by any of the nine copied files
(same grep), so they are defaulted freely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class BoundingBoxXYXY:
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float
    is_normalized: bool = False

    def __post_init__(self) -> None:
        if self.x1 > self.x2:
            logging.warning(f"x1 {self.x1} is greater than x2 {self.x2}")
        if self.y1 > self.y2:
            logging.warning(f"y1 {self.y1} is greater than y2 {self.y2}")

    @property
    def width(self) -> int | float:
        return self.x2 - self.x1

    @property
    def height(self) -> int | float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class BoundingBoxXYWH:
    x: int | float
    y: int | float
    width: int | float
    height: int | float
    is_normalized: bool = False

    @property
    def values(self) -> list[int | float]:
        return [self.x, self.y, self.width, self.height]

    @property
    def area(self) -> int | float:
        return self.width * self.height


@dataclass(frozen=True)
class BoundingBoxCXCYAH:
    center_x: int | float
    center_y: int | float
    aspect_ratio: float
    height: int | float
    is_normalized: bool = False

    @property
    def values(self) -> list[int | float]:
        return [self.center_x, self.center_y, self.aspect_ratio, self.height]


def xyxy_center(bounding_box: BoundingBoxXYXY) -> tuple[float, float]:
    return (bounding_box.x1 + bounding_box.x2) / 2, (bounding_box.y1 + bounding_box.y2) / 2


def xyxy_to_cxcyah(bounding_box: BoundingBoxXYXY) -> BoundingBoxCXCYAH:
    center_x, center_y = xyxy_center(bounding_box)
    aspect_ratio = bounding_box.width / bounding_box.height
    return BoundingBoxCXCYAH(center_x, center_y, aspect_ratio, bounding_box.height, bounding_box.is_normalized)


def xyxy_to_xywh(bounding_box: BoundingBoxXYXY) -> BoundingBoxXYWH:
    width = bounding_box.x2 - bounding_box.x1
    height = bounding_box.y2 - bounding_box.y1
    return BoundingBoxXYWH(bounding_box.x1, bounding_box.y1, width, height, is_normalized=bounding_box.is_normalized)


def xywh_to_xyxy(bounding_box: BoundingBoxXYWH) -> BoundingBoxXYXY:
    x2 = bounding_box.x + bounding_box.width
    y2 = bounding_box.y + bounding_box.height
    return BoundingBoxXYXY(bounding_box.x, bounding_box.y, x2, y2, is_normalized=bounding_box.is_normalized)


def distance_between_two_bounding_boxes(bounding_box1: BoundingBoxXYXY, bounding_box2: BoundingBoxXYXY) -> float:
    (x1, y1), (x2, y2) = xyxy_center(bounding_box1), xyxy_center(bounding_box2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class FrameTimestampUnit(str, Enum):
    SECOND = "second"
    MILLISECOND = "millisecond"
    MICROSECOND = "microsecond"
    NANOSECOND = "nanosecond"


@dataclass
class DetectionNoCrop:
    bounding_box: BoundingBoxXYXY
    frame_number: int
    frame_timestamp: float | int
    feature: tuple[float | int, ...]
    is_predicted_by_tracker: bool = False
    video_name_id: Optional[str] = None
    frame_timestamp_unit: Optional[FrameTimestampUnit] = None
    frame_rate: int | None = None
    # Visual descriptor (tracker.appearance.Appearance), filled in by the tracker when the
    # association method needs it. Not part of the production DetectionNoCrop.
    appearance: Optional[object] = None


# The nine copied files reference `Detection` only as a type hint (grep confirms none of them
# ever construct one) -- production's `Detection` is `DetectionNoCrop` plus an optional `crop`
# field this kit never populates, so the alias is exact for every attribute those hints stand
# in for.
Detection = DetectionNoCrop
