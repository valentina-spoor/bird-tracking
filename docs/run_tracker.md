# Understanding run_tracker.py

#### Inputs:
+ video
+ detections CSV
+ output video path
+ output tracks CSV path

#### CSV content
+ frame_number
+ frame_timestamp
+ x
+ y
+ w
+ h
+ area
+ classifier_name
---
Every frame must call:

    tracker.match_and_track(detections)

exactly once

The Kalman filter inside this tracker uses:

$$ dt = 1 $$

and has no elapsed-time input.

<mark>They deliberately make sure skipped prediction steps do NOT happen.</mark>

---
We have dependencies

```py
from tracker.shim import BoundingBoxXYWH, DetectionNoCrop, xywh_to_xyxy
from tracker.simple_sort_tracker import SimpleSORTTracker
from tracker.tracked_object_state import TrackedObjectState
```

---
config constants
```py
REFERENCE_WIDTH_PX = 7680
REFERENCE_EUCLIDEAN_MATCHING_THRESHOLD = 250.0
DEFAULT_MAX_AGE = 30
DEFAULT_TENTATIVE_THRESHOLD = 3
```

250 = maximum matching-distance threshold \
30 = how many missed frames a confirmed track can survive \
3 = how many matches a new track needs before becoming confirmed

___
### We start with FUNCTIONS
---

```py
def annotation_scale(frame_width: int) -> tuple[int, int, float]
```
Visualization only - CHANGES PRESENTATION

---
---

```py
def load_detections_by_frame(csv_path: Path) -> dict[int, list[BoundingBoxXYWH]]
```

It reads every row from the CSV and groups boxes by frame number. 

It creates something like this with the frames:
```py
{
    0: [box_A, box_B],
    1: [box_A],
    3: [box_A]
}
```

So,
```py
boxes = detections_by_frame.get(frame_number, [])
```
means that if there are detections use them and if there aren't use []

And,

<mark> **missing CSV frame means: zero detections on this frame** </mark>

So,

    CSV
     │
     ▼
    XYWH

---
---

```py
def build_detections(frame_number: int, frame_timestamp: int, boxes: list[BoundingBoxXYWH]) -> list[DetectionNoCrop]
```
It turns the detections into Tracking boxes, basically they contain the bounding box, frame_number, frame_timestamp, feature, etc.

Also, we convert the bounding box from $(x,y,w,h)$ to $(x_1,y_1,x_2,y_2)$ so as to have 2 coordinates

Here we establish that $ feature[x,y,area]$, then we do nearest-neighbor matching to associate current detections with previous tracks.

---
---

```py
def _track_color(track_id: int) -> tuple[int, int, int]
```
With the track ID makes a RGB color for it. This is useful so the same ID keeps the same track color.

---
---

```py
def check_frame_alignment(
    detections_by_frame: dict[int, list[BoundingBoxXYWH]],
    frame_count: int,
    video_path: Path,
    detections_path: Path,
) -> None
```

Make sure the detections CSV belongs to exactly this video.

Basically, align frames

___
---
---
---

```py
def run(
    video_path: Path,
    detections_path: Path,
    out_video_path: Path,
    out_tracks_path: Path,
    euclidean_matching_threshold: float,
    max_age: int,
    tentative_threshold: int,
) -> int
```

Central runner function

**You create the tracker once:**

`tracker = SimpleSORTTracker()`

Then reuse that same object for every frame.

---

The runner opens the video and obtains:
    
    width
    height
    fps
    frame_count

The tracking algorithm is driven by the detection objects.

---
### Frame Loop

    VIDEO FRAME 37
        │
        ├───────────────┐
        │               │
        ▼               ▼
    get image      get CSV boxes
                        │
                        ▼
                build_detections()
                        │
                        ▼
                list[DetectionNoCrop]
                        │
                        ▼
            tracker.match_and_track()

---
Now for every tracked object that is confirmed, we do:

1. It displays the last detected box, **NOT** the Kalman Prediction. `internal tracker state!=rendered output box`
2. `trails[tracked_object.id]` Visualization of the trail of the track
3. It stores, to later add to the track CSV:
    
```py
{
    "frame_number": frame_number,
    "track_id": tracked_object.id,
    "x": box.x1,
    "y": box.y1,
    "w": box.width,
    "h": box.height,
}
``` 

> So if a track remains alive but gets no new detection:

>>frame 20 -> box A \
frame 21 -> still box A \
frame 22 -> still box A \
frame 23 -> still box A

> even while the internal Kalman prediction may be elsewhere.

<mark> This is why evaluation needs some care later.

---
Finally,

```py
writer.write(frame)
frame_number += 1
```

So, the whole loop:

    read frame
        ↓
    fetch detections
        ↓
    build Detection objects
        ↓
    match_and_track()
        ↓
    inspect confirmed tracks
        ↓
       draw
        ↓
    save track rows
        ↓
    next frame


---

`finalize_tracking()` moves all remaining active tracks to the deleted-track list.

---

### `main()`

simply command-line parsing.

You should always have:

    python run_tracker.py \
    --video ... \
    --detections ... \
    --out ... \
    --tracks ...

But optionally change:

    euclidean_matching_threshold
    max_age
    tentative_threshold

---
---
---
---

Diagram for the whole file

                    run_tracker.py

        detections.csv                 video.mp4
                │                           │
                ▼                           ▼
    load_detections_by_frame()        OpenCV read frame
                │                           │
                └────────────┬──────────────┘
                            │
                            ▼
                    current frame number
                            │
                            ▼
                    boxes for frame t
                            │
                            ▼
                    build_detections()
                            │
            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼
    bounding box XYXY              feature = (x,y,area)
            │                                 │
            └────────────────┬────────────────┘
                            ▼
                    DetectionNoCrop[]
                            │
                            ▼
                SimpleSORTTracker
                .match_and_track()
                            │
                            ▼
                    tracked_objects
                            │
                confirmed only
                            │
                ┌──────────┴──────────┐
                ▼                     ▼
            draw video           tracks CSV


---
### It does **NOT** decide: 

- which detection belongs to which track,
- what the cost matrix is,
- which Hungarian assignment wins,
- how Kalman prediction works,
- when exactly a track is deleted,
- how gating works

---
---
## Summary

It is an adapter and execution loop: it converts detector CSV rows into tracker detection objects, advances the tracker once per video frame, and exports the confirmed tracking result.