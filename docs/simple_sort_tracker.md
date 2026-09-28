# Understanding simple_sort_tracker.py

<mark> manager of the whole tracker </mark>

It coordinates:

- matching current detections to existing tracks
- updating matched tracks
- confirming tentative tracks
- deleting stale tracks
- creating new tracks
- updating the feature memory used for matching
- predicting every surviving track forward one step
---
Basic Diagram

    run_tracker.py
        │
        ▼
    SimpleSORTTracker.match_and_track()
        │
        ├── match
        ├── update
        ├── confirm
        ├── delete
        ├── create
        ├── update feature memory
        └── predict

---
---

## `class SimpleSORTTracker`

### `def __init__()`

Receives:

    euclidean_matching_threshold
    max_age
    tentative_threshold

It creates:

`self.kalman_filter = KalmanFilter()`

as to have one Kalman-filter model object

Then,

    self.tracked_objects = []
    self.deleted_tracked_objects = []

That basically means,

    tracked_objects = currently alive tracks

    deleted_tracked_objects = tracks that have ended

and, `self.id_generator = count()`

creates the IDs

---

### `def match_and_track()`

Input `match_detections_with_tracked_objects()`:

    existing tracks
    +
    detections from current frame

Output:

    matched pairs
    unmatched tracks
    unmatched detections

Then, for every matched pair:

1. saves the new detection in the track history
2. uses the detection as a Kalman measurement correction

---

### Three FOR LOOPS:

####  * Matched pairs, separate them into CONFIRMED and TENTATIVE

Then, if the track has a CONFIRMED state nothing happens in this for loop, BUT if it's TENTATIVE, it compares the length of the detections of this object with the threshold of 3 we proposed earlier.

#### * Not matched tracked objects

If it was confirmed but without a match and its age is < 30, they are kept. If they were not confirmed or have a higher age than 30, they have DELETED state and go to `deleted_tracked_objects`

`deleted_tracked_objects` is useful for:

    post-processing
    track concatenation
    later analysis

> If a TENTATIVE track misses a frame, it's deleted!!!

#### * Not matched detections

These create new tracks. We assign a new ID to them. New Kalman track is born

Basically this:

    Detection
        │
        ▼
    new track ID
        │
        ▼
    KalmanFilter.initiate()
        │
        ▼
    KalmanTrackerState
        │
        ▼
    TrackedObject
        │
        ▼
    state = TENTATIVE


---
---
#### Feature PART

We first pair each feature with its track ID in two arrays:

    features = []
    features_ids = []

These features are basically `position + box area` not anything related to appearance. 

<mark> This is why feature detection needs some care later!!!

---
Then we update the stored history of feature vectors for active confirmed tracks, 

    track 4:
    remember all past features associated with ID 4

    track 8:
    remember all past features associated with ID 8

Before `match_and_track()` goes to the next iteration, it predicts the Kalman state for the next frame is already done.

```py
for tracked_object in self.tracked_objects:
    tracked_object.predict(self.kalman_filter)
```

---
---

### Summary of `match_and_track()`
First, match current detections to active tracks. Update matched tracks. Promote tentative tracks that have enough successful detections. Keep unmatched confirmed tracks alive until they become too old, but delete unmatched tentative tracks immediately. Create new tentative tracks for unmatched detections. Update the feature memory for confirmed tracks. Finally, predict every surviving track one step forward for the next frame.

---
---
---
---

#### `filter_deleted_tracks()`
Applies filters to deleted tracks after tracking.

The standalone runner does not call this.

---
#### `concatenate_tracks()` 
This is post-processing.

It attempts to stitch two tracks together if:

- one ends,
- another begins later,
- they are close enough in time,
- and close enough spatially.

run_tracker.py deliberately **does not** use this post-processing

---
#### `finalize_tracking()`

At video end, every remaining active track is marked DELETED and moved to `deleted_tracked_objects`.

Cleanup

---

We now have something like this:

    run_tracker.py
        │
        │ detections for frame t
        ▼
    SimpleSORTTracker.match_and_track()
        │
        ├── MATCH
        │
        ├── UPDATE matched tracks
        │
        ├── DELETE stale/unconfirmed tracks
        │
        ├── CREATE tracks for unmatched detections
        │
        ├── UPDATE feature memory
        │
        └── PREDICT frame t+1