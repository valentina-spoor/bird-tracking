# Understanding tracked_object.py

it represents one tracked bird and defines what happens to that track when it is updated, predicted, or concatenated.

    TrackedObject
    │
    ├── id
    ├── detections
    ├── Kalman state
    ├── lifecycle state
    └── age

Initialed:

```py
    self.__detections = [initial_detection]
    self.__state = TrackedObjectState.TENTATIVE
    self.__age = 0
``` 

STATE can be:

- TENTATIVE
- CONFIRMED
- DELETED

---
A track keeps a list of every detection that has actually been matched to it.

    Track 3 detections:

    frame 20 -> D5
    frame 21 -> D2
    frame 22 -> D4
    frame 25 -> D1
    ...

If the track survives frames 23 and 24 by Kalman prediction only, there are no new detections appended for those frames.

So:

    detections

means:

> actual observations assigned to this track

---
---

We also have Kalman State, that is the Kalman mean (position), its covariance and its feature history.

---
---

### `update()`

This happens when a track successfully matches a detection.

    new observation
        ↓
    store detection
        ↓
    correct Kalman state
        ↓
    age = 0

---

### `predict()`

    move state forward one timestep
    +
    age += 1

----

Post-processing functions and debigging functions are in this file

---
---

A `TrackedObject` is not the tracker.

It's one hypothesis about one bird.

    TrackedObject #7
    │
    ├── Identity
    │      7
    │
    ├── Lifecycle
    │      CONFIRMED
    │
    ├── Measurement history
    │      D31
    │      D35
    │      D39
    │      D42
    │
    ├── Kalman state
    │      mean
    │      covariance
    │      features
    │
    └── Time since observation
            age = 3


----
---
---
---
## Architecture after this file

    SimpleSORTTracker
            │
            ├── matching
            │
            ├── create/delete
            │
            └── calls Track.update/predict
                        │
                        ▼
                TrackedObject
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
    detection history       KalmanTrackerState
            │                     │
            ▼                     ▼
    lifecycle/age          mean + covariance