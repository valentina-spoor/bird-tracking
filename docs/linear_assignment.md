# Understanding linear_assigment.py

This file has three main jobs:

    - solve the assignment problem,
    - run the matching cascade,
    - apply Kalman gating.

each done by one function.

---
---
---
### `min_cost_matching()`

Input:

    distance_metric
    max_age
    max_distance
    tracked_objects
    detections
    track_indices
    detection_indices

Output:

    matches
    unmatched_tracks
    unmatched_detections

> `linear_assignment.py` does not decide what “distance” means.

We create the cost matrix,



---
---

`max_distance` is a threshold, any values above that are converted into a little bit mora than that threshold so that they are rejected here:

`if cost_matrix[row, col] > max_distance:`

---
---
SciPy's implementation of the linear assignment problem.

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

> each track gets at most one detection \
> each detection gets at most one track

---
Very important distinction, COST FUNCTION, is the function that creates the cost matrix. Hungarian/Linear assignment is the one that assign each detection to a track.

So, first we have to use the Cost Function to create the Cost Matrix, and from there use Hungarian assignment to have each detection assigned to one track.

---
---
---

After all assignment, if we have detection without assignment, we append them into `unmatched_detections` and that can create a new track in the future.

The same happens with unmatched tracks, if they are not match, they go into `unmatched_tracks` and could survive, die, etc.

---

Even if they are matched by Hungarian, they still need to pass this:

`if cost_matrix[row, col] > max_distance:`

Basically, the cost can't be more than 100?

---
---

### `matching_cascade()`

The purpose of the cascade is:

Give more recently observed tracks priority over older tracks.

    cascade_depth = 30

because `MAX_AGE = 30`

Tracks are grouped by age:

```py
track_indices_l = [
    k for k in track_indices
    if tracked_objects[k].age == 1 + level
]
``` 
`matching_cascade()` repeatedly calls `min_cost_matching()`

So the same Hungarian logic runs repeatedly, but on smaller subsets.

---
---

### `gate_cost_matrix()`

    Take an existing cost matrix and invalidate track-detection combinations that are inconsistent with the Kalman prediction.

Basically, this does what I wrote before, when the cost is too much we just turn it into a little bit more than the threshold.

    Calculate normal association cost
            ↓
    Calculate squared Mahalanobis gating distance
            ↓
    Compare gating distance to χ² threshold (5.99)
            ↓
    If impossible:
        normal cost → INFTY_COST
            ↓
    Apply max_distance threshold
            ↓
    Hungarian assignment
            ↓
    Reject assigned pairs whose cost > max_distance

---
## Flow of STAGES

### STAGE A

    confirmed tracks
        │
        ▼
    feature distances
        │
        ▼
    raw cost matrix
        │
        ▼
    Kalman Mahalanobis gating
        │
        ▼
    gated cost matrix
        │
        ▼
    matching cascade
        │
        ├── age=1 tracks → Hungarian
        │
        ├── age=2 tracks → Hungarian
        │
        ├── age=3 tracks → Hungarian
        │
        └── ...
        │
        ▼
    matches_a
 
---
    COST:
    nearest-neighbor feature distance

    GATE:
    Kalman Mahalanobis position gate

    ASSIGNMENT:
    Hungarian + matching cascade

### STAGE B

    predicted-position cost
            ↓
    cost matrix
            ↓
    Hungarian
            ↓
    threshold rejection
            ↓
    matches_b

---

    COST:
    predicted-position Euclidean distance

    GATE:
    fixed max-distance threshold

    ASSIGNMENT:
    Hungarian

---
---
Full flow:

    run_tracker.py
    │
    │ builds detections
    ▼
    SimpleSORTTracker
    │
    │ controls lifecycle
    ▼
    matching.py
    │
    ├─────────────────────────────────────┐
    │ Stage A                             │ Stage B
    │                                     │
    │ feature distance                    │ predicted position
    │ + Kalman gate                       │ distance
    │ + cascade                           │
    │                                     │
    └───────────────┬─────────────────────┘
                    ▼
        linear_assignment.py
                    │
                    ├── build/use cost matrix
                    ├── Hungarian
                    ├── threshold matches
                    └── return matched/unmatched