# Understanding matching.py

MOST IMPORTANT PART FOR **association**

Given active tracks and detections for the current frame, decide which track gets which detection.

#### Inputs

    tracked_objects
    detections
    kalman_filter
    nearest_neighbor_metric

#### Outputs

    matches
    unmatched_tracks
    unmatched_detections

---
---

First, it splits tracks into:

    confirmed
    unconfirmed

Once that is done, we do this with each of them:

    all active tracks
        │
        ├── confirmed ───────────────┐
        │                            ▼
        │                   matching cascade
        │                   feature distance
        │                   + Kalman gate
        │                            │
        │                            ▼
        │                     matches_a
        │
        |
        └── tentative + unmatched confirmed
                                    │
                                    ▼
                            second matching stage
                            predicted-position distance
                                    │
                                    ▼
                                    matches_b

Then, 
    
    matches = matches_a + matches_b

---
---
---
### Functions

#### `gated_metric()` 

computes the cost matrix used for confirmed tracks

For every candidate detectios it extracts:

```py
features = np.array([dets[i].feature ...])
```
not visual appearance features!!!

Then,

    feature_corresponding_ids = ...

gets the IDs of the tracks being considered.

And,
```py 
cost_matrix =
    nearest_neighbor_metric.distance(
        features,
        feature_corresponding_ids
    )
```

$$

C = \begin{bmatrix}
  c_{11} & c_{12} & c_{13}\\
  c_{21} & c_{22} & c_{23}
\end{bmatrix}

$$

where:
- row = track
- column = detection
- value = feature-space cost

---
---

Once we get the cost matrix, before we do the actual matching, we have to do Kalman Gating, that basically removes options where the Kalman prediction box is too far. 

>Even if a detection looks cheap in feature space, it can still be rejected if it lies too far outside the Kalman-predicted position.

So:

    feature cost says:
    "this could be Track 4"

    Kalman says:
    "physically impossible"

    => reject

<mark> This is a gated cost matrix.

---
---

Confirmed tracks are passed to:

```py
linear_assignment.matching_cascade(...)
```
This is more sophisticated than one giant Hungarian match.

The cascade tries recent tracks first.

> Only `confirmed_tracks = []`go into the first matching cascade

---
Stage A returns,

    matches_a
    unmatched_tracks_a
    unmatched_detections

Then,

    iou_track_candidates =
        unconfirmed_tracks
        + [
            unmatched confirmed tracks
            that are still young enough
        ]

<mark> This stage does not actually use IoU.

---

Stage B

Here all unconfirmed tracks and tracks that Stage A failed to match enter

So basically,

    Stage A:
    Can I match this confirmed track using historical feature similarity
    while staying Kalman-plausible?

    If no:

    Stage B:
    Can I simply match it to the closest predicted-position detection?

> There is special treatment of tracks with age == 1!!!!

> revisit this when we study linear_assignment.py

Stage B create a second cost matrix that does, 

    predicted track position
    vs
    detection position

---
---
Finally, we return matches made in a + b and unmatched tracks

---
---
---
Why two matching stages?

    confirmed
    → use memory + Kalman plausibility

    tentative
    → use predicted spatial proximity

---
---
This file does NOT:

- compute the final Hungarian assignment itself
- compute Mahalanobis distance directly
- compute nearest-neighbor distance directly
- update tracks
- create/delete tracks

---
---
---
Final architecture until now

    run_tracker.py
        │
        ▼
    SimpleSORTTracker
        │
        ▼
    matching.py
        │
        ├── Stage A
        │    feature cost
        │    + Kalman gating
        │    + matching cascade
        │
        └── Stage B
            predicted-position cost
            + min-cost assignment