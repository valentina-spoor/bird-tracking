# Understanding distance_matching.py

This file answers:

    For the tracks that were not matched in Stage A, which current detection is closest to the track’s Kalman-predicted position?

The logic is basically:

    predicted box position → distance to detections → cost matrix

---
---

### `closest_track()`

It compares the top-left corner of the predicted track box against the top-left corners of the candidate detection boxes.

    It is literally top-left-corner Euclidean distance.

---
---

### `closest_track_cost()`

This function builds the Stage B cost matrix.

So again:

rows = tracks
columns = detections

---
So if a track is older than the allowed age, every detection gets:

$$ 100000 $$

as its cost.

<mark> Though in practice, SimpleSORTTracker is also managing track deletion separately.

Stage B compares detections against predicted positions

---
---

Important:

`distance_matching.py` itself only builds the distance matrix.

It does not reject a detection based on the Euclidean threshold.

That happens later inside:

`min_cost_matching(...)`

---
---

### STAGE B PIPELINE

    predicted TL position
            ↓
    Euclidean distance matrix (THIS FILE)
            ↓
    Hungarian
            ↓
    max-distance rejection

---
---

### NOTES

Stage B uses top-left corner distance:

$$ (x_{\text{left}},y_{\text{top}}) $$

not box-center distance.

For similarly sized boxes, that may not matter much.

But if box size changes, the top-left corner can shift even if the physical bird center stays almost the same.

---

The function:

`closest_track(...)`

builds:

`KDTree([bbox_tl])`

with only one point.

That's technically valid, but it means KDTree is unnecessary for this specific calculation; ordinary vectorized Euclidean distance could do it directly.

That is more of an implementation observation than a tracking problem.

---
---
---
---

HEART OF THE TRACKER

                      DETECTIONS
                           │
                           ▼
                      matching.py
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
        STAGE A                       STAGE B
    confirmed tracks          tentative + fallback
            │                             │
            ▼                             ▼
    nn_matching.py              distance_matching.py
            │                             │
    historical feature             predicted-position
        distance                     distance
            │                             │
            ▼                             │
    Mahalanobis gate                     │
            │                             │
            ▼                             ▼
    matching cascade            min_cost_matching
            │                             │
            └──────────────┬──────────────┘
                           ▼
                    linear_assignment.py
                            │
                            ▼
                        Hungarian
                            │
                            ▼
            matches / unmatched tracks /
                unmatched detections