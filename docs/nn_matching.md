# Understanding nn_matching.py

This file answers:

- For a confirmed track with many past features, how do we calculate the cost of matching a new detection to that track?

- How close is this new detection's feature to anything I've previously seen for that track?

#### Three parts:

    raw distance between vectors
            ↓
    nearest-neighbor distance
            ↓
    track-specific memory

---
---

### `_pdist(a, b)`

```py

closest_track, index = spatial.KDTree(a).query(b)
return closest_track

```

So,

    a = stored feature samples
    b = query feature samples
    KDTree finds the nearest point in a for each point in b

Euclidean distance

---
---

### `_nn_euclidean_distance(x, y)`

```py
distances = _pdist(x, y)
return np.maximum(0.0, distances.min(axis=0))
```

A current detection might not be close to the oldest sample, but could still be close to one of the more recent samples.

Using:

$$ \min_k $$

means:

If this detection resembles any past sample of the track, give it a low cost.

That gives the track some tolerance to change over time.

---
---

There is a `_nn_cosine_distance(x, y)` function but is not used, could be helpful when features are things like learned appearance embeddings.

---

### Class `NearestNeighborDistanceMetric`

We have:

    self.samples = {}

That stores tracks like this:

    {
        track_id_0: [f0, f1, f2, ...],
        track_id_4: [f0, f1, ...],
        track_id_9: [f0, f1, f2, f3, ...],
    }

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

---
---

### `partial_fit()`

Receives tracks and their features, it sotres them in `self.samples{}`

Also, at the end it deletes feature memory for identities that are no longer active.

---
---

### `distance(features, targets)`

It creates:

```py
cost_matrix = np.zeros(
    (len(targets), len(features))
)
```
So that it stores tracks with their detections, and then it 

    compares every current detection feature against the entire stored history of this track.

---
---

Because the feature is:

$$ (x,y,\text{area}) $$

these three dimensions are mixed directly in Euclidean distance.

<mark> The feature dimensions have different numerical scales.

----
---
---

STAGE A

    For Track \(i\) and Detection \(j\):

    Feature cost
$$ C_{ij} = \min_k \left\| f_j-f_{i,k} \right\|_2 $$

    with:

$$ f=(x,y,\text{area}) $$

    Then:

    Kalman gating

    If:

$$ d^2_{\text{Mahalanobis}} > 5.9915 $$

    for position-only 2D gating, then:

$$ C_{ij}\leftarrow\text{INFTY\_COST} $$

    Then the cascade/Hungarian logic handles assignment.

    That is the whole Stage A association mechanism.

---
---
---
---
## Where do we use this file?

    matching.py
    │
    │ Stage A
    ▼
    nn_matching.py
    │
    ├── track feature history
    ├── nearest-neighbor Euclidean distance
    └── build cost matrix
            │
            ▼
    linear_assignment.gate_cost_matrix()
            │
            ▼
    matching_cascade()
            │
            ▼
    Hungarian