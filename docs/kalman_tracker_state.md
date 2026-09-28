# Understanding kalman_tracker_state.py

This file is the wrapper around the Kalman state for one track. It does not contain the Kalman equations themselves; those are in `kalman_filter.py`.

---

Its job is to hold:

- the current Kalman mean,
- the covariance,
- the stored features,
- and provide simple predict() / update() methods for one tracked object.

---

### `to_tlwh()`

This converts the Kalman state into a normal bounding box:

$$ (x_{left},y_{top},w,h) $$

---

### `to_tlbr()`

So:

$$ [x,y,w,h] $$

becomes:

$$ [x_1,y_1,x_2,y_2]$$

---

### `predict()`

It just says:

    Here is my current state. Advance it one timestep.

The actual math is delegated to:

`KalmanFilter.predict(...)`

---

### `update()`

Perform Kalman filter measurement update step and update the feature cache.

---
---

Two kinds of STATES:

`TrackedObject.state`

means:

    lifecycle status

while:

`KalmanTrackerState.mean`

means:

    estimated physical/box state

---
---
---

Summary:

    store mean/covariance/features
    convert state to box formats
    delegate predict
    delegate update

----
---
---
---

## Where it fits in the whole tracker

    TrackedObject
        │
        ├── lifecycle state
        ├── detection history
        ├── age
        │
        └── KalmanTrackerState
                │
                ├── mean
                ├── covariance
                ├── feature history
                │
                ├── predict()
                │      ↓
                │   KalmanFilter.predict()
                │
                └── update()
                        ↓
                    KalmanFilter.update()