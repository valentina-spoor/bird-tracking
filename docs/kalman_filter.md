# Understanding kalman_filter.py

1. state vector and matrices,
1. initialization,
1. prediction,
1. projection,
1. update,
1. Mahalanobis gating

---
---

## State Vector and Matrices

8-dimensional state:

$$ \mathbf{x} = [x,\ y,\ a,\ h,\ v_x,\ v_y,\ v_a,\ v_h]^T $$

where:

- $x,y$: bounding-box center,
- $a$: aspect ratio,
- $h$: box height,
- $v_x,v_y,v_a,v_h$: their velocities.

---
### Motion Matrix

The code creates:

`self._motion_mat = np.eye(2 * ndim, 2 * ndim)`

so initially this is an $8\times8$ identity matrix.

Then:

```py
for i in range(ndim):
    self._motion_mat[i, ndim + i] = dt
```

This creates the state transition matrix:

$$ F= \begin{bmatrix} 1&0&0&0&1&0&0&0\\ 0&1&0&0&0&1&0&0\\ 0&0&1&0&0&0&1&0\\ 0&0&0&1&0&0&0&1\\ 0&0&0&0&1&0&0&0\\ 0&0&0&0&0&1&0&0\\ 0&0&0&0&0&0&1&0\\ 0&0&0&0&0&0&0&1 \end{bmatrix} $$

because $dt=1$.

Multiplying:

$$ \mathbf{x}_{t+1}=F\mathbf{x}_t $$

gives:

$$ x_{t+1}=x_t+v_x $$ $$ y_{t+1}=y_t+v_y $$ $$ a_{t+1}=a_t+v_a $$ $$ h_{t+1}=h_t+v_h $$

and velocities stay the same:

$$ v_{x,t+1}=v_{x,t} $$

etc.

That is the constant-velocity assumption.

---
### Observation/update Matrix

The constructor also creates:

`self._update_mat = np.eye(ndim, 2 * ndim)`

This is a $4\times8$ matrix:

$$ H= \begin{bmatrix} 1&0&0&0&0&0&0&0\\ 0&1&0&0&0&0&0&0\\ 0&0&1&0&0&0&0&0\\ 0&0&0&1&0&0&0&0 \end{bmatrix} $$

Its job is:

    take the 8D hidden state and extract the 4D part that can actually be measured.

---

### Uncertainty weights

```py
self._std_weight_position = 1.0 / 20
self._std_weight_velocity = 1.0 / 160
```

This means:

    larger boxes get larger absolute uncertainty.

That choice is important and potentially tunable later, but for now just understand that uncertainty depends on object scale.

----
---
---

## Initialization

he initial state is:

$$\mathbf{x}_0=[x,y,a,h,0,0,0,0]^T$$

That makes sense:

    on the first detection, the tracker knows where the bird is, but has no evidence yet about its velocity.

So velocity starts at zero.

initially, the code assumes no cross-correlation between state variables.

For example, uncertainty in $x$ is treated separately from uncertainty in $v_x$.

---
---
---

## Prediction

The function takes:

    mean
    covariance

and returns predicted versions.

If we used:

$$ \mathbf{x}_{t+1}=F\mathbf{x}_t $$

with zero process uncertainty, the filter would become unrealistically confident.

So the prediction step adds uncertainty.

---

This line:

mean = np.dot(self._motion_mat, mean)

is simply:

$$ \mu_t^- = F\mu_{t-1} $$

The minus superscript often means:

predicted before measurement correction.

---

### Covariance prediction

This is one of the fundamental Kalman equations:

$$ P_t^- = F P_{t-1}F^T + Q $$

This equation means:

1. propagate old uncertainty through the motion model,
1. add new uncertainty because the model isn't perfect.

---

Suppose the bird disappears.

Each frame:

$$ P_{t+1}^- = F P_t F^T + Q $$

and you're repeatedly adding:

$$ Q $$

So uncertainty generally grows.

That means after many missed frames:

    prediction center exists
    but confidence around it is larger

This later affects the Mahalanobis gate.

That is the connection:

    missed detections
        ↓
    repeated prediction
        ↓
    larger covariance
        ↓
    wider plausible 
    

---

The Kalman prediction actually answers two questions:

Predicted mean
$$ \mu^- $$

    Where do I expect it to be?

Predicted covariance
$$ P^- $$

    How uncertain am I about that prediction?

----

Uncertainty wighted with box height:

The implicit assumption is:

    larger objects can reasonably have larger absolute pixel displacement/error.

Again, this comes from the source code; we're not judging whether it's ideal yet.

---
---

## Till now:

    previous track state
        │
        ▼
    Kalman predict()
        │
        ├── predicted mean
        └── predicted covariance
                │
                ▼
        matching on next frame

----
---
---

## Projection

The Kalman filter stores:

$$ \mu= [x,y,a,h,v_x,v_y,v_a,v_h]^T $$

but a detection only gives:

$$ z= [x,y,a,h]^T $$

So before comparing prediction and detection, the filter projects the 8D state down to 4D.

---
---

## Update

Now we get the function that combines:

$$ \text{prediction} + \text{measurement} $$

The method is:

`def update(self, mean, covariance, measurement):`

This is the part that explains the thing we discussed before:

    If the detector says 12, why doesn't the Kalman filter just use 12?

Because it combines the detector with the prior prediction according to uncertainty.

---

The Kalman gain controls:

    How much should I move my prediction toward the measurement?

### If prediction uncertainty is high
$$ K \text{ tends to be larger} $$

so the filter trusts the measurement more.

### If prediction uncertainty is low
$$ K \text{ tends to be smaller} $$

so the filter trusts its model more.

---

### Innovation

`innovation = measurement - projected_mean`

Mathematically:

$$ y = z-\hat z $$

This is usually called the innovation or residual.

It answers:

    How wrong was my prediction compared with what I actually observed?

---

The code:

`new_mean = mean + np.dot(innovation, kalman_gain.T)`

is mathematically:

$$ \mu = \mu^- + Ky $$

This is the famous Kalman update equation.

---

### Example

Why not choose 112 if the detector observed 112?

Because the detector is not assumed perfect.

The filter is saying:

    The measurement says 112, but I also had prior evidence supporting 100. Based on how uncertain both sources are, my best estimate is somewhere between them.

If the filter had very high prior uncertainty, perhaps:

$$ K\approx0.95 $$

and then:

$$ new\ estimate\approx111.4 $$

Much closer to the detection.

If the prior were very confident:

$$ K\approx0.2 $$

then:

$$ new\ estimate\approx102.4 $$

So the result depends on uncertainty.

!!!

If the model repeatedly underpredicts movement, the filter can infer:

    Maybe velocity should be larger.

This is how the hidden velocity gets learned even though the detector never measures velocity directly.

---

### Covariance correction

Then:

```py
new_covariance =
    covariance
    - np.linalg.multi_dot(
        (kalman_gain, projected_cov, kalman_gain.T)
    )
```

Conceptually:

$$ P = P^- - KSK^T $$

The important interpretation is:

    After receiving a measurement, uncertainty usually decreases.

### Loop

    predict
    → uncertainty grows

    measure/update
    → uncertainty shrinks

    predict
    → uncertainty grows

    measure/update
    → uncertainty shrinks
---
---
---

## Kalman Cycle

    corrected state at frame t
            │
            ▼
        PREDICT
            │
            ├── mean moves
            └── uncertainty grows
            │
            ▼
    predicted state for frame t+1
            │
            ▼
    get detection
            │
            ▼
        PROJECT
            │
            ▼
    calculate innovation
            │
            ▼
    calculate Kalman gain
            │
            ▼
        UPDATE
            │
            ├── state moves toward measurement
            └── uncertainty shrinks

---
---
---

## Gating

Returns the squared Mahalanobis distance from the predicted track to every candidate measurement.

This is what we kept calling the "Kalman cost" informally.

More precisely, it is a Kalman-based gating distance.

----
---
---
---

Final Flow:

    Detection CSV
    ↓
    run_tracker.py
    ↓
    DetectionNoCrop
    ↓
    SimpleSORTTracker
    ↓
    matching.py
    ↓
    Stage A:
    nn_matching
    ↓
    feature cost matrix
    ↓
    Kalman gating
    │
    └─ kalman_filter.gating_distance()
            ↓
        Mahalanobis distance
    ↓
    matching cascade
    ↓
    Hungarian

And,

    Stage B:
    Kalman predicted state
    ↓
    to_tlwh()
    ↓
    distance_matching.py
    ↓
    Euclidean top-left distance
    ↓
    Hungarian

----
----
----

## The Kalman filter in five equations

### Prediction
$$ \mu^-_t = F\mu_{t-1} $$
### Prediction uncertainty
$$ P^-_t = FP_{t-1}F^T+Q $$
### Innovation
$$ y_t = z_t-H\mu^-_t $$
### Kalman gain
$$ K_t = P^-_tH^T (HP^-_tH^T+R)^{-1} $$
### Correction
$$ \mu_t = \mu^-_t + K_ty_t $$

And for gating:

$$ d_M^2 = (z-\hat z)^T S^{-1} (z-\hat z) $$
