### Kalman state through one frame

For each tracked object, the tracker keeps one evolving Kalman state. In this implementation, that state is mainly stored as:

```python
self.mean
self.covariance
```

The `mean` contains the estimated object state:

$$
[x,\ y,\ a,\ h,\ v_x,\ v_y,\ v_a,\ v_h]
$$

where \(x,y\) are the bounding-box center, \(a\) is aspect ratio, \(h\) is height, and the remaining values are their estimated velocities. The `covariance` represents how uncertain the filter is about those estimates.

For a frame \(t\), the Kalman state goes through three conceptual stages:

$$
\hat{x}_{t|t-1}
\rightarrow
\hat{x}_{t|t}
\rightarrow
\hat{x}_{t+1|t}
$$

1. **Prediction for frame \(t\)**
   Information from frame \(t-1\) is used to predict where the object should be in frame \(t\):

$$
\hat{x}_{t|t-1}
$$

2. **Correction using the detection at frame \(t\)**
   The detector provides a measurement \(z_t\). The Kalman filter combines the prediction with this measurement:

$$
\hat{x}_{t|t-1} + z_t
\rightarrow
\hat{x}_{t|t}
$$

This is not redundant because the detector measurement is assumed to contain noise. For example, if the prediction says \(x=100\) and the detector says \(x=106\), the corrected estimate could be something like \(x=103\), depending on the uncertainty of both sources.

3. **Prediction for frame \(t+1\)**
   After processing frame \(t\), the tracker predicts forward again:

$$
\hat{x}_{t|t}
\rightarrow
\hat{x}_{t+1|t}
$$

In Spoor's implementation, this last prediction happens at the end of `match_and_track()`. Therefore, after that function returns, the stored Kalman state is already the prediction for the next frame.

This ordering is not unusual mathematically; it is mainly an implementation choice. Another tracker could instead call `predict()` at the beginning of the next frame. The important thing is to know what the stored state represents at each point in the code.

The same `mean` and `covariance` variables are repeatedly replaced as the track evolves:

```text
prediction for t
      ↓
correction using detection t
      ↓
prediction for t+1
      ↓
correction using detection t+1
      ↓
...
```

So when we talk about several "Kalman states," we are usually referring to different moments of the same continuously updated state, not completely separate variables.
