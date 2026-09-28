# Copied from libs/tracker/src/spoortracker/kalman_tracker_state.py at spoor-main commit
# 145fc6696241ec3ea1e1d3b930164bebf226538b (origin/main); imports only. See ../PROVENANCE.md.
# vim: expandtab:ts=4:sw=4
from typing import List

from tracker.kalman_filter import KalmanFilter
from tracker.shim import Detection, xyxy_to_cxcyah


class KalmanTrackerState:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.

    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    state : TrackState
        The current track state.

    """

    def __init__(
        self,
        mean,
        covariance,
        max_age,
        features: List = None,
    ):
        self.mean = mean
        self.covariance = covariance
        self.features = features
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kalman_filter: KalmanFilter):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kalman_filter : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kalman_filter.predict(self.mean, self.covariance)

    def update(self, kalman_filter: KalmanFilter, detection: Detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kalman_filter : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kalman_filter.update(
            self.mean, self.covariance, xyxy_to_cxcyah(detection.bounding_box).values
        )
        if self.features is None:
            self.features = []

        self.features.append(detection.feature)
