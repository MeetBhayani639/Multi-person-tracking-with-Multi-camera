"""
kalman.py

Compact Kalman filter for bounding boxes (cx,cy,s,r) with velocity components.
Provides predict(), update(), gating (mahalanobis distance) utilities.

State vector x = [cx, cy, s, r, vx, vy, vs]^T
Measurement z = [cx, cy, s, r]^T
"""
import numpy as np
from scipy.stats import chi2

CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}  # useful table


class KalmanFilterBox:
    def __init__(self):
        # state dim 7, measurement dim 4
        self._dim_x = 7
        self._dim_z = 4

        # create matrices
        self._F = np.eye(self._dim_x)
        for i in range(3):  # allow cx,cy,s to integrate velocity
            self._F[i, i + 4] = 1.0

        self._H = np.zeros((self._dim_z, self._dim_x))
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0

        # process and measurement noise — sensible defaults; tune for your data
        self._Q = np.eye(self._dim_x) * 1.0
        # less certain about velocities; increase Q for velocity indices
        for i in range(4, 7):
            self._Q[i, i] = 10.0

        self._R = np.eye(self._dim_z) * 10.0

    def initiate(self, bbox):
        """Initialize state mean and covariance from bbox [x1,y1,x2,y2]"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        mean = np.array([cx, cy, s, r, 0., 0., 0.], dtype=float)
        cov = np.eye(self._dim_x) * 10.0
        return mean, cov

    def predict(self, mean, cov):
        mean_pred = self._F.dot(mean)
        cov_pred = self._F.dot(cov).dot(self._F.T) + self._Q
        return mean_pred, cov_pred

    def update(self, mean_pred, cov_pred, measurement):
        """measurement: [cx, cy, s, r]"""
        z = np.array(measurement, dtype=float)
        S = self._H.dot(cov_pred).dot(self._H.T) + self._R
        K = cov_pred.dot(self._H.T).dot(np.linalg.inv(S))
        y = z - self._H.dot(mean_pred)
        mean_upd = mean_pred + K.dot(y)
        cov_upd = (np.eye(self._dim_x) - K.dot(self._H)).dot(cov_pred)
        return mean_upd, cov_upd

    def gating_distance(self, mean_pred, cov_pred, measurements):
        """
        Compute squared Mahalanobis distance between predicted distribution and measurements.
        measurements: Nx4 array of [cx,cy,s,r]
        returns: array of shape (N,) distances
        """
        S = self._H.dot(cov_pred).dot(self._H.T) + self._R
        S_inv = np.linalg.inv(S)
        mean_z = self._H.dot(mean_pred)  # predicted measurement (4,)
        dists = []
        for m in measurements:
            y = m - mean_z
            dist = float(y.T.dot(S_inv).dot(y))
            dists.append(dist)
        return np.array(dists)

    def gate_threshold(self, dim_z=4, prob=0.995):
        # returns chi-square threshold for gating
        return chi2.ppf(prob, df=dim_z)

