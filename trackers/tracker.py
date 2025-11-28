"""
Minimal SORT-like tracker
- KalmanBoxTracker: an individual tracked object (with a Kalman filter)
- Tracker: manages multiple trackers, does data association with Hungarian

This implementation is intentionally compact and readable — replace or improve
with Deep SORT or another library as you progress.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """Computes IOU between two bboxes in [x1,y1,x2,y2] format"""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    return inter / (area1 + area2 - inter + 1e-6)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # bbox: [x1,y1,x2,y2]
        # State: [cx, cy, s, r, vx, vy, vs]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        s = w * h
        r = w / float(h + 1e-6)

        self.x = np.array([cx, cy, s, r, 0., 0., 0.], dtype=float)
        self.P = np.eye(7) * 10.
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        # simple constant velocity model for cx,cy,s
        F = np.eye(7)
        for i in range(3):
            F[i, i+4] = 1.0  # cx+=vx, cy+=vy, s+=vs

        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T)
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox):
        # measurement z = [cx, cy, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        z = np.array([cx, cy, s, r], dtype=float)

        H = np.zeros((4, 7))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1

        R = np.eye(4) * 10.
        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(7) - K.dot(H)).dot(self.P)

        self.hits += 1
        self.time_since_update = 0

    def get_state(self):
        cx, cy, s, r = self.x[0], self.x[1], self.x[2], self.x[3]
        w = np.sqrt(np.maximum(s * r, 1e-6))
        h = np.sqrt(np.maximum(s / r, 1e-6))
        x1 = cx - w / 2.
        y1 = cy - h / 2.
        x2 = cx + w / 2.
        y2 = cy + h / 2.
        return np.array([x1, y1, x2, y2])


class Tracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets):
        """
        dets: list of [x1,y1,x2,y2, score]
        returns list of tracks: [x1,y1,x2,y2, track_id]
        """
        # predict
        for trk in self.trackers:
            trk.predict()

        N = len(self.trackers)
        M = len(dets)

        if N == 0:
            for i in range(M):
                trk = KalmanBoxTracker(dets[i][:4])
                self.trackers.append(trk)
            return []

        iou_matrix = np.zeros((N, M), dtype=float)
        for t, trk in enumerate(self.trackers):
            for d, det in enumerate(dets):
                iou_matrix[t, d] = iou(trk.get_state(), det[:4])

        matched_indices = []
        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    continue
                matched_indices.append((r, c))

        unmatched_trks = set(range(N)) - {r for r, _ in matched_indices}
        unmatched_dets = set(range(M)) - {c for _, c in matched_indices}

        # update matched
        for r, c in matched_indices:
            self.trackers[r].update(dets[c][:4])

        # new trackers for unmatched detections
        for idx in unmatched_dets:
            trk = KalmanBoxTracker(dets[idx][:4])
            self.trackers.append(trk)

        ret = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or trk.age <= self.min_hits):
                ret.append(np.concatenate((trk.get_state(), [trk.id])).tolist())

            if trk.time_since_update > self.max_age:
                to_del.append(t)

        # delete dead tracks
        for idx in sorted(to_del, reverse=True):
            self.trackers.pop(idx)

        return ret
