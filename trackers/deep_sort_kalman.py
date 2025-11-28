"""
deep_sort_kalman.py

Deep SORT-like tracker with:
- Kalman motion model (kalman.py)
- Mahalanobis gating for motion-based gating
- Appearance matching (cosine) then IOU fallback
- Reappearance matching: keep 'lost' tracks buffer to reassign IDs
- Simple track smoothing (moving average of last K boxes)

This keeps an approachable API: tracker.update(frame, detections, frame_idx)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .reid import FeatureExtractor
from .kalman import KalmanFilterBox
import math
import collections


def iou(bb1, bb2):
    xx1 = max(bb1[0], bb2[0])
    yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2])
    yy2 = min(bb1[3], bb2[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    a1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    a2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return inter / (a1 + a2 - inter + 1e-6)


class TrackState:
    Active = 1
    Lost = 2
    Removed = 3


class Track:
    _count = 0

    def __init__(self, mean, cov, feature, frame_idx, max_age=30, smoothing_k=5):
        self.mean = mean.copy()
        self.cov = cov.copy()
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self.start_frame = int(frame_idx)
        self.end_frame = int(frame_idx)
        self.time_since_update = 0
        self.age = 0
        self.id = Track._count
        Track._count += 1
        self.max_age = max_age
        self.hits = 1
        self.state = TrackState.Active
        self.smoothing_k = smoothing_k
        self.box_history = collections.deque(maxlen=smoothing_k)
        # push initial bbox from mean
        self.box_history.append(self.to_tlbr())
        self.last_frame = int(frame_idx)

    def predict(self, kf: KalmanFilterBox):
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, measurement, feature, frame_idx, kf: KalmanFilterBox):
        # measurement is bbox [x1,y1,x2,y2]; convert to measurement vector z
        x1, y1, x2, y2 = measurement
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        z = np.array([cx, cy, s, r], dtype=float)
        self.mean, self.cov = kf.update(self.mean, self.cov, z)
        self.time_since_update = 0
        self.hits += 1
        self.end_frame = int(frame_idx)
        self.last_frame = int(frame_idx)
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 30:
                self.features = self.features[-30:]
        self.box_history.append(self.to_tlbr())
        self.state = TrackState.Active

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def is_tentative(self):
        return self.hits < 3

    def is_confirmed(self):
        return self.hits >= 3

    def is_lost(self):
        return self.state == TrackState.Lost

    def get_feature(self):
        if len(self.features) == 0:
            return None
        feat = np.mean(np.stack(self.features, axis=0), axis=0)
        norm = np.linalg.norm(feat) + 1e-6
        return feat / norm

    def to_tlbr(self):
        # decode mean state to bbox
        cx, cy, s, r = self.mean[0], self.mean[1], self.mean[2], self.mean[3]
        w = math.sqrt(max(s * r, 1e-6))
        h = math.sqrt(max(s / r, 1e-6))
        x1 = cx - w / 2.
        y1 = cy - h / 2.
        x2 = cx + w / 2.
        y2 = cy + h / 2.
        return [float(x1), float(y1), float(x2), float(y2)]

    def smoothed_box(self):
        # return smoothed bbox as average of last K boxes
        if len(self.box_history) == 0:
            return self.to_tlbr()
        arr = np.array(self.box_history)
        return arr.mean(axis=0).tolist()


class DeepSortKalman:
    def __init__(self, feature_extractor: FeatureExtractor = None,
                 max_age=30, max_cosine_distance=0.4, iou_threshold=0.3,
                 gate_threshold=9.488, smoothing_k=5, reid_reappear_max=90):
        self.kf = KalmanFilterBox()
        self.tracks = []
        self.max_age = max_age
        self.max_cosine_distance = max_cosine_distance
        self.iou_threshold = iou_threshold
        self.feature_extractor = feature_extractor or FeatureExtractor(mode='hist')
        self.gate_threshold = gate_threshold  # mahalanobis gating threshold
        self.smoothing_k = smoothing_k
        self.reid_reappear_max = reid_reappear_max  # frames to keep lost tracks for reappearance

    @staticmethod
    def cosine_distance(a, b):
        if a is None or b is None:
            return 1.0
        a = a.astype(float)
        b = b.astype(float)
        num = np.dot(a, b)
        den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        return 1.0 - (num / den)

    def predict(self):
        for tr in self.tracks:
            tr.predict(self.kf)

    def gating(self, track, detections_cxysr):
        """Return True/False mask for which detections pass Mahalanobis gating for this track."""
        dists = self.kf.gating_distance(track.mean, track.cov, detections_cxysr)
        return dists <= self.gate_threshold

    def update(self, frame, detections, frame_idx=0):
        """
        detections: list of [x1,y1,x2,y2,score]
        returns list of [x1,y1,x2,y2,track_id]
        """
        det_boxes = [d[:4] for d in detections]
        det_scores = [d[4] if len(d) > 4 else 0.0 for d in detections]
        M = len(det_boxes)

        # extract features
        det_feats = []
        det_meas = []  # measurements as [cx,cy,s,r]
        for bb in det_boxes:
            x1, y1, x2, y2 = map(int, bb)
            crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else frame
            feat = self.feature_extractor.extract(crop)
            det_feats.append(feat)
            w = max(1.0, (x2 - x1))
            h = max(1.0, (y2 - y1))
            cx = x1 + w / 2.
            cy = y1 + h / 2.
            s = w * h
            r = w / float(h + 1e-6)
            det_meas.append([cx, cy, s, r])

        # 1) Predict
        self.predict()

        N = len(self.tracks)
        if N == 0:
            # initialize new tracks
            for i in range(M):
                mean, cov = self.kf.initiate(det_boxes[i])
                tr = Track(mean, cov, det_feats[i], frame_idx, max_age=self.max_age, smoothing_k=self.smoothing_k)
                self.tracks.append(tr)
            return [[*tr.to_tlbr(), tr.id] for tr in self.tracks]

        # 2) Compute appearance cost (N x M)
        cost_app = np.zeros((N, M), dtype=float)
        for i, tr in enumerate(self.tracks):
            tref = tr.get_feature()
            for j in range(M):
                cost_app[i, j] = self.cosine_distance(tref, det_feats[j])

        # 3) Gating mask using Mahalanobis
        gating_mask = np.ones((N, M), dtype=bool)
        for i, tr in enumerate(self.tracks):
            pass_gates = self.kf.gating_distance(tr.mean, tr.cov, det_meas) <= self.gate_threshold
            gating_mask[i, :] = pass_gates

        # set cost large where gating fails
        INF = 1e6
        cost_app_masked = cost_app.copy()
        cost_app_masked[~gating_mask] = INF

        # 4) Hungarian on appearance cost
        matched, unmatched_tracks, unmatched_dets = [], list(range(N)), list(range(M))
        if cost_app_masked.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_app_masked)
            for r, c in zip(row_ind, col_ind):
                if cost_app_masked[r, c] <= self.max_cosine_distance:
                    matched.append((r, c))
                    unmatched_tracks.remove(r)
                    unmatched_dets.remove(c)

        # 5) IOU matching for unmatched
        if len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            iou_mat = np.zeros((len(unmatched_tracks), len(unmatched_dets)), dtype=float)
            for ti, tr_idx in enumerate(unmatched_tracks):
                for dj, det_idx in enumerate(unmatched_dets):
                    iou_mat[ti, dj] = iou(self.tracks[tr_idx].to_tlbr(), det_boxes[det_idx])
            row2, col2 = linear_sum_assignment(-iou_mat)
            for rr, cc in zip(row2, col2):
                if iou_mat[rr, cc] >= self.iou_threshold:
                    t_idx = unmatched_tracks[rr]
                    d_idx = unmatched_dets[cc]
                    matched.append((t_idx, d_idx))
            # recompute unmatched lists
            matched_tracks = set([m[0] for m in matched])
            matched_dets_set = set([m[1] for m in matched])
            unmatched_tracks = [t for t in unmatched_tracks if t not in matched_tracks]
            unmatched_dets = [d for d in unmatched_dets if d not in matched_dets_set]

        # 6) Update matched
        for tr_idx, det_idx in matched:
            self.tracks[tr_idx].update(det_boxes[det_idx], det_feats[det_idx], frame_idx, self.kf)

        # 7) Mark unmatched tracks as missed
        for tr_idx in unmatched_tracks:
            tr = self.tracks[tr_idx]
            tr.time_since_update += 1
            if tr.time_since_update > tr.max_age:
                tr.mark_lost()

        # 8) Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            mean, cov = self.kf.initiate(det_boxes[det_idx])
            tr = Track(mean, cov, det_feats[det_idx], frame_idx, max_age=self.max_age, smoothing_k=self.smoothing_k)
            self.tracks.append(tr)

        # 9) Reappearance logic: try to match lost tracks (within reid_reappear_max frames)
        # Build list of lost but recent tracks
        lost_tracks = [tr for tr in self.tracks if tr.is_lost() and (frame_idx - tr.last_frame) <= self.reid_reappear_max]
        if len(lost_tracks) > 0 and M > 0:
            # compute dist between lost prototype feature and unmatched detections' features
            for lt in lost_tracks:
                lt_feat = lt.get_feature()
                best_j = None
                best_dist = 1.0
                for j in unmatched_dets:
                    dfeat = det_feats[j]
                    dist = self.cosine_distance(lt_feat, dfeat)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                if best_j is not None and best_dist <= self.max_cosine_distance:
                    # reassign detection j to lost track lt
                    lt.update(det_boxes[best_j], det_feats[best_j], frame_idx, self.kf)
                    lt.time_since_update = 0
                    lt.state = TrackState.Active
                    # remove that det from unmatched_dets
                    if best_j in unmatched_dets:
                        unmatched_dets.remove(best_j)

        # 10) Remove tracks that are lost > max_age
        for tr in list(self.tracks):
            if tr.is_lost() and (frame_idx - tr.last_frame) > self.max_age:
                tr.mark_removed()
                self.tracks.remove(tr)

        # 11) prepare outputs: only confirmed/active tracks
        out = []
        for tr in self.tracks:
            if tr.is_confirmed() and not tr.is_lost():
                # return smoothed box
                sb = tr.smoothed_box()
                out.append([sb[0], sb[1], sb[2], sb[3], tr.id])
        return out
