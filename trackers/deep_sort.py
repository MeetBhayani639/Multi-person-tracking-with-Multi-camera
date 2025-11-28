"""
Deep SORT style tracker (simplified) with track history stored for offline stitching.

- Tracks now store bbox and frame history: useful to write tracklets to disk.
- Appearance matching first (cosine), IOU fallback next.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .reid import FeatureExtractor


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


class Track:
    _count = 0

    def __init__(self, bbox, feature, frame_idx, max_age=30):
        # bbox: [x1,y1,x2,y2]; feature: 1D numpy vector
        self.bbox = np.array(bbox, dtype=float)
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self.history = []         # list of (frame_idx, [x1,y1,x2,y2])
        self.history.append((int(frame_idx), self.bbox.tolist()))
        self.start_frame = int(frame_idx)
        self.end_frame = int(frame_idx)
        self.time_since_update = 0
        self.age = 0
        self.id = Track._count
        Track._count += 1
        self.max_age = max_age
        self.hits = 1

    def predict(self):
        # no motion model here (keeps simple)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, feature, frame_idx):
        self.bbox = np.array(bbox, dtype=float)
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 20:
                self.features = self.features[-20:]
        self.history.append((int(frame_idx), self.bbox.tolist()))
        self.end_frame = int(frame_idx)
        self.time_since_update = 0
        self.hits += 1

    def get_feature(self):
        if len(self.features) == 0:
            return None
        feat = np.mean(np.stack(self.features, axis=0), axis=0)
        norm = np.linalg.norm(feat) + 1e-6
        return feat / norm

    def to_tlbr(self):
        return self.bbox.tolist()

    def as_tracklet_dict(self, cam_id):
        # For logging: produces a serializable dict with averaged feature and history
        feat = self.get_feature()
        feat_list = feat.tolist() if feat is not None else []
        return {
            'cam_id': int(cam_id),
            'local_id': int(self.id),
            'start_frame': int(self.start_frame),
            'end_frame': int(self.end_frame),
            'avg_feature': feat_list,
            'history': [(int(f), [float(b) for b in bb]) for (f, bb) in self.history]
        }


class DeepSortTracker:
    def __init__(self, feature_extractor: FeatureExtractor = None,
                 max_age=30, max_cosine_distance=0.4, iou_threshold=0.3):
        self.tracks = []
        self.max_age = max_age
        self.max_cosine_distance = max_cosine_distance
        self.iou_threshold = iou_threshold
        self.feature_extractor = feature_extractor or FeatureExtractor(mode='hist')

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
        for t in self.tracks:
            t.predict()

    def update(self, frame, detections, frame_idx=0):
        """
        detections: list of [x1,y1,x2,y2,score]
        frame: current frame (BGR) to crop for features
        frame_idx: current frame index (int)
        returns: list of [x1,y1,x2,y2,track_id]
        """
        # extract features for each detection
        det_boxes = [d[:4] for d in detections]
        det_feats = []
        for bb in det_boxes:
            x1, y1, x2, y2 = map(int, bb)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(1, x2), max(1, y2)
            crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else frame
            feat = self.feature_extractor.extract(crop)
            det_feats.append(feat)

        # 1) Predict
        self.predict()

        N = len(self.tracks)
        M = len(detections)
        if N == 0:
            # create tracks for all detections
            for i in range(M):
                tr = Track(det_boxes[i], det_feats[i], frame_idx, max_age=self.max_age)
                self.tracks.append(tr)
            return [[*self.tracks[i].to_tlbr(), self.tracks[i].id] for i in range(len(self.tracks))]

        # 2) Appearance cost matrix (N x M)
        cost_matrix = np.zeros((N, M), dtype=float)
        for i, tr in enumerate(self.tracks):
            tref = tr.get_feature()
            for j in range(M):
                cost_matrix[i, j] = self.cosine_distance(tref, det_feats[j])

        # 3) First matching by appearance (threshold)
        matched_indices = []
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.max_cosine_distance:
                    matched_indices.append((r, c))

        matched_tracks = set([r for r, _ in matched_indices])
        matched_dets = set([c for _, c in matched_indices])

        # 4) For unmatched, use IOU matching between remaining tracks and remaining detections
        unmatched_tracks = [i for i in range(N) if i not in matched_tracks]
        unmatched_dets = [j for j in range(M) if j not in matched_dets]

        iou_matrix = np.zeros((len(unmatched_tracks), len(unmatched_dets)), dtype=float)
        for ti, t in enumerate(unmatched_tracks):
            for dj, d in enumerate(unmatched_dets):
                iou_matrix[ti, dj] = iou(self.tracks[t].to_tlbr(), det_boxes[dj])

        iou_matches = []
        if iou_matrix.size > 0:
            # Hungarian on -iou so we maximize IOU
            r, c = linear_sum_assignment(-iou_matrix)
            for rr, cc in zip(r, c):
                if iou_matrix[rr, cc] >= self.iou_threshold:
                    iou_matches.append((unmatched_tracks[rr], unmatched_dets[cc]))

        # aggregate matches
        all_matches = matched_indices + iou_matches
        matched_tracks = set([r for r, _ in all_matches])
        matched_dets = set([c for _, c in all_matches])

        # update matched tracks
        for r, c in all_matches:
            self.tracks[r].update(det_boxes[c], det_feats[c], frame_idx)

        # create new tracks for unmatched detections
        for j in range(M):
            if j not in matched_dets:
                tr = Track(det_boxes[j], det_feats[j], frame_idx, max_age=self.max_age)
                self.tracks.append(tr)

        # remove dead tracks
        for tr in list(self.tracks):
            if tr.time_since_update > self.max_age:
                self.tracks.remove(tr)

        # return active tracks
        out = []
        for tr in self.tracks:
            if tr.time_since_update == 0 or tr.hits >= 1:
                out.append([*tr.to_tlbr(), tr.id])
        return out
