"""
tools/stitch_tracks.py

Offline stitcher: merges per-camera tracklets into global IDs using cosine similarity + time-window constraint.

Usage:
    python tools/stitch_tracks.py
"""
import os
import json
import glob
import numpy as np
from collections import defaultdict

LOG_DIR = 'outputs/logs'
OUT_MERGED = os.path.join(LOG_DIR, 'merged_tracklets.json')

# parameters
MAX_FRAME_GAP = 150   # maximum frames allowed between end of A and start of B (tune for your FPS & scene)
COSINE_THRESH = 0.45  # lower = stricter match (0..1)

def cosine_dist(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 1.0
    num = np.dot(a, b)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return 1.0 - (num / den)


def load_tracklets():
    files = glob.glob(os.path.join(LOG_DIR, 'cam*_tracklets.json'))
    all_t = []
    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)
            all_t.extend(data)
    print(f"Loaded {len(all_t)} tracklets from {len(files)} camera logs")
    return all_t


def stitch(tracklets):
    # simple greedy stitching:
    # - sort tracklets by start_frame
    # - iterate and try to assign to an existing global_id if cosine_dist <= thresh and time gap <= MAX_FRAME_GAP
    tracklets = sorted(tracklets, key=lambda x: (x['start_frame'], x['cam_id']))
    global_id = 0
    merged = []  # list of dict: {global_id, members: [tracklet indices], prototype_feature, last_end_frame}
    for idx, t in enumerate(tracklets):
        assigned = False
        t_feat = t.get('avg_feature', [])
        t_start = t['start_frame']
        t_cam = t['cam_id']

        # look for candidate merged entries
        # prefer matches where last_end_frame <= t_start and gap small
        candidates = []
        for m in merged:
            gap = t_start - m['last_end_frame']
            if gap < 0:
                # candidate starts earlier than merged last end — skip (we prefer forward time)
                continue
            if gap > MAX_FRAME_GAP:
                continue
            # spatial/graph constraints could be added here (camera connectivity)
            dist = cosine_dist(m['prototype_feature'], t_feat)
            candidates.append((dist, gap, m))

        if candidates:
            # choose lowest dist (best match)
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_dist, best_gap, best_m = candidates[0]
            if best_dist <= COSINE_THRESH:
                # assign
                best_m['members'].append(idx)
                # update prototype feature (mean of merged features)
                best_m['prototype_feature'] = list((np.array(best_m['prototype_feature']) * (len(best_m['members']) - 1) + np.array(t_feat)) / len(best_m['members']))
                best_m['last_end_frame'] = max(best_m['last_end_frame'], t['end_frame'])
                best_m['tracklets'].append(t)
                assigned = True

        if not assigned:
            # create new merged group
            merged.append({
                'global_id': global_id,
                'members': [idx],
                'prototype_feature': t_feat,
                'last_end_frame': t['end_frame'],
                'tracklets': [t]
            })
            global_id += 1

    # build output structure
    merged_out = []
    for m in merged:
        merged_out.append({
            'global_id': m['global_id'],
            'num_members': len(m['members']),
            'last_end_frame': int(m['last_end_frame']),
            'tracklets': m['tracklets'],
            'prototype_feature': m['prototype_feature']
        })
    return merged_out


if __name__ == '__main__':
    tracklets = load_tracklets()
    merged = stitch(tracklets)
    with open(OUT_MERGED, 'w') as fh:
        json.dump(merged, fh)
    print(f"Saved {len(merged)} merged global tracklets -> {OUT_MERGED}")
