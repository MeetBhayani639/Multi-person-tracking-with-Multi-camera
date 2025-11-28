"""
Compute simple transition statistics (observed times) between cameras using DB.

This script queries outputs/tracks.db and finds candidate transitions where the same local track_id
appears across cameras (this is a heuristic) or where centroid world positions are available and we
can match spatially. Because we do offline stitching separately, this script is a helper to estimate
min/max travel times between camera pairs from logged detections.

Note: This is a starter script — refine for your dataset / ground truth.
"""
import sqlite3
import json
from collections import defaultdict
import numpy as np
import config

DB_PATH = config.DB_PATH


def load_all_detections():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT cam_id, frame_idx, ts, local_track_id, centroid_wld_x, centroid_wld_y FROM detections")
    rows = cur.fetchall()
    conn.close()
    return rows


def compute_stats(rows):
    # Group by local track within camera: (cam, local_track) -> list of ts
    groups = defaultdict(list)
    for cam_id, frame_idx, ts, local_id, x, y in rows:
        groups[(cam_id, local_id)].append((ts, x, y))

    # For each pair (camA, localA) -> (camB, localB) where start/end time order makes sense,
    # compute time difference between end of A and start of B if spatial data exists.
    # This is approximate and only for guidance.
    transitions = defaultdict(list)
    items = list(groups.items())
    for (camA, localA), seqA in items:
        seqA_sorted = sorted([t for t in seqA])
        endA = seqA_sorted[-1][0]
        for (camB, localB), seqB in items:
            if camA == camB:
                continue
            seqB_sorted = sorted([t for t in seqB])
            startB = seqB_sorted[0][0]
            # require reasonable ordering
            if 0 < (startB - endA) < 30.0:  # only consider transitions within 30s
                transitions[(camA, camB)].append(startB - endA)

    # summarize
    summary = {}
    for k, vals in transitions.items():
        arr = np.array(vals)
        summary[k] = {
            'count': int(len(arr)),
            'min_s': float(np.min(arr)),
            'median_s': float(np.median(arr)),
            'max_s': float(np.max(arr)),
            'mean_s': float(np.mean(arr))
        }
    return summary


if __name__ == '__main__':
    rows = load_all_detections()
    print(f"Loaded {len(rows)} detections from DB")
    stats = compute_stats(rows)
    print(json.dumps(stats, indent=2))
    # Optionally save results
    with open('outputs/transition_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("Saved outputs/transition_stats.json")
