"""
tools/evaluate_mot.py

Simple MOT evaluator. Input:
- GT file in MOT TXT format per camera or combined (frame, id, x, y, w, h)
- Tracker output in same MOT TXT format (frame, id, x, y, w, h)

This script:
- reads GT and tracker outputs
- for each frame performs matching between GT boxes and predicted boxes (Hungarian) by IoU
- accumulates TP, FP, FN and ID switches
- computes MOTA, MOTP, IDF1 (approximate) metrics.

Note: For rigorous benchmarking use py-motmetrics package. This simple tool is for quick local evaluation.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import argparse
import os

def iou_xywh(a, b):
    # a and b are [x,y,w,h] with x,y top-left
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    xx1 = max(ax1, bx1); yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2); yy2 = min(ay2, by2)
    w = max(0., xx2 - xx1); h = max(0., yy2 - yy1)
    inter = w * h
    area1 = aw * ah; area2 = bw * bh
    return inter / (area1 + area2 - inter + 1e-6)

def load_mot_txt(path):
    """
    Expecting lines: frame, id, x, y, w, h, [..]
    Returns dict: frame -> list of dicts {'id':id, 'bbox': [x,y,w,h]}
    """
    frames = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
            frames.setdefault(frame, []).append({'id': tid, 'bbox': [x,y,w,h]})
    return frames

def evaluate(gt_path, pred_path, iou_thresh=0.5):
    gt = load_mot_txt(gt_path)
    pred = load_mot_txt(pred_path)
    frames = sorted(set(list(gt.keys()) + list(pred.keys())))
    total_gt = 0
    total_matches = 0
    total_fp = 0
    total_fn = 0
    total_overlap = 0.0
    id_switches = 0

    # Map: gt_id -> last matched pred_id
    last_match = {}

    for f in frames:
        g_list = gt.get(f, [])
        p_list = pred.get(f, [])
        total_gt += len(g_list)

        if len(g_list) == 0 and len(p_list) == 0:
            continue

        if len(g_list) == 0:
            total_fp += len(p_list)
            continue
        if len(p_list) == 0:
            total_fn += len(g_list)
            continue

        G = len(g_list); P = len(p_list)
        cost = np.zeros((G, P), dtype=float)
        for i, gi in enumerate(g_list):
            for j, pj in enumerate(p_list):
                cost[i, j] = 1.0 - iou_xywh(gi['bbox'], pj['bbox'])
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = set()
        for i, j in zip(row_ind, col_ind):
            iou_val = 1.0 - cost[i, j]
            if iou_val >= iou_thresh:
                total_matches += 1
                total_overlap += iou_val
                g_id = g_list[i]['id']
                p_id = p_list[j]['id']
                # ID switch?
                last = last_match.get(g_id, None)
                if last is not None and last != p_id:
                    id_switches += 1
                last_match[g_id] = p_id
                matched.add((i, j))
            else:
                # not a valid match
                pass
        # count unmatched
        matched_g = set([i for i, j in matched])
        matched_p = set([j for i, j in matched])
        total_fn += (G - len(matched_g))
        total_fp += (P - len(matched_p))

    mota = 1.0 - (total_fn + total_fp + id_switches) / max(1.0, total_gt)
    motp = total_overlap / max(1.0, total_matches)
    # IDF1 approximate: 2*TP / (2*TP + FP + FN) where TP=total_matches
    idf1 = 2.0 * total_matches / max(1.0, 2.0 * total_matches + total_fp + total_fn)

    stats = {
        'total_gt': int(total_gt),
        'total_matches': int(total_matches),
        'fp': int(total_fp),
        'fn': int(total_fn),
        'id_switches': int(id_switches),
        'MOTA': float(mota),
        'MOTP': float(motp),
        'IDF1_approx': float(idf1)
    }
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='path to ground-truth MOT txt')
    parser.add_argument('--pred', required=True, help='path to predictions MOT txt')
    parser.add_argument('--iou', type=float, default=0.5)
    args = parser.parse_args()
    res = evaluate(args.gt, args.pred, iou_thresh=args.iou)
    print("Evaluation results:")
    print(res)
