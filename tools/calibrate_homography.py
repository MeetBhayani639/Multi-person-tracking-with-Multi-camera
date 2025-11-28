"""
tools/calibrate_homography.py

Usage:
  1) Prepare a JSON file with correspondences for each camera:
     tools/calib_points_cam1.json
     Format: { "pairs": [ { "pixel": [x,y], "world": [X,Y] }, ... ] }

  2) Run:
     python tools/calibrate_homography.py --input tools/calib_points_cam1.json --out outputs/homography_cam1.npy

This script computes 3x3 homography H such that:
  [Xw, Yw, 1]^T  ~ H * [x_pix, y_pix, 1]^T
"""
import json
import argparse
import numpy as np
import cv2
import os

def compute_homography_from_pairs(pairs):
    src = []
    dst = []
    for p in pairs:
        px = p['pixel']
        wx = p['world']
        src.append(px)
        dst.append(wx)
    src = np.array(src, dtype=float)
    dst = np.array(dst, dtype=float)
    H, inliers = cv2.findHomography(src, dst, method=cv2.RANSAC)
    return H, inliers

def save_homography(H, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, H)
    print("Saved homography:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON file with pixel<->world pairs")
    parser.add_argument("--out", required=True, help="Output .npy file for homography matrix")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
    pairs = data.get('pairs', [])
    if len(pairs) < 4:
        raise ValueError("Need at least 4 correspondence pairs for homography")

    H, inliers = compute_homography_from_pairs(pairs)
    if H is None:
        raise RuntimeError("Homography computation failed")
    save_homography(H, args.out)
