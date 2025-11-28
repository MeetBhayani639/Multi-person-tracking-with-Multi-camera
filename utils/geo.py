"""
utils/geo.py
Helpers to load homography matrix and map pixel coordinates to world coords.
"""
import numpy as np

def load_homography(path):
    H = np.load(path)
    return H

def pixel_to_world(px, py, H):
    """
    px,py = pixel coordinates
    H = 3x3 homography that maps pixel -> world (up to scale)
    returns (Xw, Yw) in same unit as homography destination.
    """
    pt = np.array([px, py, 1.0], dtype=float)
    wp = H.dot(pt)
    if abs(wp[2]) < 1e-8:
        return None
    return (float(wp[0] / wp[2]), float(wp[1] / wp[2]))

def bbox_centroid(bbox):
    x1,y1,x2,y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy
