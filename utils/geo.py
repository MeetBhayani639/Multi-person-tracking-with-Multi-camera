# utils/geo.py
import numpy as np
from camera_calibration import load_homography, pixel_to_world


def bbox_to_centroid(bbox):
    """
    bbox: [x1,y1,x2,y2] (floats)
    returns (cx, cy) pixel centroid
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return float(cx), float(cy)


def centroid_to_world(homography, centroid):
    """
    homography: 3x3 matrix mapping pixels->world
    centroid: (cx, cy)
    returns (X, Y) world coordinates (float)
    """
    cx, cy = centroid
    return pixel_to_world(homography, cx, cy)


def safe_load_homography(path):
    try:
        H = load_homography(path)
    except Exception:
        H = None
    return H
