"""
camera_calibration.py

Utilities to compute homography between image pixel coordinates and world (ground) plane coordinates.

Workflow:
1. Create a list of corresponding points between image and world (e.g. using a text editor):
   img_points = [(x1,y1), (x2,y2), ...]
   world_points = [(X1,Y1), (X2,Y2), ...]  # measured in meters (or any consistent units)
2. Call compute_and_save_homography(cam_id, img_points, world_points, out_path)

You can also manually collect points by printing clicks, but here we provide the basic math functions.
"""
import numpy as np
import cv2
import os


def compute_homography(img_pts, world_pts):
    """
    img_pts: list of (x,y) pixel coordinates
    world_pts: list of (X,Y) world coordinates (same length)
    returns 3x3 homography H that maps (x,y,1) -> (X,Y,1) in homogeneous coords
    """
    assert len(img_pts) >= 4 and len(img_pts) == len(world_pts), "Need >=4 corresponding points"
    img_pts = np.array(img_pts, dtype=np.float32)
    world_pts = np.array(world_pts, dtype=np.float32)
    H, status = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC)
    return H


def save_homography(H, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, H)
    print(f"Saved homography to {path}")


def load_homography(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    H = np.load(path)
    return H


def pixel_to_world(H, x, y):
    """
    Returns world (X, Y) for pixel (x, y) using homography H.
    """
    pt = np.array([x, y, 1.0])
    wpt = H.dot(pt)
    if abs(wpt[2]) < 1e-8:
        return float('nan'), float('nan')
    X = wpt[0] / wpt[2]
    Y = wpt[1] / wpt[2]
    return float(X), float(Y)


if __name__ == '__main__':
    # Example usage (manual)
    # img_pts = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    # world_pts = [(X1,Y1), (X2,Y2), (X3,Y3), (X4,Y4)]
    # H = compute_homography(img_pts, world_pts)
    # save_homography(H, 'calib/homography_cam1.npy')
    print("camera_calibration.py loaded. Use compute_homography(img_pts, world_pts) and save_homography(H,path).")
