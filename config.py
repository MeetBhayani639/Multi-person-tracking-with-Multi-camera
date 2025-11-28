# config.py
# Central configuration for Phase C

VIDEO_FILES = [
    'input_videos/cam1.mp4',
    'input_videos/cam2.mp4',
    # add more cameras here
]

# Homography file per camera (saved as numpy .npy). Create these with camera_calibration.py
HOMOGRAPHY_FILES = {
    1: 'calib/homography_cam1.npy',
    2: 'calib/homography_cam2.npy',
    # add more camera homography mapping as you calibrate them
}

# Output DB path
DB_PATH = 'outputs/tracks.db'

# Queue size for inter-process comms
QUEUE_MAXSIZE = 2000

# camera connectivity graph: expected travel time in SECONDS between cameras (if known).
# If unknown, leave None and the stitcher/statistics can learn or you can set heuristics.
#
# Example: camera 1 -> camera 2 expected travel time between 2 and 12 seconds:
CAMERA_TRANSITION = {
    (1, 2): (2.0, 12.0),
    (2, 1): (2.0, 12.0),
    # add more pairs as (src_cam, dst_cam): (min_seconds, max_seconds)
}

# Frames per second default (if video missing fps info)
DEFAULT_FPS = 25.0
