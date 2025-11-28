"""
Orchestrator for multiple videos (cameras).
This starter runs each video sequentially (one process) but uses separate Tracker instances per camera.
"""

import os
import cv2
from trackers.tracker import Tracker
from detectors.detector import Detector
from utils.visualize import draw_tracks


VIDEO_FILES = [
    'input_videos/cam1.mp4',
    'input_videos/cam2.mp4',
]

OUTPUT_DIR = 'outputs/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_camera(video_path, out_path, camera_id=0):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    detector = Detector()  # YOLOv8 wrapper
    tracker = Tracker(max_age=30, min_hits=1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detections: [x1, y1, x2, y2, conf]
        dets = detector.detect(frame)

        # tracking
        tracks = tracker.update(dets)

        # draw
        vis = draw_tracks(frame, tracks)

        out.write(vis)
        frame_idx += 1

    cap.release()
    out.release()


if __name__ == '__main__':
    # process each camera sequentially
    for i, vf in enumerate(VIDEO_FILES):
        outp = os.path.join(OUTPUT_DIR, f'res_cam{i+1}.mp4')
        print('Processing', vf, '->', outp)
        process_camera(vf, outp, camera_id=i)
        print('done')
