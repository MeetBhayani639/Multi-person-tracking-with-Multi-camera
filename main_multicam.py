"""
main_multicam.py

Runs independent trackers per camera, logs per-camera tracklets (JSON files) for offline stitching.

Usage:
    python main_multicam.py
"""
import os
import cv2
import json
from detectors.detector import Detector
from trackers.deep_sort import DeepSortTracker
from trackers.reid import FeatureExtractor
from utils.visualize import draw_tracks

VIDEO_FILES = [
    'input_videos/cam1.mp4',
    'input_videos/cam2.mp4',
]

OUTPUT_DIR = 'outputs/results'
LOG_DIR = 'outputs/logs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def process_camera(video_path, out_path, cam_id, detector, tracker, feat_extractor):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"[CAM {cam_id}] start processing {video_path} (fps={fps})")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        tracks = tracker.update(frame, dets, frame_idx=frame_idx)
        vis = draw_tracks(frame, tracks)
        out.write(vis)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[CAM {cam_id}] finished. Frames processed: {frame_idx}")

    # After processing, build per-camera tracklets from tracker.tracks
    tracklets = []
    for tr in tracker.tracks:
        tracklets.append(tr.as_tracklet_dict(cam_id))

    # save JSON
    out_json = os.path.join(LOG_DIR, f'cam{cam_id}_tracklets.json')
    with open(out_json, 'w') as f:
        json.dump(tracklets, f)
    print(f"[CAM {cam_id}] saved {len(tracklets)} tracklets -> {out_json}")


if __name__ == '__main__':
    # choose hist or torch mode for feature extractor
    feat_extractor = FeatureExtractor(mode='hist')  # or mode='torch' with model provided
    # create a tracker per camera
    trackers = []
    detectors = []
    for i in range(len(VIDEO_FILES)):
        detectors.append(Detector(model_path='yolov8n.pt', device='cpu'))
        trackers.append(DeepSortTracker(feature_extractor=feat_extractor, max_age=30,
                                       max_cosine_distance=0.4, iou_threshold=0.3))

    for i, vf in enumerate(VIDEO_FILES):
        outp = os.path.join(OUTPUT_DIR, f'res_cam{i+1}.mp4')
        process_camera(vf, outp, cam_id=i+1, detector=detectors[i], tracker=trackers[i], feat_extractor=feat_extractor)

    print("All cameras processed. Logs in outputs/logs/")
