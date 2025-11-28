"""
main_multicam_db.py

Runs per-camera trackers, writes per-frame tracks to SQLite DB (outputs/tracks.db),
and writes robust AVI outputs (with optional ffmpeg -> MP4 conversion).
"""
import os
import cv2
import json
from detectors.detector import Detector
from trackers.deep_sort import DeepSortTracker
from trackers.reid import FeatureExtractor
from utils.visualize import draw_tracks
from utils.video_io import create_writer, convert_avi_to_mp4, ffmpeg_exists
from db.tracks_db import init_db, insert_frame_track, insert_tracklet

VIDEO_FILES = [
    'input_videos/cam1.mp4',
    'input_videos/cam2.mp4',
]

OUTPUT_DIR = 'outputs/results'
LOG_DIR = 'outputs/logs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def process_camera_db(video_path, out_mp4_path, cam_id, detector, tracker, db_conn):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_path, out = create_writer(out_mp4_path, fps, (width, height))
    print(f"[CAM {cam_id}] Writing to {out_path} (fps={fps})")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        tracks = tracker.update(frame, dets, frame_idx=frame_idx)
        vis = draw_tracks(frame, tracks)
        out.write(vis)

        # save every track to DB
        for t in tracks:
            x1, y1, x2, y2, local_id = t
            insert_frame_track(db_conn, cam_id, frame_idx, int(local_id), float(x1), float(y1), float(x2), float(y2))

        frame_idx += 1

    cap.release()
    out.release()
    print(f"[CAM {cam_id}] finished. Frames processed: {frame_idx}")

    # after processing, also insert tracklets summary into tracklets table
    for tr in tracker.tracks:
        feat = tr.get_feature()
        feat_str = json.dumps(feat.tolist() if feat is not None else [])
        insert_tracklet(db_conn, cam_id, tr.id, tr.start_frame, tr.end_frame, feat_str)
    print(f"[CAM {cam_id}] tracklets saved: {len(tracker.tracks)}")

    # Optionally convert to mp4 if ffmpeg present
    target_mp4 = os.path.splitext(out_mp4_path)[0] + ".mp4"
    if ffmpeg_exists():
        ok = convert_avi_to_mp4(out_path, target_mp4)
        if ok:
            print(f"[CAM {cam_id}] Converted {out_path} -> {target_mp4}")
        else:
            print(f"[CAM {cam_id}] Conversion failed for {out_path}")
    else:
        print("[INFO] ffmpeg not found; skipping AVI -> MP4 conversion.")


if __name__ == "__main__":
    feat_extractor = FeatureExtractor(mode='hist')
    detectors = []
    trackers = []
    for i in range(len(VIDEO_FILES)):
        detectors.append(Detector(model_path='yolov8n.pt', device='cpu'))
        trackers.append(DeepSortTracker(feature_extractor=feat_extractor, max_age=30))

    conn = init_db()
    for i, vf in enumerate(VIDEO_FILES):
        outp = os.path.join(OUTPUT_DIR, f'res_cam{i+1}.mp4')  # API expects mp4 but writer will create avi
        process_camera_db(vf, outp, cam_id=i+1, detector=detectors[i], tracker=trackers[i], db_conn=conn)
    conn.close()
    print("All cameras processed and DB populated at outputs/tracks.db")
