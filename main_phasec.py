"""
main_phasec.py

Asynchronous multi-camera runner for Phase C.

- One process per camera: reads frames, runs detector + tracker, sends per-frame detections to a central queue.
- One DB writer process consumes from the queue and writes per-detection rows into SQLite DB with world coords.
- Uses homography per camera if present to compute world coords for centroid.

Run:
    python main_phasec.py
"""
import os
import time
import json
import multiprocessing as mp
import cv2
from detectors.detector import Detector
from trackers.deep_sort import DeepSortTracker
from trackers.reid import FeatureExtractor
from utils.visualize import draw_tracks
from utils.geo import bbox_to_centroid, safe_load_homography, centroid_to_world
from db.sqlite_db import TrackDB
import config

VIDEO_FILES = config.VIDEO_FILES
HOMOS = config.HOMOGRAPHY_FILES
DB_PATH = config.DB_PATH
QUEUE_MAXSIZE = config.QUEUE_MAXSIZE
DEFAULT_FPS = config.DEFAULT_FPS


def camera_worker(cam_index, video_path, homography_path, queue: mp.Queue, device='cpu'):
    """
    Each camera runs in its own process.
    It pushes dictionaries to queue with detection & track info for DB writing and optional visualization.
    """
    cam_id = cam_index + 1
    print(f"[CAM {cam_id}] starting worker for {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    detector = Detector(model_path='yolov8n.pt', device=device)
    feat_extractor = FeatureExtractor(mode='hist')  # change to 'torch' and load model for real ReID
    tracker = DeepSortTracker(feature_extractor=feat_extractor, max_age=30,
                              max_cosine_distance=0.4, iou_threshold=0.3)

    H = None
    if homography_path is not None:
        try:
            H = safe_load_homography(homography_path)
            print(f"[CAM {cam_id}] loaded homography {homography_path}")
        except Exception:
            H = None
            print(f"[CAM {cam_id}] no homography found at {homography_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        tracks = tracker.update(frame, dets, frame_idx=frame_idx)

        # for each active track, send DB record for that detection
        ts = frame_idx / float(fps)
        for tr in tracks:
            x1, y1, x2, y2, local_id = tr
            # find score from detections if possible (best-effort). Use 0 if not known.
            score = 0.0
            for d in dets:
                # match by IOU (simple)
                dx1, dy1, dx2, dy2, ds = d
                iou_val = 0.0
                # compute IOU quickly
                ix1 = max(x1, dx1); iy1 = max(y1, dy1); ix2 = min(x2, dx2); iy2 = min(y2, dy2)
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                inter = iw * ih
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (dx2 - dx1) * (dy2 - dy1)
                if area1 + area2 - inter > 0:
                    iou_val = inter / (area1 + area2 - inter)
                if iou_val > 0.3:
                    score = float(ds)
                    break

            # centroid and world coords
            cx, cy = bbox_to_centroid([x1, y1, x2, y2])
            world_x, world_y = (None, None)
            if H is not None:
                try:
                    world_x, world_y = centroid_to_world(H, (cx, cy))
                except Exception:
                    world_x, world_y = (None, None)

            # push record to queue
            rec = {
                'cam_id': int(cam_id),
                'frame_idx': int(frame_idx),
                'ts': float(ts),
                'local_track_id': int(local_id),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'score': float(score),
                'centroid_px': [float(cx), float(cy)],
                'centroid_wld': [world_x, world_y]
            }
            # block if queue full (backpressure)
            queue.put(rec)

        frame_idx += 1

    cap.release()
    print(f"[CAM {cam_id}] finished processing. Sent all records.")


def db_writer_worker(queue: mp.Queue, db_path):
    print("[DB WRITER] starting")
    db = TrackDB(db_path)
    try:
        while True:
            item = queue.get()
            if item is None:
                # sentinel for shutdown
                break
            db.insert_detection(
                cam_id=item['cam_id'],
                frame_idx=item['frame_idx'],
                ts=item['ts'],
                local_track_id=item['local_track_id'],
                bbox=item['bbox'],
                score=item['score'],
                centroid_px_x=item['centroid_px'][0],
                centroid_px_y=item['centroid'][1] if len(item['centroid_px']) > 1 else item['centroid_px'][0],
                centroid_wld_x=(None if item['centroid_wld'][0] is None else item['centroid_wld'][0]),
                centroid_wld_y=(None if item['centroid_wld'][1] is None else item['centroid_wld'][1]),
            )
    except KeyboardInterrupt:
        print("[DB WRITER] interrupted")
    finally:
        db.close()
        print("[DB WRITER] closed DB")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    queue = manager.Queue(maxsize=QUEUE_MAXSIZE)

    processes = []
    # spawn DB writer
    db_proc = mp.Process(target=db_writer_worker, args=(queue, DB_PATH))
    db_proc.start()
    processes.append(db_proc)

    # spawn camera workers
    for idx, vf in enumerate(VIDEO_FILES):
        hom_path = HOMOS.get(idx+1, None)
        p = mp.Process(target=camera_worker, args=(idx, vf, hom_path, queue, 'cpu'))
        p.start()
        processes.append(p)

    # wait for camera workers to finish
    for p in processes[1:]:
        p.join()

    # send shutdown sentinel to DB writer
    queue.put(None)
    db_proc.join()
    print("All done. DB saved at:", DB_PATH)
