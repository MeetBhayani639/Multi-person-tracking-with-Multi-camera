"""
main_multiproc.py
Simple multiprocessing runner: one process per camera.
This keeps each camera isolated and faster on multi-core CPUs.
"""
import multiprocessing as mp
from main_multicam_db import process_camera_db, VIDEO_FILES
from detectors.detector import Detector
from trackers.deep_sort import DeepSortTracker
from trackers.reid import FeatureExtractor
from db.tracks_db import init_db

def worker(args):
    idx, vf = args
    feat_extractor = FeatureExtractor(mode='hist')
    detector = Detector(model_path='yolov8n.pt', device='cpu')
    tracker = DeepSortTracker(feature_extractor=feat_extractor)
    conn = init_db()
    outp = f'outputs/results/res_cam{idx+1}.mp4'
    process_camera_db(vf, outp, cam_id=idx+1, detector=detector, tracker=tracker, db_conn=conn)
    conn.close()

if __name__ == "__main__":
    pool = mp.Pool(processes=min(len(VIDEO_FILES), mp.cpu_count()))
    pool.map(worker, list(enumerate(VIDEO_FILES)))
    pool.close()
    pool.join()
