"""
detectors/detector.py
YOLOv8 wrapper (uses ultralytics if installed). Returns detections in [x1,y1,x2,y2,score].
If ultralytics is not available it returns an empty list so you can test the pipeline.
"""
from typing import List
import numpy as np

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


class Detector:
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        if _HAS_YOLO:
            # set device inside model call when running (model will auto-download if not found)
            self.model = YOLO(model_path)
        else:
            self.model = None

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.35) -> List[List[float]]:
        """
        Returns list of detections: [x1, y1, x2, y2, conf]
        Only returns class=person (COCO class 0).
        """
        if self.model is None:
            return []

        # Ultralytics returns results; we pick first batch (single image)
        results = self.model(frame)[0]
        dets = []
        if results.boxes is None:
            return dets

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls.item()) != 0:
                continue
            c = float(conf.cpu().numpy())
            if c < conf_thresh:
                continue
            x1, y1, x2, y2 = box.cpu().numpy().tolist()
            dets.append([float(x1), float(y1), float(x2), float(y2), c])
        return dets
