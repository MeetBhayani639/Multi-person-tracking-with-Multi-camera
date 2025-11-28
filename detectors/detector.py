"""Simple detector wrapper.
If you install `ultralytics`, the wrapper will use YOLOv8. Otherwise it provides a dummy detection
for testing the pipeline.
"""
from typing import List, Tuple
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
            self.model = YOLO(model_path)
        else:
            self.model = None
    
    def detect(self, frame: np.ndarray, conf_thresh: float = 0.4):
        """Return list of detections as [x1,y1,x2,y2,conf]
        Coordinates are pixel coordinates.
        """
        if self.model is None:
            # Dummy: return empty list (or can generate random boxes for testing)
            return []


        results = self.model(frame)[0]
        dets = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            # class 0 is person in common COCO models
            if int(cls.item()) != 0:
                continue
            x1, y1, x2, y2 = box.cpu().numpy().tolist()
            c = float(conf.cpu().numpy())
            if c < conf_thresh:
                continue
            dets.append([x1, y1, x2, y2, c])
        return dets