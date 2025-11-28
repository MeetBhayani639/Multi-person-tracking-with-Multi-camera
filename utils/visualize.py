import cv2
import numpy as np

def draw_tracks(frame, tracks):
    """
    tracks: list of [x1, y1, x2, y2, id]
    Draw bounding boxes and track IDs on the frame.
    """
    for t in tracks:
        x1, y1, x2, y2, tid = map(int, t)
        color = tuple(int(c) for c in np.random.RandomState(tid).randint(0, 255, size=3))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{tid}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
