import cv2

def draw_tracks(frame, tracks):
    """
    tracks: list of [x1, y1, x2, y2, id]
    Draw bounding boxes and track IDs on the frame.
    """
    for t in tracks:
        x1, y1, x2, y2, tid = map(int, t)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return frame
