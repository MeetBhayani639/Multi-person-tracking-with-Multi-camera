from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Open video file or camera
cap = cv2.VideoCapture("videos/input_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if model.names[cls] == 'person' and conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Deep SORT expects [x, y, width, height]
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked bounding boxes and IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltwh()
        track_id = track.track_id
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("YOLO + Deep SORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
