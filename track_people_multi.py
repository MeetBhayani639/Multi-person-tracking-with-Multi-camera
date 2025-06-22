import cv2
import csv
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
from database import create_database, insert_tracking_record
from reid_utils import extract_embedding, cosine_similarity

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize Deep SORT tracker
tracker_cam1 = DeepSort(max_age=30)
tracker_cam2 = DeepSort(max_age=30)

# Open both camera feeds (videos)
cap1 = cv2.VideoCapture("videos/cam1.mp4")
cap2 = cv2.VideoCapture("videos/cam2.mp4")

# Initialize ReID memory across cameras
global_embeddings = {}  # {global_id: (embedding, camera_id)}
id_counter = 1000       # unique ID counter for assigning global IDs

# Open CSV log file
log_file = open("logs/people_log.csv", mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "CameraID", "TrackID", "X", "Y", "W", "H"])
create_database()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 and not ret2:
        break

    now = datetime.now().strftime("%H:%M:%S")

    for ret, frame, tracker, cam_id in [(ret1, frame1, tracker_cam1, "Cam1"), (ret2, frame2, tracker_cam2, "Cam2")]:
        if not ret:
            continue

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls] == "person" and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_tlwh()  # (x, y, w, h)
            x, y, w, h = map(int, bbox)
            track_id = track.track_id

            # STEP 3: Crop and extract embedding
            person_crop = frame[y:y+h, x:x+w]
            if person_crop.shape[0] < 64 or person_crop.shape[1] < 64:
                continue  # skip small crops

            embedding = extract_embedding(person_crop)

            # STEP 4: Match embedding with global embeddings
            matched_id = None
            for existing_id, (existing_emb, existing_cam) in global_embeddings.items():
                if existing_cam != cam_id:  # avoid matching within same camera
                    sim = cosine_similarity(embedding, existing_emb)
                    if sim > 0.9:
                        matched_id = existing_id
                        break

            if matched_id is not None:
                global_id = matched_id
            else:
                global_id = id_counter
                global_embeddings[global_id] = (embedding, cam_id)
                id_counter += 1

            # Use global ID instead of local track_id
            track_id = global_id

            # Draw results
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Log to database
            insert_tracking_record(now, cam_id, track_id, x, y, w, h)

        # Show the frame
        cv2.imshow(f'{cam_id} Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
log_file.close()
cv2.destroyAllWindows()
