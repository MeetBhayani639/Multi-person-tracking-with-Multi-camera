# Multi-Person Tracking with Multiple Cameras (Deep SORT + YOLOv8)

### 🚀 End-to-End Multi-Camera Multi-Person Tracking System  
This project implements a complete **person tracking pipeline** using multiple cameras, YOLOv8 detection, DeepSORT-style tracking, and an **offline cross-camera Re-ID association module**.

---

## 📌 Features

### 🔹 Per-Camera Tracking (Online)
- YOLOv8 for person detection  
- Deep SORT-style tracker with appearance features  
- Independent tracker per camera  
- Logs per-camera tracklets to JSON  
- Outputs tracking videos per camera  

---

### 🔹 Cross-Camera Tracking (Offline)
After per-camera processing:
- Computes appearance embeddings  
- Uses cosine similarity + time window to merge tracklets  
- Produces **global IDs** across all cameras  
- Saves merged results to `merged_tracklets.json`

---

## 📁 Folder Structure

Multi-person-tracking-with-Multi-camera/
│
├── detectors/ # YOLOv8 wrapper
├── trackers/ # DeepSORT-style tracker + ReID
├── utils/ # Drawing utilities
├── tools/ # Offline stitching scripts
├── input_videos/ # Place cam1.mp4, cam2.mp4 (gitignored)
├── outputs/ # Tracking videos + logs (gitignored)
├── main_multicam.py # Run multi-camera tracking
├── requirements.txt
├── README.md
└── .gitignore


--------------------------------------

## 🛠 Installation

```bash
pip install -r requirements.txt

---------------------------------------

Run Tracking (Per-Camera):
python main_multicam.py

Produces:
outputs/results/res_cam1.mp4
outputs/results/res_cam2.mp4
outputs/logs/cam1_tracklets.json
outputs/logs/cam2_tracklets.json

---------------------------------------
Run Cross-Camera Stitching (Offline):
python tools/stitch_tracks.py

Produces:
outputs/logs/merged_tracklets.json

---------------------------------------

📌 Future Work

Camera geometry & calibration

Flask dashboard for multi-camera playback

Kalman motion model for DeepSORT

OSNet-based deep Re-ID model

Real-time association server

---------------------------------------

# ✅ Step 4 — Initialize Git and Push to GitHub  
Run these commands in the terminal:

### 1. Go to your project folder
```bash
cd Multi-person-tracking-with-Multi-camera

