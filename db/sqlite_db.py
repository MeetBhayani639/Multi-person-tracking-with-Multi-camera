"""
sqlite_db.py

Simple wrapper for inserting per-frame track records into an SQLite DB.

Schema (table: detections):
- id (auto)
- cam_id INTEGER
- frame_idx INTEGER
- ts REAL  -- time in seconds from start (frame_idx / fps)
- local_track_id INTEGER  -- track id local to that camera
- bbox TEXT  -- JSON stringified [x1,y1,x2,y2]
- score REAL
- centroid_px_x REAL
- centroid_px_y REAL
- centroid_wld_x REAL
- centroid_wld_y REAL
"""
import sqlite3
import json
import os
from typing import Optional


class TrackDB:
    def __init__(self, db_path='outputs/tracks.db'):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        self._prepare_statements()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id INTEGER,
            frame_idx INTEGER,
            ts REAL,
            local_track_id INTEGER,
            bbox TEXT,
            score REAL,
            centroid_px_x REAL,
            centroid_px_y REAL,
            centroid_wld_x REAL,
            centroid_wld_y REAL
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cam_frame ON detections(cam_id, frame_idx);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_local_track ON detections(cam_id, local_track_id);")
        self.conn.commit()

    def _prepare_statements(self):
        self.insert_sql = """
            INSERT INTO detections (cam_id, frame_idx, ts, local_track_id, bbox, score,
                                    centroid_px_x, centroid_px_y, centroid_wld_x, centroid_wld_y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

    def insert_detection(self, cam_id: int, frame_idx: int, ts: float,
                         local_track_id: int, bbox, score: float,
                         centroid_px_x: float, centroid_px_y: float,
                         centroid_wld_x: Optional[float], centroid_wld_y: Optional[float]):
        bbox_json = json.dumps([float(b) for b in bbox])
        cur = self.conn.cursor()
        cur.execute(self.insert_sql, (int(cam_id), int(frame_idx), float(ts), int(local_track_id),
                                      bbox_json, float(score), float(centroid_px_x),
                                      float(centroid_px_y),
                                      None if centroid_wld_x is None else float(centroid_wld_x),
                                      None if centroid_wld_y is None else float(centroid_wld_y)))
        # commit can be batched for speed. For simplicity commit here:
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()
