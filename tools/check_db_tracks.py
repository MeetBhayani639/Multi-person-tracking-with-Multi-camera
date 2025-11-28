# tools/check_db_tracks.py
import sqlite3, json, os
DB = "outputs/tracks.db"

if not os.path.exists(DB):
    print("DB not found:", DB)
    raise SystemExit(1)

conn = sqlite3.connect(DB)
c = conn.cursor()

# list tables
print("Tables:")
for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(" -", row[0])

# count rows by camera
print("\nCounts per cam:")
for cam_id, in c.execute("SELECT DISTINCT cam_id FROM tracks"):
    r = c.execute("SELECT COUNT(*) FROM tracks WHERE cam_id=?", (cam_id,)).fetchone()[0]
    mn = c.execute("SELECT MIN(frame_idx), MAX(frame_idx) FROM tracks WHERE cam_id=?", (cam_id,)).fetchone()
    print(f"Cam {cam_id}: rows={r}, frame_min={mn[0]}, frame_max={mn[1]}")

# sample first 20 rows
print("\nSample rows (first 20):")
for r in c.execute("SELECT cam_id, frame_idx, local_id, global_id, x1,y1,x2,y2 FROM tracks ORDER BY cam_id, frame_idx LIMIT 20"):
    print(r)

# sample tracklets summary
print("\nTracklets (first 10):")
for r in c.execute("SELECT id, cam_id, local_id, start_frame, end_frame FROM tracklets ORDER BY cam_id LIMIT 10"):
    print(r)

conn.close()
