import sqlite3

def create_database():
    conn = sqlite3.connect("tracking.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracking_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera_id TEXT,
            track_id INTEGER,
            x INTEGER,
            y INTEGER,
            w INTEGER,
            h INTEGER
        )
    """)
    conn.commit()
    conn.close()

def insert_tracking_record(timestamp, camera_id, track_id, x, y, w, h):
    conn = sqlite3.connect("tracking.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tracking_log (timestamp, camera_id, track_id, x, y, w, h)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, camera_id, track_id, x, y, w, h))
    conn.commit()
    conn.close()
