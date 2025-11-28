"""
db/tracks_db.py
SQLite helper for storing per-frame tracks and merged/global track mappings.

Schema:
- tracks table: stores per-frame detection & track id (global or local)
- tracklets table: summary per local track (cam_id/local_id -> aggregated info)
- global_map table: maps local track entries to global_id (used after stitching)

This is simple and works for analytics and the web UI.
"""
import sqlite3
import os
from typing import List, Dict, Any

DB_PATH = "outputs/tracks.db"

def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cam_id INTEGER,
        frame_idx INTEGER,
        local_id INTEGER,
        global_id INTEGER,
        x1 REAL, y1 REAL, x2 REAL, y2 REAL
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS tracklets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cam_id INTEGER,
        local_id INTEGER,
        start_frame INTEGER,
        end_frame INTEGER,
        avg_feature TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS global_map (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cam_id INTEGER,
        local_id INTEGER,
        global_id INTEGER
    )
    ''')
    conn.commit()
    return conn

def insert_frame_track(conn, cam_id:int, frame_idx:int, local_id:int, x1:float, y1:float, x2:float, y2:float, global_id: int = None):
    c = conn.cursor()
    c.execute('''
        INSERT INTO tracks (cam_id, frame_idx, local_id, global_id, x1, y1, x2, y2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (cam_id, frame_idx, local_id, global_id, x1, y1, x2, y2))
    conn.commit()

def insert_tracklet(conn, cam_id:int, local_id:int, start_frame:int, end_frame:int, avg_feature: str):
    c = conn.cursor()
    c.execute('''
        INSERT INTO tracklets (cam_id, local_id, start_frame, end_frame, avg_feature)
        VALUES (?, ?, ?, ?, ?)
    ''', (cam_id, local_id, start_frame, end_frame, avg_feature))
    conn.commit()

def set_global_mapping(conn, cam_id:int, local_id:int, global_id:int):
    c = conn.cursor()
    c.execute('''
        INSERT INTO global_map (cam_id, local_id, global_id)
        VALUES (?, ?, ?)
    ''', (cam_id, local_id, global_id))
    conn.commit()

def query_tracks_timewindow(conn, cam_ids:List[int], frame_from:int, frame_to:int):
    c = conn.cursor()
    ids_str = ",".join(str(i) for i in cam_ids)
    q = f'''
        SELECT cam_id, frame_idx, local_id, global_id, x1, y1, x2, y2
        FROM tracks
        WHERE cam_id IN ({ids_str}) AND frame_idx BETWEEN ? AND ?
        ORDER BY cam_id, frame_idx
    '''
    c.execute(q,(frame_from, frame_to))
    return c.fetchall()
