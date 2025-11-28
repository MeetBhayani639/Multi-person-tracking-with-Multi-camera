"""
web/app.py
Simple Flask app:
- /      -> redirects to /grid
- /grid  -> shows HTML grid of videos (synchronized with a slider)
- /video/<id> -> serves processed videos (supports .avi or .mp4 and HTTP Range)
- /api/tracks -> returns JSON tracks for requested cam_ids and frame window
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, redirect, send_file, Response
from db.tracks_db import init_db, query_tracks_timewindow

app = Flask(__name__, template_folder='templates')
DB_PATH = "outputs/tracks.db"


@app.route("/")
def index():
    return redirect("/grid")


@app.route("/grid")
def grid():
    # only need cam_id; video path served by /video/<id>
    videos = [{"cam_id": 1}, {"cam_id": 2}]
    return render_template("grid.html", videos=videos)


# ----- Range-capable file sender ----- #
from flask import Response, request, send_file

def send_file_partial(path):
    """Range request-aware sender (same as earlier)."""
    file_size = os.path.getsize(path)
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(path, mimetype="video/mp4")

    bytes_unit, ranges = range_header.split("=", 1)
    if bytes_unit != "bytes":
        return Response(status=416)
    start_str, end_str = ranges.split("-")
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else file_size - 1
    end = min(end, file_size - 1)
    chunk_size = (end - start) + 1
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)
    resp = Response(data, status=206, mimetype="video/mp4")
    resp.headers.add("Content-Range", f"bytes {start}-{end}/{file_size}")
    resp.headers.add("Accept-Ranges", "bytes")
    resp.headers.add("Content-Length", str(chunk_size))
    return resp


@app.route("/video/<int:cam_id>")
def serve_video(cam_id):
    """
    Serve res_cam{cam_id}.avi if exists; else res_cam{cam_id}.mp4.
    The streaming helper supports Range requests.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    avi_path = os.path.join(project_root, f"outputs/results/res_cam{cam_id}.avi")
    mp4_path = os.path.join(project_root, f"outputs/results/res_cam{cam_id}.mp4")

    if os.path.exists(avi_path):
        path = avi_path
    elif os.path.exists(mp4_path):
        path = mp4_path
    else:
        return jsonify({"error": "Video not found", "paths": [avi_path, mp4_path]}), 404

    # send as mp4 mimetype (browsers accept it); AVI still plays fine
    return send_file_partial(path)


@app.route("/api/tracks")
def api_tracks():
    """
    API: /api/tracks?cams=1,2&from=0&to=10&unit=frames
         or /api/tracks?cams=1,2&from=0.0&to=0.5&unit=seconds&fps=25
    - cams: comma-separated camera ids
    - from/to: numeric bounds (frames or seconds)
    - unit: 'frames' (default) or 'seconds'
    - fps: frames-per-second when using seconds
    """
    try:
        cam_ids = request.args.get("cams", "1,2")
        raw_from = request.args.get("from", "0")
        raw_to = request.args.get("to", "100")
        unit = request.args.get("unit", "frames").lower()
        fps = float(request.args.get("fps", "25"))

        cam_list = [int(x) for x in cam_ids.split(",") if x.strip().isdigit()]
        if len(cam_list) == 0:
            return jsonify({"error": "Invalid cams parameter"}), 400

        if unit == "seconds":
            try:
                sec_from = float(raw_from)
                sec_to = float(raw_to)
            except ValueError:
                return jsonify({"error": "Invalid numeric value for 'from' or 'to' with unit=seconds"}), 400
            frame_from = int(max(0, round(sec_from * fps)))
            frame_to = int(max(0, round(sec_to * fps)))
        else:
            try:
                frame_from = int(float(raw_from))
                frame_to = int(float(raw_to))
            except ValueError:
                return jsonify({"error": "Invalid numeric value for 'from' or 'to' with unit=frames"}), 400

        if frame_from > frame_to:
            return jsonify({"error": "`from` must be <= `to`"}), 400

        conn = init_db(DB_PATH)
        rows = query_tracks_timewindow(conn, cam_list, frame_from, frame_to)
        conn.close()

        # keys for JSON output
        keys = ["cam_id", "frame_idx", "local_id", "global_id", "x1", "y1", "x2", "y2"]

        # --- DEDUPE STEP: remove near-identical duplicate rows (common when DB had duplicates) ---
        deduped = []
        seen = set()
        for r in rows:
            try:
                cam_id_r = int(r[0])
                frame_r = int(r[1])
                local_r = int(r[2])
                x1_r = round(float(r[4]), 2)
                y1_r = round(float(r[5]), 2)
                x2_r = round(float(r[6]), 2)
                y2_r = round(float(r[7]), 2)
                key = (cam_id_r, frame_r, local_r, x1_r, y1_r, x2_r, y2_r)
            except Exception:
                deduped.append(r)
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)

        out = [dict(zip(keys, r)) for r in deduped] if deduped else []
        return jsonify(out)

    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
