#!/usr/bin/env python3
"""
convert_tracklets_to_mot.py

Convert per-camera tracklets JSON (outputs/logs/cam*_tracklets.json) to a single MOT-format TXT file.

Usage:
    python tools/convert_tracklets_to_mot.py --input_dir outputs/logs --out predictions_mot.txt --use_global merged_tracklets.json

If you pass --use_global merged_tracklets.json, the script will use global_id mapping and
write global ids. Otherwise it will write per-camera local ids prefixed by cam (camid_localid)
but the evaluator expects integer ids - if you have multiple cameras you should use merged global ids.

Note: The script expects each tracklet file to use the format produced previously:
    tracklet = {
       'cam_id': int,
       'local_id': int,
       'start_frame': int,
       'end_frame': int,
       'avg_feature': [...],
       'history': [(frame_idx, [x1,y1,x2,y2]), ...]
    }
"""
import os
import glob
import json
import argparse

def load_tracklets(input_dir):
    files = glob.glob(os.path.join(input_dir, 'cam*_tracklets.json'))
    all_t = []
    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)
        all_t.append((os.path.basename(f), data))
    return all_t

def build_global_map(merged_path):
    # merged_tracklets.json should have entries with 'global_id' and 'tracklets' where each tracklet has cam_id and local_id
    mapping = {}  # (cam_id, local_id) -> global_id
    if not os.path.exists(merged_path):
        return mapping
    with open(merged_path, 'r') as fh:
        merged = json.load(fh)
    for group in merged:
        gid = group.get('global_id')
        for t in group.get('tracklets', []):
            cam = int(t.get('cam_id'))
            local = int(t.get('local_id'))
            mapping[(cam, local)] = int(gid)
    return mapping

def convert(input_dir, out_path, merged_path=None, frame_offset=0):
    files_data = load_tracklets(input_dir)
    mapping = {}
    if merged_path:
        mapping = build_global_map(merged_path)
        print(f"Loaded global mapping for {len(mapping)} local tracklets from {merged_path}")

    lines = []
    for fname, data in files_data:
        # filename example: cam1_tracklets.json
        for tr in data:
            cam = int(tr.get('cam_id', 1))
            local = int(tr.get('local_id', tr.get('local_id', -1)))
            gid = None
            if mapping:
                gid = mapping.get((cam, local), None)
            # choose id to write: global if exists else use local id with camera offset (dangerous if evaluator expects int ids)
            write_id = gid if gid is not None else local + (cam * 1000000)  # fallback to unique integer
            for (frame_idx, bbox) in tr.get('history', []):
                x1, y1, x2, y2 = bbox
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                frame_out = int(frame_idx) + int(frame_offset)
                line = f"{frame_out},{write_id},{float(x1):.2f},{float(y1):.2f},{float(w):.2f},{float(h):.2f}"
                lines.append(line)
    # sort lines by frame
    lines = sorted(lines, key=lambda s: int(s.split(',')[0]))
    with open(out_path, 'w') as fo:
        fo.write("\n".join(lines))
    print(f"Wrote {len(lines)} lines to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='outputs/logs', help='folder with cam*_tracklets.json')
    parser.add_argument('--out', dest='out_path', default='outputs/predictions_mot.txt', help='output MOT txt')
    parser.add_argument('--use_global', default=None, help='path to merged_tracklets.json to use global ids')
    parser.add_argument('--frame_offset', type=int, default=0, help='optional frame offset to add to frame indices')
    args = parser.parse_args()
    convert(args.input_dir, args.out_path, merged_path=args.use_global, frame_offset=args.frame_offset)

