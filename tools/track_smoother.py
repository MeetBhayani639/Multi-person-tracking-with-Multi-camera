"""
tools/track_smoother.py

Read per-camera tracklets (outputs/logs/cam*_tracklets.json) and produce smoothed tracklets
with moving average or gaussian smoothing. Output saved to outputs/logs/cam*_tracklets_smoothed.json
"""
import glob
import json
import numpy as np
import os

LOG_DIR = 'outputs/logs'

def moving_average(arr, k=5):
    kernel = np.ones(k) / float(k)
    return np.convolve(arr, kernel, mode='same')

def smooth_tracklet(tracklet, k=5):
    # tracklet['history'] = [(frame_idx, [x1,y1,x2,y2]), ...]
    hist = tracklet['history']
    frames = [h[0] for h in hist]
    boxes = np.array([h[1] for h in hist])  # Nx4
    if len(boxes) < k:
        return tracklet
    sm_boxes = np.zeros_like(boxes)
    for i in range(4):
        sm_boxes[:, i] = moving_average(boxes[:, i], k=k)
    # replace history
    new_hist = [(int(frames[i]), [float(sm_boxes[i,0]), float(sm_boxes[i,1]),
                                 float(sm_boxes[i,2]), float(sm_boxes[i,3])]) for i in range(len(frames))]
    tracklet['history_smoothed'] = new_hist
    return tracklet

def main(k=5):
    files = glob.glob(os.path.join(LOG_DIR, 'cam*_tracklets.json'))
    for f in files:
        with open(f, 'r') as fh:
            tlist = json.load(fh)
        out = []
        for t in tlist:
            out.append(smooth_tracklet(t, k=k))
        outp = f.replace('.json', '_smoothed.json')
        with open(outp, 'w') as of:
            json.dump(out, of)
        print(f"Saved smoothed -> {outp}")

if __name__ == '__main__':
    main(k=5)
