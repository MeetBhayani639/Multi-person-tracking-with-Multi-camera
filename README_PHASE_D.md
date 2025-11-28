# Phase D — Improvements & Evaluation

This folder adds Kalman motion, Mahalanobis gating, reappearance handling, smoothing, evaluation scripts, and a Re-ID fine-tuning scaffold.

## Files added
- trackers/kalman.py
- trackers/deep_sort_kalman.py
- trackers/reid_finetune.py
- tools/track_smoother.py
- tools/evaluate_mot.py

## How to use

### 1) Replace tracker usage with DeepSortKalman
In your runner (e.g., main_multicam.py or main_phasec.py) replace:
```py
from trackers.deep_sort import DeepSortTracker
