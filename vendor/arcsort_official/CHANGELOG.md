# Changelog

## 0.1.1

- **Output boxes**: tracks now report the detection box they were matched to this frame
  (`output_box="detection"`, new default). `output_box="kalman"` keeps the 0.1.0 behaviour (the
  Kalman-filtered state). Association is unchanged; HOTA rises in every setting tested, e.g.
  validation with the full pipeline: SportsMOT 84.957 → 86.295, DanceTrack 60.617 → 61.329.
  `scripts/reproduce.sh` keeps `OUTPUT_BOX=kalman` by default so arXiv:2610.22306 v1 still
  reproduces exactly; `scripts/regression_gate.sh` stays byte-identical on all three submissions.
- **Motion-compensated trails**: `Visualizer(motion_compensated_trails=True)` with the new
  `ARCTracker.camera_motion` property; `tools/pipeline.py --motion-trails`. Uses Roboflow
  `trackers` (now part of the optional `viz` extra).
- `tools/run_cached.py --output-box`; `tools/evaluate.py --benchmark soccernet`.
- README: comparison with Roboflow `trackers` 2.6.1 (SoccerNet test, SportsMOT/DanceTrack val with
  detections and GT boxes); tested-detectors table rerun with 0.1.1.

## 0.1.0

First public release.
