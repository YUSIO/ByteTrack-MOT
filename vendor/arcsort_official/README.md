# ARC-SORT

**Adaptive, Robust, Compensated Multi-Object Tracking with Interacting Multiple Models**

[![arXiv](https://img.shields.io/badge/arXiv-2610.22306-b31b1b.svg)](https://arxiv.org/abs/2610.22306)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/Athlitix/arcsort/blob/main/LICENSE)
[![Project page](https://img.shields.io/badge/project-page-1f4fd8.svg)](https://athlitix.github.io/arcsort/)
[![PyPI](https://img.shields.io/pypi/v/arcsort.svg)](https://pypi.org/project/arcsort/)

ARC-SORT is a tracking-by-detection multi-object tracker for fast, non-linear motion under a
moving camera, such as athletes in sports broadcasts. Nothing is trained. It works with any detector and
runs at about 50 FPS on a laptop RTX 3060.

The idea: **the tracker measures its own uncertainty, and that measurement decides how it acts.**

- **IMM Kalman filter** (constant velocity / constant acceleration / coordinated turn) gives each
  track a predictive covariance.
- **Uncertainty-adaptive Expansion IoU**: the less reliable a track's motion prediction, the larger
  the region in which a detection may match it.
- **Deferred identity arbitration**: when two tracks cross and the assignment was a close call,
  the pair stays under review; at most one corrective identity swap is applied once post-crossing
  appearance evidence shows the first decision was wrong.
- **Camera-motion compensation**, BYTE two-stage matching, OC-SORT observation-centric recovery,
  and appearance (ReID) costs.
- **Offline tracklet merge** that links players who leave the frame by appearance alone (no
  motion gate, since extrapolating motion over seconds off-screen carries no information).

## Results

Test-set numbers are from the official evaluation servers (CodaBench).

| Benchmark (test) | HOTA | AssA | DetA | MOTA | IDF1 | IDs |
|---|---|---|---|---|---|---|
| SportsMOT | **81.82** | 76.24 | 87.84 | 97.09 | 88.47 | 1414 |
| SportsMOT, Deep-EIoU's detections + embeddings | 80.31 | 74.14 | 87.04 | 96.65 | 86.92 | 1320 |
| DanceTrack | **62.1** | 48.2 | 80.3 | 92.4 | 66.5 | — |

As of August 2026, SportsMOT has the fewest identity switches of all 69 leaderboard entries.
On Deep-EIoU's own inputs, ARC-SORT is +3.1 HOTA over Deep-EIoU (77.2).

Validation numbers, reproducible locally (pooled TrackEval):

| Benchmark (val) | HOTA | AssA | IDF1 |
|---|---|---|---|
| SportsMOT | 84.96 | 82.96 | 92.79 |
| DanceTrack | 60.62 | 47.44 | 64.80 |

## Installation

Python 3.10+ (developed on 3.12).

```bash
git clone https://github.com/Athlitix/arcsort.git && cd arcsort
pip install -e ".[reid,eval]"
```

For the GPU stages (detection, embeddings), install a CUDA build of PyTorch first; see
[pytorch.org](https://pytorch.org/get-started/locally/). Tracking from cached inputs runs on CPU.

External code (clone into `external/`, which is gitignored):

| Repo | Needed for | Command |
|---|---|---|
| [TrackEval](https://github.com/JonathonLuiten/TrackEval) | val evaluation | `git clone https://github.com/JonathonLuiten/TrackEval external/TrackEval` |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | regenerating detections | `git clone https://github.com/Megvii-BaseDetection/YOLOX external/YOLOX` |
| [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT) | converting DanceTrack ReID weights (bundles FastReID) | `git clone https://github.com/GerardMaggiolino/Deep-OC-SORT external/Deep-OC-SORT` |

## Quick start: track with any detector

ARC-SORT takes boxes and scores from any detector and returns boxes with persistent IDs:

```python
from arcsort import ARCTracker

tracker = ARCTracker.from_preset("sportsmot")       # motion-only, runs on CPU
# tracker = ARCTracker.from_preset("sportsmot", reid_weights="weights/sportsmot/sports_osnet_traced.pt")

for frame in video:                                   # BGR numpy image
    dets = my_detector(frame)                         # (N, 6): x1, y1, x2, y2, score, class_id
    tracks = tracker.update(dets, frame)              # (M, 5): x1, y1, x2, y2, track_id
```

- **Presets**: `sportsmot` (recommended default for sports footage), `dancetrack`, and
  `sportsmot_deepeiou_inputs`. Override any key, e.g.
  `ARCTracker.from_preset("sportsmot", track_buffer=60, eiou={"adaptive_scale": 3.0})`.
- **`frame`** is used for camera-motion compensation and ReID crops. Pass it every frame
  (or `use_cmc=False` for a static camera).
- **ReID** needs the `reid` extra, a CUDA GPU and converted weights (see [Weights](#weights)).
  Without it the tracker runs motion-only.
- **Score thresholds**: the defaults (`track_high_thresh` 0.6, `new_track_thresh` 0.7) worked
  best for every detector we tested. Lowering them reduced HOTA for ARC-SORT and for the
  baselines alike. Run the detector itself with a low confidence floor (e.g. 0.05) so that BYTE's
  second stage sees low-score boxes.
- **Output boxes**: since 0.1.1 each track reports the detection box it was matched to this frame
  (`output_box="detection"`, the default). `output_box="kalman"` reports the filtered state
  estimate instead, which is what 0.1.0 and the arXiv v1 results used. The detection box scored
  higher in every setting we tested (see [Comparison with Roboflow trackers](#comparison-with-roboflow-trackers)).
- **supervision** users: `arcsort.interop.from_supervision(detections)` and
  `to_supervision(tracks)` convert to and from `sv.Detections` (`pip install arcsort[viz]`).
- **Trails with a moving camera**: plain trails also record the camera's motion. Use
  `Visualizer(motion_compensated_trails=True)` and pass `camera_motion=tracker.camera_motion` to
  `annotate()` to keep trails fixed to the scene (Roboflow `trackers`' `MotionAwareTraceAnnotator`,
  installed with `arcsort[viz]`), or `tools/pipeline.py --motion-trails`.
- Start a new video with `tracker.reset()`. Several trackers can run in one process (e.g. one per
  camera); each numbers its own tracks.

A detector factory is included for common backends (`pip install arcsort[yolo]` for Ultralytics;
RF-DETR and transformers are installed separately):

```python
from arcsort.detection import create_detector

detector = create_detector("yolo11x.pt", imgsz=1280, conf_thresh=0.05, classes=[0])
# also: "rfdetr-large", "PekingU/rtdetr_v2_r101vd", "rtdetr-x.pt" (use imgsz=640), YOLOX
dets = detector.detect(frame)    # (N, 6), class ids in COCO-80 order (person = 0)
```

### Tested detectors

ARC-SORT was run unchanged (`from_preset("sportsmot")` with live ReID and CMC) behind the
detectors below. The test used three SportsMOT validation clips (basketball, football, volleyball;
1,616 frames). Every tracker got identical detections. Baselines are the Roboflow
[`trackers`](https://github.com/roboflow/trackers) 2.6.1 implementations with default settings;
ARC-SORT is 0.1.1.

| Detector | Library | ByteTrack | BoT-SORT | ARC-SORT online | ARC-SORT + offline merge |
|---|---|---|---|---|---|
| YOLOv8x | Ultralytics | 57.0 | 59.4 | **63.1** | **67.1** |
| YOLOv10x | Ultralytics | 61.8 | **63.9** | 62.5 | **65.9** |
| YOLO11x | Ultralytics | 57.5 | **60.3** | 59.4 | **63.4** |
| YOLO12x | Ultralytics | 56.7 | 59.8 | **61.8** | **66.7** |
| YOLO26x | Ultralytics | 57.0 | 57.0 | **60.7** | **65.2** |
| YOLOv13x | [iMoonLab fork](https://github.com/iMoonLab/yolov13) | 57.2 | 58.6 | **61.1** | **62.5** |
| RT-DETRv2-R101 | transformers | 57.9 | 58.7 | **60.4** | **64.6** |
| RF-DETR-L | rfdetr | 65.2 | 65.9 | **68.6** | **72.6** |
| YOLOX-X (SportsMOT-trained) | YOLOX | 77.0 | 78.6 | **82.8** | **86.2** |

HOTA; bold marks the best online and best overall result per row. This is a compatibility check,
not a benchmark: three clips are too few to rank trackers. COCO detectors also detect referees
and spectators, who are not labelled in SportsMOT, so their absolute scores are low for every
tracker. Run Ultralytics RT-DETR at its training size (`imgsz=640`); at 1280 it detects only a
third of the players.

## Comparison with Roboflow trackers

Every tracker below got identical inputs. Baselines are Roboflow
[`trackers`](https://github.com/roboflow/trackers) 2.6.1 at default settings with each
sequence's real frame rate, the same protocol as their benchmark runner. ARC-SORT is 0.1.1 with
its stock preset. Online trackers only (no interpolation or offline merge), scored with TrackEval.

**SoccerNet-Tracking test (49 sequences), ground-truth boxes as input.** This is the split and
input Roboflow reports; our SORT, ByteTrack, OC-SORT and BoT-SORT runs match their published
numbers within 0.05 HOTA. ARC-SORT uses the `sportsmot` preset with no SoccerNet tuning.

| Tracker | HOTA | AssA | IDF1 | IDs |
|---|---|---|---|---|
| **ARC-SORT** | **86.1** | **75.8** | **81.0** | **1516** |
| BoT-SORT | 84.5 | 73.7 | 79.3 | 2941 |
| C-BIoU | 84.1 | 71.8 | 78.2 | 3700 |
| ByteTrack | 84.0 | 71.7 | 78.1 | 4016 |
| SORT | 81.6 | 69.7 | 76.2 | 2996 |
| OC-SORT | 78.4 | 65.0 | 72.5 | 3255 |

Roboflow reports McByte, which adds SAM/Cutie mask propagation, at 85.0 HOTA on this split; we did
not run it.

**Validation splits (HOTA).** "YOLOX-X" means the same cached detections for every tracker;
"GT boxes" means ground-truth boxes with confidence 1.0.

| Tracker | SportsMOT, YOLOX-X | DanceTrack, YOLOX-X | SportsMOT, GT boxes | DanceTrack, GT boxes |
|---|---|---|---|---|
| **ARC-SORT** | **79.8** | **59.1** | **90.1** | **84.7** |
| BoT-SORT | 76.5 | 53.5 | 87.9 | 79.9 |
| C-BIoU | 75.2 | 53.6 | 87.4 | 79.9 |
| ByteTrack | 75.0 | 50.5 | 87.5 | 80.1 |
| OC-SORT | 72.5 | 52.0 | 83.8 | 77.8 |
| SORT | 71.8 | 45.2 | 86.3 | 78.7 |

**Where it does not lead.** On MOT17 (pedestrians, mostly static cameras; Roboflow's val split,
identical YOLOX-X detections, nothing tuned) ARC-SORT is mid-pack at 70.8 HOTA, behind BoT-SORT
(72.5) and C-BIoU (72.2) and ahead of OC-SORT (70.3), ByteTrack (68.1) and SORT (61.4). Its gains
come from moving cameras and fast, non-linear motion.

## Reproducing the paper

One command per row:

```bash
scripts/reproduce.sh sportsmot val            # prints HOTA 84.957
scripts/reproduce.sh dancetrack val           # prints HOTA 60.617
scripts/reproduce.sh sportsmot test           # writes outputs/arcsort_sportsmot_test.zip
scripts/reproduce.sh dancetrack test          # writes outputs/arcsort_dancetrack_test.zip
scripts/reproduce.sh sportsmot_deepeiou test  # Deep-EIoU's inputs
```

These commands reproduce arXiv:2610.22306 v1 exactly, so the script defaults to
`OUTPUT_BOX=kalman`. With the 0.1.1 default (`OUTPUT_BOX=detection scripts/reproduce.sh ...`)
the validation rows become 86.295 (SportsMOT, from 84.957) and 61.329 (DanceTrack, from 60.617).

Upload test zips to the [SportsMOT](https://www.codabench.org/competitions/13077/) or
[DanceTrack](https://www.codabench.org/competitions/14885/) CodaBench server.

The script runs five stages and skips any stage whose output already exists:

1. **detections**: YOLOX-X, one file per sequence (GPU)
2. **camera motion**: one affine transform per frame (CPU)
3. **embeddings**: ReID features for every detection (GPU)
4. **tracking**: online tracker → interpolation (gaps ≤ 10 frames) → offline tracklet merge (CPU)
5. **evaluation** (val) or **packaging** (test)

### Fast path: released caches

The [v0.1.0 release](https://github.com/Athlitix/arcsort/releases/tag/v0.1.0) contains our cached
detections and camera-motion transforms (`arcsort-v0.1.0-caches.tar.gz`, ~40 MB). Extract it at
the repository root and stages 1–2 are skipped:

```bash
tar xzf arcsort-v0.1.0-caches.tar.gz      # creates cache/detections/ and cache/cmc/
```

```
cache/
  detections/{sportsmot,dancetrack}_{val,test}/<seq>.txt
  cmc/{sportsmot,dancetrack}_{val,test}/<seq>.npy
  embeddings/{sportsmot,dancetrack}_{val,test}/<seq>.npz   # computed by stage 3 (GPU)
```

The release also contains the result files we submitted to the evaluation servers
(`sportsmot_test.zip`, `dancetrack_test.zip`, `sportsmot_deepeiou_test.zip`). Put them in a folder,
e.g. `reference/`, and check your run against them:

```bash
scripts/regression_gate.sh cache reference
```

Detector and ReID outputs can differ in the last bits across GPU models and library versions, and
a few changed embeddings can flip a few association decisions. On other hardware, expect numbers
within noise of the paper, not byte-identical files. Our results were produced on an RTX 3060 Laptop
GPU with torch 2.5.1+cu121, numpy 2.4.3, opencv-python 4.13.0, scipy 1.17.1 and lap 0.5.13. On that
setup, every stage regenerates byte-identically from the raw upstream downloads.

### Datasets

Download [SportsMOT](https://github.com/MCG-NJU/SportsMOT) and
[DanceTrack](https://github.com/DanceTrack/DanceTrack) and arrange them in MOTChallenge layout:

```
data/
  SportsMOT/{val,test}/<seq>/{img1/,gt/,seqinfo.ini}
  DanceTrack/{val,test}/<seq>/{img1/,gt/,seqinfo.ini}
```

Frames are needed for stages 1–3 only. Val evaluation needs `gt/`.

### Weights

Weights are not redistributed here. Download them from the original authors:

| File in `weights/` | Used for | Source |
|---|---|---|
| `sportsmot/yolox_x_sports_train.pth.tar` | SportsMOT **val** detections (train-split checkpoint, so no val frame was seen) | [MixSort model zoo](https://github.com/MCG-NJU/MixSort) |
| `sportsmot/yolox_x_sports_mix.pth.tar` | SportsMOT **test** detections (train+val checkpoint) | [MixSort model zoo](https://github.com/MCG-NJU/MixSort) |
| `ocsort_dance_model.pth.tar` | DanceTrack detections | [OC-SORT model zoo](https://github.com/noahcao/OC_SORT) |
| `sportsmot/sports_osnet_traced.pt` | SportsMOT ReID | converted from Deep-EIoU's `sports_model.pth.tar-60` ([Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU)), see below |
| `dance_sbs_S50_traced.pt` | DanceTrack ReID | converted from FastReID `dance_sbs_S50.pth` ([Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT)), see below |

ReID conversion to TorchScript (one-time, needs a CUDA GPU):

```bash
# SportsMOT: sports-domain OSNet from Deep-EIoU
python tools/extract_state_dict.py --input sports_model.pth.tar-60 \
    --output weights/sportsmot/sports_osnet_clean.pth
python tools/convert_osnet_weights.py --input weights/sportsmot/sports_osnet_clean.pth \
    --output weights/sportsmot/sports_osnet_traced.pt

# DanceTrack: FastReID SBS-S50 (needs external/Deep-OC-SORT)
python tools/convert_reid_weights.py --input dance_sbs_S50.pth \
    --output weights/dance_sbs_S50_traced.pt --config dancetrack
```

`extract_state_dict.py` loads the training checkpoint with `weights_only=False`, so only run it
on files from the official release.

For the Deep-EIoU-inputs row, download Deep-EIoU's released SportsMOT test detections and
embeddings into `external/Deep-EIoU/{detection,embedding}/` (or set `DEEP_EIOU_ASSETS`).

## Repository layout

```
arcsort/          tracker package (IMM-KF, EIoU, CMC, ReID, arbitration, OC-SORT components)
arcsort/configs/  presets = the exact configurations behind each paper row
configs/detector/ YOLOX experiment files
tools/            caching, tracking, post-processing, evaluation, FPS benchmark, video demo
scripts/          reproduce.sh, regression_gate.sh
tests/            unit tests (pytest)
third_party/      vendored OSNet architecture (MIT)
docs/             project page (GitHub Pages)
```

## Citation

```bibtex
@article{adari2026arcsort,
  title   = {{ARC-SORT}: Adaptive, Robust, Compensated Multi-Object Tracking with Interacting Multiple Models},
  author  = {Adari, Girish Kumar},
  journal = {arXiv preprint arXiv:2610.22306},
  year    = {2026}
}
```

## License

Apache License 2.0; see [LICENSE](https://github.com/Athlitix/arcsort/blob/main/LICENSE) and [NOTICE](https://github.com/Athlitix/arcsort/blob/main/NOTICE). ARC-SORT builds on OC-SORT,
ByteTrack, BoT-SORT and Deep OC-SORT; see [THIRD_PARTY_NOTICES.md](https://github.com/Athlitix/arcsort/blob/main/THIRD_PARTY_NOTICES.md).
Pretrained weights and datasets are subject to their own licenses and terms, which may restrict
commercial use independently of this code's license.
