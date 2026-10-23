#!/usr/bin/env bash
# Reproduce the ARC-SORT paper results (arXiv:2610.22306).
#
# Usage:
#   scripts/reproduce.sh BENCHMARK SPLIT
#
#   BENCHMARK  sportsmot | dancetrack | sportsmot_deepeiou (test only; Deep-EIoU's inputs)
#   SPLIT      val  -> tracks + evaluates locally with TrackEval (prints HOTA)
#              test -> tracks + writes the zip to upload to the CodaBench server
#
# Stages (each one is skipped when its output already exists, so the released
# caches make stages 1-3 free):
#   1. detections   YOLOX-X, cached per sequence           (GPU; needs external/YOLOX)
#   2. cmc          camera-motion transforms                (CPU; needs frames)
#   3. embeddings   ReID features for every detection       (GPU; needs frames)
#   4. track        online tracker -> interpolate -> offline tracklet merge (CPU)
#   5. evaluate (val) or package (test)
#
# Env overrides: DATA (default: data), CACHE (default: cache), OUT (default: outputs),
#                PYTHON (default: python), DEEP_EIOU_ASSETS (default: external/Deep-EIoU),
#                OUTPUT_BOX (default: kalman, which reproduces arXiv:2610.22306 v1;
#                "detection" is the library default since 0.1.1)
set -euo pipefail

usage() { sed -n '2,22p' "$0"; exit 2; }
[[ $# -eq 2 ]] || usage
BENCH=$1; SPLIT=$2
[[ $SPLIT == val || $SPLIT == test ]] || usage

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
DATA=$(realpath -m "${DATA:-data}"); CACHE=$(realpath -m "${CACHE:-cache}")
OUT=$(realpath -m "${OUT:-outputs}"); PY=${PYTHON:-python}
DEEP_EIOU_ASSETS=${DEEP_EIOU_ASSETS:-external/Deep-EIoU}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

case $BENCH in
    sportsmot)
        CONFIG=arcsort/configs/sportsmot.yaml; DATASET=SportsMOT; EVAL=sportsmot
        EXP=configs/detector/yolox_x_sportsmot.py
        # Val runs use the train-only checkpoint so the detector never saw a val frame.
        if [[ $SPLIT == val ]]; then DET_WEIGHTS=weights/sportsmot/yolox_x_sports_train.pth.tar
        else DET_WEIGHTS=weights/sportsmot/yolox_x_sports_mix.pth.tar; fi ;;
    dancetrack)
        CONFIG=arcsort/configs/dancetrack.yaml; DATASET=DanceTrack; EVAL=dancetrack
        EXP=configs/detector/yolox_x_dance.py
        DET_WEIGHTS=weights/ocsort_dance_model.pth.tar ;;
    sportsmot_deepeiou)
        [[ $SPLIT == test ]] || { echo "Deep-EIoU released test-split inputs only"; exit 2; }
        CONFIG=arcsort/configs/sportsmot_deepeiou_inputs.yaml; DATASET=SportsMOT; EVAL=sportsmot ;;
    *) usage ;;
esac

FRAMES=$DATA/$DATASET/$SPLIT
NAME=${BENCH}_${SPLIT}
DETS=$CACHE/detections/$NAME
CMC=$CACHE/cmc/${DATASET,,}_$SPLIT
EMBS=$CACHE/embeddings/$NAME
RUN=$OUT/$NAME

have() { [[ -d $1 ]] && [[ -n $(ls -A "$1" 2>/dev/null) ]]; }
step() { echo; echo "== $*"; }
need_frames() {
    have "$FRAMES" || { echo "Frames not found at $FRAMES (see README: Datasets)"; exit 1; }
}

# 1-3. Inputs ---------------------------------------------------------------
if [[ $BENCH == sportsmot_deepeiou ]]; then
    if ! have "$DETS" || ! have "$EMBS"; then
        step "convert Deep-EIoU released detections + embeddings"
        need_frames
        "$PY" tools/convert_deep_eiou.py --assets "$DEEP_EIOU_ASSETS" \
            --out-dets "$DETS" --out-embs "$EMBS" --seqinfo-root "$FRAMES"
    fi
else
    if ! have "$DETS"; then
        step "detections ($DET_WEIGHTS)"
        need_frames
        [[ -f $DET_WEIGHTS ]] || { echo "Missing $DET_WEIGHTS (see README: Weights)"; exit 1; }
        PYTHONPATH="$PYTHONPATH:$REPO/external/YOLOX" "$PY" tools/cache_detections.py \
            --exp-file "$EXP" --weights "$DET_WEIGHTS" --data "$FRAMES" --output "$DETS" \
            --test-size 800 1440 --fp16
    fi
    if ! have "$EMBS"; then
        step "embeddings"
        need_frames
        "$PY" tools/cache_embeddings.py --detections "$DETS" --frames "$FRAMES" \
            --config "$CONFIG" --output "$EMBS"
    fi
fi
if ! have "$CMC"; then
    step "camera-motion compensation"
    need_frames
    "$PY" tools/cache_cmc.py --frames "$FRAMES" --output "$CMC" --config "$CONFIG"
fi

# 4. Track ------------------------------------------------------------------
rm -rf "$RUN"
step "online tracker ($CONFIG)"
"$PY" tools/run_cached.py --config "$CONFIG" --detections "$DETS" \
    --cmc-cache "$CMC" --emb-cache "$EMBS" --output "$RUN/online" \
    --output-box "${OUTPUT_BOX:-kalman}"
step "interpolate (max gap 10)"
"$PY" tools/interpolate_tracks.py --input "$RUN/online" --output "$RUN/interp" --max-gap 10
step "offline tracklet merge"
"$PY" tools/merge_tracklets.py --input "$RUN/interp" --emb-cache "$EMBS" \
    --output "$RUN/final" --merge-thresh 0.85 --max-gap 300 --motion-tol inf --split

# 5. Evaluate or package ----------------------------------------------------
if [[ $SPLIT == val ]]; then
    step "evaluate with TrackEval"
    seqmap=$DATA/$DATASET/${SPLIT}_seqmap.txt
    if [[ ! -f $seqmap ]]; then
        { echo name; ls "$FRAMES"; } > "$seqmap"
    fi
    "$PY" tools/evaluate.py --benchmark "$EVAL" --split "$SPLIT" --gt "$FRAMES" \
        --trackers "$RUN/final" > "$RUN/eval.log" 2>&1 || { tail -20 "$RUN/eval.log"; exit 1; }
    summary=$RUN/final/pedestrian_summary.txt
    "$PY" - "$summary" <<'EOF'
import sys
names, values = (line.split() for line in open(sys.argv[1]).read().splitlines()[:2])
row = dict(zip(names, values))
print("  ".join(f"{k} {row[k]}" for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")))
EOF
else
    step "package for the evaluation server"
    zip=$OUT/arcsort_${NAME}.zip
    rm -f "$zip"
    if [[ $BENCH == dancetrack ]]; then
        # DanceTrack expects results under tracker/
        stage=$RUN/zip/tracker; mkdir -p "$stage"; cp "$RUN"/final/*.txt "$stage"/
        (cd "$RUN/zip" && zip -qr "$zip" tracker)
    else
        (cd "$RUN/final" && zip -q "$zip" ./*.txt)
    fi
    echo "Upload $zip to CodaBench."
fi
