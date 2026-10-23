#!/usr/bin/env bash
# Regression gate: rerun the paper's tracking chain on cached inputs and check that the
# output is byte-identical to the result files submitted to the evaluation servers.
#
# Chain (all paper rows): online tracker -> interpolate (gap 10) -> offline tracklet merge
# (cos 0.85, gap 300, motion gate off, split).
#
# Usage:
#   scripts/regression_gate.sh CACHE_ROOT REF_ROOT [row ...]
#
#   CACHE_ROOT  cache layout used by scripts/reproduce.sh:
#               detections/<row>_test/, cmc/<dataset>_test/, embeddings/<row>_test/
#   REF_ROOT    directory with the submitted result zips from the GitHub release
#               (sportsmot_test.zip, dancetrack_test.zip, sportsmot_deepeiou_test.zip)
#   row         any of: sportsmot dancetrack sportsmot_deepeiou (default: all)
#
# Embeddings computed on a different GPU or library version can differ in the last
# bits, which can change a handful of association decisions; byte-identity is only
# guaranteed with the same embedding cache.
#
# Env: PYTHON (default: python), WORK (default: mktemp -d)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,22p' "$0"; exit 2
fi
CACHE=$(realpath "$1"); REF=$(realpath "$2"); shift 2
ROWS=("${@:-sportsmot dancetrack sportsmot_deepeiou}")
ROWS=(${ROWS[@]})
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
WORK=${WORK:-$(mktemp -d)}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# row -> config | detections | cmc cache | embeddings | reference zip
declare -A ROW=(
    [sportsmot]="sportsmot.yaml|sportsmot_test|sportsmot_test|sportsmot_test|sportsmot_test.zip"
    [dancetrack]="dancetrack.yaml|dancetrack_test|dancetrack_test|dancetrack_test|dancetrack_test.zip"
    [sportsmot_deepeiou]="sportsmot_deepeiou_inputs.yaml|sportsmot_deepeiou_test|sportsmot_test|sportsmot_deepeiou_test|sportsmot_deepeiou_test.zip"
)

failed=0
for row in "${ROWS[@]}"; do
    IFS='|' read -r cfg det cmc emb zip <<< "${ROW[$row]:?unknown row: $row}"
    out="$WORK/$row"
    rm -rf "$out"; mkdir -p "$out"
    echo "[$row] online tracker ($cfg)"
    "$PY" "$REPO/tools/run_cached.py" --config "$REPO/arcsort/configs/$cfg" \
        --detections "$CACHE/detections/$det" --cmc-cache "$CACHE/cmc/$cmc" \
        --emb-cache "$CACHE/embeddings/$emb" --output "$out/online" \
        --output-box kalman > "$out/online.log" 2>&1
    echo "[$row] interpolate"
    "$PY" "$REPO/tools/interpolate_tracks.py" --input "$out/online" --output "$out/interp" \
        --max-gap 10 > "$out/interp.log" 2>&1
    echo "[$row] merge"
    "$PY" "$REPO/tools/merge_tracklets.py" --input "$out/interp" \
        --emb-cache "$CACHE/embeddings/$emb" --output "$out/final" \
        --merge-thresh 0.85 --max-gap 300 --motion-tol inf --split > "$out/merge.log" 2>&1

    unzip -qo "$REF/$zip" -d "$out/ref"
    n=0; bad=0
    while IFS= read -r -d '' f; do
        n=$((n + 1))
        cmp -s "$f" "$out/final/$(basename "$f")" || { bad=$((bad + 1)); echo "  DIFF $(basename "$f")"; }
    done < <(find "$out/ref" -name '*.txt' -print0)
    if [[ $n -gt 0 && $bad -eq 0 ]]; then
        echo "[$row] PASS: $n/$n sequences byte-identical to $zip"
    else
        echo "[$row] FAIL: $bad of $n sequences differ from $zip"; failed=1
    fi
done
echo "work dir: $WORK"
exit $failed
