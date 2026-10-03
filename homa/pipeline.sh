#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/homatracker-exp042
DATA=/root/autodl-tmp/UAVSwarmV2_MOT
PY="$ROOT/venv/bin/python"
CONFIG="${HOMA_CONFIG:-$ROOT/inputs/protocol_v1.json}"
FIRST_RUN="${HOMA_FIRST_RUN:-1}"
printf -v TRAIN_RUN 'run_%03d' "$FIRST_RUN"
VALIDATION="$ROOT/validation_$TRAIN_RUN"
TEST="$ROOT/test_$TRAIN_RUN"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -m homa.run --run "$ROOT/results/$TRAIN_RUN" --phase train --config "$CONFIG" --parent "${HOMA_PARENT_RUN:-none}" -- \
  "$PY" -m homa.train --data "$DATA" --config "$CONFIG" --output "$ROOT/results/$TRAIN_RUN/model" --tensorboard "/root/tf-logs/exp042/$TRAIN_RUN"
"$PY" -m homa.sweep --root "$ROOT" --data "$DATA" --config "$CONFIG" --checkpoint "$ROOT/results/$TRAIN_RUN/model/best.pt" --output "$VALIDATION" --first-run "$((FIRST_RUN + 1))" --parent "$TRAIN_RUN"
if [ "${HOMA_RUN_TEST:-1}" = 1 ]; then
  VALIDATION_COUNT=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$VALIDATION/metrics.json")
  "$PY" -m homa.sweep --root "$ROOT" --data "$DATA" --config "$CONFIG" --checkpoint "$ROOT/results/$TRAIN_RUN/model/best.pt" --output "$TEST" --selection "$VALIDATION/selection.json" --first-run "$((FIRST_RUN + 1 + VALIDATION_COUNT))" --parent "$TRAIN_RUN"
fi
