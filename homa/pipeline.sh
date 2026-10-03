#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/homatracker-exp042
DATA=/root/autodl-tmp/UAVSwarmV2_MOT
PY="$ROOT/venv/bin/python"
CONFIG="$ROOT/inputs/protocol_v1.json"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -m homa.run --run "$ROOT/results/run_001" --phase train --config "$CONFIG" -- \
  "$PY" -m homa.train --data "$DATA" --config "$CONFIG" --output "$ROOT/results/run_001/model" --tensorboard /root/tf-logs/exp042/run_001
"$PY" -m homa.sweep --root "$ROOT" --data "$DATA" --config "$CONFIG" --checkpoint "$ROOT/results/run_001/model/best.pt" --output "$ROOT/validation"
"$PY" -m homa.sweep --root "$ROOT" --data "$DATA" --config "$CONFIG" --checkpoint "$ROOT/results/run_001/model/best.pt" --output "$ROOT/test" --selection "$ROOT/validation/selection.json"
