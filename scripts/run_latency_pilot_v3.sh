#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

"${PYTHON_BIN}" -m confik.latency_pilot_v3.runner \
  --config configs/latency_pilot_v3.yaml
