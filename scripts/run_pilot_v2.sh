#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

"${PYTHON_BIN}" -m confik run-all-v2 \
  --config configs/pilot_v2.yaml \
  --robot ur5e

