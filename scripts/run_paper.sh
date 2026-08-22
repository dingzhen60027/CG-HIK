#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

for robot in ur5e panda; do
  "${PYTHON_BIN}" -m confik run-repetitions \
    --config configs/paper.yaml \
    --robot "${robot}" \
    --seeds 17 29 43
done
