#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"

"${PYTHON_BIN}" -m confik.release_v3_locked.runner \
  --config configs/release_v3_locked.yaml
