#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"

# Pass --smoke for a temporary validation-only exercise. Without --smoke the
# runner requires a clean Git tree and atomically creates release_v4_locked.
"${PYTHON_BIN}" -m confik.release_v4_locked.runner \
  --config configs/release_v4_locked.yaml "$@"
