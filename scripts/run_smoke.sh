#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
"${PYTHON_BIN}" -m confik run-all --config configs/smoke.yaml --robot toy --force

