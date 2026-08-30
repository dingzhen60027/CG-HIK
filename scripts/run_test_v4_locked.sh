#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="${WORKSPACE}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "$#" -gt 1 ]] || [[ "$#" -eq 1 && "$1" != "--resume" ]]; then
  echo "usage: $0 [--resume]" >&2
  exit 2
fi

exec "${PYTHON_BIN}" -m confik.test_v4_locked.runner \
  --config "${WORKSPACE}/configs/test_v4_locked.yaml" \
  "$@"
