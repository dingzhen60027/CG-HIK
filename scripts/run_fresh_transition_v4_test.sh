#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="/home/eric/anaconda3/envs/isaaclab_3/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Required Python environment is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  echo "This final fresh evaluation has no smoke, resume, or rerun mode." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"
export OPENBLAS_NUM_THREADS="8"
export PYTHONPATH="${WORKSPACE}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${WORKSPACE}"
exec "${PYTHON_BIN}" -m confik.fresh_transition_v4_test.runner \
  --config "${WORKSPACE}/configs/fresh_transition_v4_test.yaml"
