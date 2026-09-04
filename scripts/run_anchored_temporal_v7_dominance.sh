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
  echo "No smoke mode is allowed because the development holdout is one-shot." >&2
  exit 2
fi

export PYTHONPATH="${WORKSPACE}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${WORKSPACE}"
exec "${PYTHON_BIN}" -m confik.anchored_temporal_v7.dominance_pilot \
  --config "${WORKSPACE}/configs/anchored_temporal_v7_dominance.yaml"
