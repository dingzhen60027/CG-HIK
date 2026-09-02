#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="/home/eric/anaconda3/envs/isaaclab_3/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Required Python environment is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="${WORKSPACE}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${WORKSPACE}"
exec "${PYTHON_BIN}" -m confik.hierarchical_v5_lite.pilot \
  --config "${WORKSPACE}/configs/hierarchical_v5_lite_pilot.yaml" "$@"
