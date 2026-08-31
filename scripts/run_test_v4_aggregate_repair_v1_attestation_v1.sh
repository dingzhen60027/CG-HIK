#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

if [[ -n "${GIT_DIR:-}" || -n "${GIT_WORK_TREE:-}" ]]; then
  echo "attestation forbids GIT_DIR/GIT_WORK_TREE overrides" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"

exec "${PYTHON_BIN}" -I \
  "${WORKSPACE}/src/confik/test_v4_locked/aggregate_repair_attestation.py" \
  --config "${WORKSPACE}/configs/test_v4_aggregate_repair_v1_attestation_v1.json"
