#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

PILOT_GATE="outputs/pilot_v2_4/ur5e/results/claim_gate_v2.json"
if [[ ! -f "${PILOT_GATE}" ]]; then
  echo "Refusing full run: first execute ./scripts/run_pilot_v2.sh" >&2
  exit 2
fi
if [[ "$(jq -r '.pilot_gate_pass' "${PILOT_GATE}")" != "true" ]]; then
  echo "Refusing full run: the preregistered v2 pilot gate did not pass." >&2
  exit 3
fi

for robot in ur5e panda; do
  "${PYTHON_BIN}" -m confik run-repetitions-v2 \
    --config configs/paper_v2.yaml \
    --robot "${robot}" \
    --seeds 17 29 43
done

"${PYTHON_BIN}" -m confik aggregate-v2 \
  --config configs/paper_v2.yaml \
  --robots ur5e panda \
  --seeds 17 29 43
