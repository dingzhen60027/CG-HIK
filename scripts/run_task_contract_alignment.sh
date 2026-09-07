#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONFIK_PYTHON="${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"
case "${1:-}" in
  prepare|seal) exec "$CONFIK_PYTHON" -m confik.task_contract_alignment.study "$1" ;;
  points) exec "$CONFIK_PYTHON" -m confik.task_contract_alignment.point_study ;;
  sensitivity) exec "$CONFIK_PYTHON" -m confik.task_contract_alignment.sensitivity ;;
  trajectories) exec "$CONFIK_PYTHON" -m confik.task_contract_alignment.trajectory_study ;;
  aggregate) exec "$CONFIK_PYTHON" -m confik.task_contract_alignment.aggregate ;;
  reporting) exec "$CONFIK_PYTHON" -m confik.task_contract_alignment.reporting ;;
  *) echo 'Usage: run_task_contract_alignment.sh {prepare|seal|points|sensitivity|trajectories|aggregate|reporting}' >&2;exit 2 ;;
esac
