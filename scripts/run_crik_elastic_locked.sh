#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="tmp/crik_dependencies/python:src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
CRIK_PYTHON="/home/eric/anaconda3/envs/isaaclab_3/bin/python"
case "${1:-check}" in
  prepare|check) exec "$CRIK_PYTHON" -m confik.correction_reserve.locked_study "$1" ;;
  panda) exec taskset -c 0,2 "$CRIK_PYTHON" -m confik.correction_reserve.locked_study run --robot panda ;;
  ur5e) exec taskset -c 4,6 "$CRIK_PYTHON" -m confik.correction_reserve.locked_study run --robot ur5e ;;
  *) echo 'Use prepare, check, panda, or ur5e. Commit prepared identities before run.' >&2; exit 2 ;;
esac
