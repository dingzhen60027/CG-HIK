#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="tmp/crik_dependencies/python:src${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
CRIK_PYTHON="${CRIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}"
case "${1:-}" in
  prepare) "$CRIK_PYTHON" -m confik.correction_reserve.minimal_study prepare ;;
  test) "$CRIK_PYTHON" -m pytest -q tests/test_crik_minimal.py ;;
  panda) taskset -c 0,2 "$CRIK_PYTHON" -m confik.correction_reserve.minimal_study run --robot panda ;;
  ur5e) taskset -c 4,6 "$CRIK_PYTHON" -m confik.correction_reserve.minimal_study run --robot ur5e ;;
  check) "$CRIK_PYTHON" -m confik.correction_reserve.minimal_study check ;;
  report) "$CRIK_PYTHON" -m confik.correction_reserve.minimal_reporting ;;
  *) printf '%s\n' 'usage: run_crik_minimal.sh {prepare|test|panda|ur5e|check|report}' >&2; exit 2 ;;
esac
