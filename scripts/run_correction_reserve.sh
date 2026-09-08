#!/usr/bin/env bash
set -euo pipefail
CRIK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRIK_ROOT"
CRIK_PYTHON="${CRIK_PYTHON:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CRIK_ROOT/tmp/crik_dependencies/python:$CRIK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CRIK_STAGE="${1:-help}"
case "$CRIK_STAGE" in
  test)
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CRIK_RANGED_NATIVE_SMOKE=1 "$CRIK_PYTHON" -m pytest -q \
      tests/test_correction_reserve.py tests/test_crik_pink.py tests/test_crik_ranged.py tests/test_crik_study.py
    ;;
  prepare)
    "$CRIK_PYTHON" -m confik.correction_reserve.data
    ;;
  formal)
    "$CRIK_PYTHON" -m confik.correction_reserve.seal_check
    taskset -c 0,2 "$CRIK_PYTHON" -m confik.correction_reserve.study \
      --folder outputs/correction_reserve_ik/formal/panda --robot panda \
      --inputs outputs/correction_reserve_ik/formal_protocol/online_targets.json &
    CRIK_PANDA_PID=$!
    taskset -c 4,6 "$CRIK_PYTHON" -m confik.correction_reserve.study \
      --folder outputs/correction_reserve_ik/formal/ur5e --robot ur5e \
      --inputs outputs/correction_reserve_ik/formal_protocol/online_targets.json &
    CRIK_UR5E_PID=$!
    CRIK_STATUS=0
    wait "$CRIK_PANDA_PID" || CRIK_STATUS=1
    wait "$CRIK_UR5E_PID" || CRIK_STATUS=1
    exit "$CRIK_STATUS"
    ;;
  report)
    "$CRIK_PYTHON" -m confik.correction_reserve.reporting \
      --folders outputs/correction_reserve_ik/formal/panda outputs/correction_reserve_ik/formal/ur5e \
      --out outputs/correction_reserve_ik/formal_reports
    ;;
  *)
    echo 'Stages: test, prepare, formal, report. Existing output directories are never overwritten.'
    ;;
esac
