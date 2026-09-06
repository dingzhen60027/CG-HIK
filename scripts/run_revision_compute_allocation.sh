#!/usr/bin/env bash
set -euo pipefail
REV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="${REV_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${REV_ROOT}/tmp/revision_dependencies/prefix/usr/lib/x86_64-linux-gnu:/opt/ros/humble/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${REV_ROOT}"
exec /home/eric/anaconda3/envs/isaaclab_3/bin/python -m confik.revision_compute_allocation.runner "$@"
