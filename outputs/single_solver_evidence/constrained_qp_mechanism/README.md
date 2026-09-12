# Real constrained-QP follow-up

This directory is separate from frozen seven-setting evidence. No new online
trajectory evaluation was run. See the [findings](../../../docs/SINGLE_SOLVER_CONSTRAINED_QP_FINDINGS.md)
and [claim list](../../../docs/SINGLE_SOLVER_CLAIM_EVIDENCE.md).
The report links above are repository-relative from this directory: the complete
paths are `docs/SINGLE_SOLVER_CONSTRAINED_QP_FINDINGS.md` and
`docs/SINGLE_SOLVER_CLAIM_EVIDENCE.md` at repository root.

## Contents

- `capture/`: every captured H/g/lo/hi/d/q/step/e in NPZ; chronological JSONL,
  frame input-replay discrepancies, selected indices, source SHA-256 hashes.
  Each input is the saved GN repeat-0 previous state; this is not a new closed loop.
- `benchmark/`: original 20000-cap OSQP comparison; retained quality failures.
- `benchmark_quality_cap/`: final **common timing round for all four methods**,
  only offline OSQP cap increased to 200000 for precision. Three Panda inputs
  still fail common quality; never counted as faster qualified solutions.
- Both rounds contain raw and returned directions, native statuses, timing
  phases, KKT/box/objective checks; `classification.jsonl.gz` includes actual
  unmodified-function active-set trace and original source-sequence changes.
- `mechanism/`: six same-input direction/FK/backtracking comparisons, actual
  historical commands and outcomes. Trial variables failing verification are
  not accepted commands or feasible witnesses. Future original outcomes are
  read-only associations, not new counterfactual rollouts.
- `reports/`: complete quality/time strata and UID-paired descriptive intervals,
  replay census, setup and local-to-trajectory tables, one editable SVG/PDF figure
  with PNG preview. No natural/targeted-cohort pooling.
- `qpoases_3.2.1_source.tar.gz`: unmodified official source at the pinned SHA;
  contains upstream LGPL-2.1-or-later license and notices. The repository bridge
  is separately provided, so the measured library can be rebuilt/relinked.

## Fixed environment and commands

Use the existing Python/NumPy environment, not a new accelerated backend:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=tmp/crik_dependencies/python:src
```

Official source (the archive is an offline alternative to cloning):

```bash
git clone --depth 1 --branch releases/3.2.1 https://github.com/coin-or/qpOASES.git tmp/single_solver_qpoases
git -C tmp/single_solver_qpoases rev-parse HEAD
# Required: 51d3fbea30142d3acbf40cf7a1c519efc27ea67b
cmake -S tmp/single_solver_qpoases -B tmp/single_solver_qpoases/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DQPOASES_BUILD_EXAMPLES=OFF
cmake --build tmp/single_solver_qpoases/build -j 2
c++ -O3 -DNDEBUG -std=c++11 -fPIC -shared -I tmp/single_solver_qpoases/include scripts/native/qpoases_box_reference.cpp tmp/single_solver_qpoases/build/libs/libqpOASES.a -o tmp/single_solver_qpoases/build/libbox_reference.so
```

The single entry is `scripts/run_constrained_qp_mechanism.py`. The executed order
was provenance → capture --robot panda → capture --robot ur5e → mechanism →
benchmark → benchmark --tag benchmark_quality_cap --osqp-max-iter 200000 → report.
Output creation is exclusive; these commands are **not an instruction to rerun
or overwrite the frozen delivery**. Python used the existing `isaaclab_3`
environment, and the entry fixes affinity to CPU4. Setup and all measured phases
are retained; benchmark timing includes independent common quality checks.

The only cap adjustment followed the recorded KKT failures, not trajectory
success or speed. `quality_followup.md` preserves that decision. No main solver
settings were changed. Source provenance at capture start and final delivery
hashes are both recorded; plotting-only layout edits did not rerun numerical data.
