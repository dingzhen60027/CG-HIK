# Task Balance GN — single-core completion prototype

## Purpose

Continue the existing two-block position/orientation minimax solver. Keep its task, current-frame bounds, lack of future information, and no-fallback structure. Add a verifiable *relative local-model decrease* stopping condition and remove duplicate residual/verification work. See METHOD.md for the mathematical definition and its limited guarantee.

## Actual results from this conversation

All results are **local, exploratory, Panda-only**. They are not original-server measurements or independent test results. All variants use the same warmed Numba local geometry/box-QP and the same independent Python FK acceptance check. No TRAC, Pink, Clarabel or UR5e performance was measured here. Setup/JIT is outside timing; the selected final verifier is inside timing. Old local balance performs more verifier calls by design; cached tight and relative both perform one final external verification and share the same implementation.

### Original 40 reconstructed development trajectories; one sweep per variant

| Variant (kappa=1) | Complete | Deadline complete | Cumulative ms | P50/P95/P99 ms |
|---|---:|---:|---:|---|
| Frozen GN reference |40|40|1738.243|0.2691 / 0.3557 / 0.5024|
| Previous uploaded LOCAL balance reference |40|40|4009.239|0.6316 / 0.8910 / 1.2980|
| Cached, tight-gap balance |40|40|1813.709|0.2775 / 0.4365 / 0.6717|
| Cached, relative-decrease balance |40|40|1778.814|0.2730 / 0.4141 / 0.5868|

These single-pass timing differences have no inferential confidence interval. The large reduction versus the old local reference is mostly removal of repeated work, not solely the proposed stopping test. Relative vs cached tight is the proper same-core algorithmic comparison.

### Previously observed 160-Panda generator reconstructed locally

The generator recipe and seed base were read from the current GitHub branch. No seed collision is assumed. One known UID/target was checked against the historical selected input (position difference 5.55e-17 m, rotation element difference 1.67e-16). **All-array comparison with the repository inputs remains required.** These inputs were already used in earlier research and are not fresh for the new method.

| Variant (kappa=1) | Complete | Deadline complete | Cumulative ms | Local weighted-QP calls |
|---|---:|---:|---:|---:|
| Frozen GN reference |154|154|7414.000|not recorded in same unit|
| Cached, tight-gap balance |154|154|10311.719|58901|
| Cached, relative-decrease balance |154|154|9636.748|48740|

Relative stopping reduced weighted-QP calls by 17.25% and measured total cost by 6.55% versus cached tight. It still costs approximately 30.0% more than the GN reference and recovers **no additional complete trajectory**. Some first failures move later, not into full success. Do not hide this result or claim general speed/continuation gains.

An initial combined command hit the container time limit during the relative run, before a result file was written. A subsequent separate relative run completed. Other variants were not replaced by faster repeats. See PROVENANCE.json.

### Known conflict input

On the previously examined trajectory_094/frame53 input, the local frozen GN remains inadmissible. The relative prototype independently returns a locally admissible joint vector with position error 0.973170585 mm and angular error 0.008042243803 rad (about 0.4608 deg). Three calls agree geometrically. This is a selected mechanism input, not a general recovery rate. The exact input and vectors are in results/selected.json and still need repository-verifier confirmation.

### Mathematical checks

48 reproducible convex local problems were compared with an independent cold-start SciPy SLSQP epigraph solver. All reference outputs passed primal feasibility checks, lower/upper bounds were consistent within the numerical checking tolerance, and the proposed relative stop achieved the stated >=80% local-model decrease fraction against the independent solution. Mean weighted-QP count was 1.9167. This does not establish nonlinear/global convergence or runtime superiority. The external-rejection regression confirms that a locally small error alone never grants final acceptance.

## Contents

- inexact_balance.py: executable generic core, exact and relative stops in one implementation;
- check_and_run.py: mathematical, selected-input and local full-path checks;
- reconstruct_observed.py: transparent prior-generator reconstruction (not a new test);
- reference/: existing uploaded local numerical references; no new reference tuning;
- METHOD.md: equations, proof and prior-art boundaries;
- results/: complete local outputs, not just best cases;
- CODEX_TASK.md: original-environment integration and finite evaluation task;
- PROVENANCE.json: environments, selection and interruption details.

Run from this directory with NumPy/SciPy/Numba available and one BLAS thread:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python check_and_run.py math
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python check_and_run.py selected
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python check_and_run.py relative --kappa 1 --repeat 1
```

Saved output is not overwritten. Run in a copied empty output folder for a full deliberate reproduction. When integrating, import the repository's current geometry and box QP; do not replace the server with the local Panda model.

## Current scientific status

This package completes a concrete algorithm candidate and a local model-progress property. It does **not** complete the user's requested paper-level validation of algorithmic innovation. It is not evidence of a new minimax principle, a higher independent TSR, low-cost UR5e performance, or an advantage over a same-goal mature conic solver. The scope for the next task is to quantify current-query recovery and the extra cost under one implementation and to preserve whole-trajectory outcomes as a required deployment check, not to rebrand an absent whole-trajectory gain.
