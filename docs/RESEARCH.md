# Research record: task-contract-aware online inverse kinematics

Updated: 2026-09-08.
Research baseline: `02aa287b65d4f1b7bb5317d8e83fcf972559103f`.
Final measurement delivery: `089a3fe7ec0456002aa0f3b9661119b99072fad7`.

## Permanent research question

How should an online inverse-kinematics solver align its internal convergence
conditions with the task-level command contract under pose, joint-limit and
joint-rate constraints?

**Solver convergence is not command admissibility.**

The application requires a finite joint command satisfying specified pose and
motion limits, rather than an arbitrarily precise root. Native convergence, public
command admissibility and admissibility within a deadline are separate outcomes.
A saved legal configuration proves a missed solution only for the identical
target, previous accepted state, dt and contract.

## Engineering principle and implemented interface

Align the solver with the task contract before introducing adaptive computation.

The interface maps public pose tolerances conservatively to native conditions,
intersects URDF limits with the actual previous state's single-frame rate interval,
calls an existing numerical solver, and independently verifies its returned
configuration. The two implementations are official TRAC-IK and the existing
Adaptive DLS. Neither is a new solver. The component box used by TRAC-IK is not
equivalent to the public norm ball.

The nominal contract and original verifier are unchanged. Dynamic intervals use the
previous accepted state, not the reference path. A failed frame holds that state
while the target index advances. A verified reference path establishes one possible
continuous path, not the feasibility of every solver-induced state.

## Final evidence hierarchy

| Evidence | Status and role |
|---|---|
| `outputs/task_contract_alignment/02_point_study/` | Fixed previously observed witnessed/inexecutable points on both robots; separate native, task and deadline outcomes |
| `outputs/task_contract_alignment/03_contract_sensitivity/` | Same preselected witnessed points across half, nominal and double pose tolerances |
| `outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/` | Authoritative earlier Panda trajectories, read directly, never remeasured |
| `outputs/task_contract_alignment/04_trajectory_study/` | Symmetric prespecified UR5e trajectories and complete online search repeats |
| `outputs/task_contract_alignment/05_aggregate/` | Paired query/trajectory statistics, families, completion UIDs, errors, traces and independent replay |
| `outputs/revision_compute_allocation/` | Historical secondary allocation evidence; not a new router-on-aligned-solver experiment |
| Older fresh/formal and mechanism outputs | Preserved under their original protocols, including unfavorable results and old gates |

The final experiment report is `TASK_CONTRACT_ALIGNMENT_FINDINGS.md`. Manuscript
values are generated from these artifacts, not copied into this research record.

## What the evidence supports

Native-failure/admissible-return disagreement is prominent in strict DLS. Strict
TRAC already succeeds throughout the nominal witnessed-point primary comparison;
the point study therefore supports a computational difference, not recovery of
strict TRAC misses. Full TRAC alignment improves continuous completion and
cumulative computation on both robots, with family-dependent effects.

Task-level DLS stopping reduces point computation and preserves point acceptance,
but the UR5e trajectory completion loss shows that earliest legal acceptance is not
a universally better closed-loop strategy. Residual accuracy and the subsequent
configuration evolve together; this comparison does not identify a unique cause.

Historical learned routing does not generally accelerate successful solving.
Geometry and simpler controls are competitive; reject is useful when unsuccessful
numerical work is expensive. The study does not directly measure a learned router
added to the newly aligned native solvers.

## Final manuscript and stopping decision

`paper/main.tex` / `paper/main.pdf` are now organized around the contract interface,
not learned routing. See `FINAL_PAPER_CLAIM_MAP.md`,
`TASK_CONTRACT_PAPER_GUIDE.md` and `paper/README.md`.

Research development ends here. Do not add new solvers, networks, routing
thresholds, configuration scores, preview optimization, periodic anchors, new
robot populations, OOD, collisions, contacts or hardware experiments. Unequal
robot/workload effects remain part of the result. Subsequent work is limited to
author-supplied submission metadata and editorial preparation without changing
the evidence or this research question.

The prior routing-led research record is preserved in
`history/task_contract_predecessor_02aa287.tar.gz`.
