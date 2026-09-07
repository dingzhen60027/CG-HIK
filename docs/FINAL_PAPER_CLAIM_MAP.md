# Final paper claim map: task-contract-aware online IK

Updated: 2026-09-08. Frozen experimental delivery:
`089a3fe7ec0456002aa0f3b9661119b99072fad7`.
This document replaces the routing-led claim map, whose exact predecessor is retained
in `docs/history/task_contract_predecessor_02aa287.tar.gz`.

## One claim, three contributions

Solver convergence is not command admissibility. Align the solver with the task
contract before introducing adaptive computation.

1. Separate native convergence, command admissibility and deadline-qualified
   admissibility, including exact-input witness-confirmed misses.
2. Formulate a common interface: conservative native tolerance mapping, dynamic
   joint/rate bounds, numerical solve, independent verification. Neither TRAC-IK
   bounds nor DLS is claimed as a new algorithm.
3. Quantify the effects on two robots across witnessed points, contract scales and
   full closed-loop trajectories; use frozen learned allocation as a secondary
   empirical boundary study.

## Claims and authoritative evidence

All current CSV names below are under `paper/source_data/`; they are generated
copies of the frozen outputs, not separately maintained measurements.

| Claim / question | Frozen evidence | Paper source data and presentation | Necessary interpretation |
|---|---|---|---|
| RQ1: native failure can accompany an admissible returned command | `outputs/task_contract_alignment/05_aggregate/point_main.json` | `task_point_main.csv`, `task_status_rows.tex`; Tables 2/5, Figures 1/2 | Observed prominently in strict DLS, not universally in TRAC-IK |
| Exact-input misses differ from solver failure without a witness | Fixed point identities and source NPZ; raw point returns; `verification.json` | Point UID indices, outcome matrix, Problem Formulation | A reference trajectory does not witness every method-specific input |
| Position-only / orientation-only effects | `point_paired.json`, `point_families.json` | `task_point_paired.csv`; RQ1/RQ2 | Strict nominal TRAC already succeeds throughout; no recovered strict misses to attribute |
| DLS continues after task admissibility | `02_point_study/*trace*` and `dls_excess_iterations.json` | `task_dls_excess_iterations.csv`; Figure 6, Appendix B | Direct DLS trace only; undefined excess stays undefined; no inferred TRAC iterations |
| RQ2: full TRAC alignment preserves witnessed-point success and modestly reduces mean time | `point_main.json`, query-paired intervals | `task_point_main.csv`, `task_point_paired.csv`; Table 2, Figure 2 | Not a point-level recovery result; quantiles need not all improve |
| DLS task stops lower point cost without changing the observed accepted UID set | Same point records and paired outcomes | Same CSVs, RQ2 | Native-success improvement must not be confused with recovered commands |
| Contract scale changes computational cost and accepted precision | `sensitivity_main.json`, `sensitivity_paired.json` | `task_sensitivity_*.csv`; Figure 3 | Same witnessed inputs at all scales; nominal stays the primary contract |
| Accepted error is larger but remains within each stated contract | Accepted-error distributions in main/family JSON | Main CSVs and Figure 3; RQ2/Discussion | Report position and orientation, not only acceptance |
| RQ3: aligned TRAC improves whole-trajectory completion on both robots and reduces cumulative cost | Authoritative old Panda and new UR5e raw records; `trajectory_main.json`, `trajectory_paired.json` | `task_trajectory_*.csv`; Table 3, Figure 4 | Three searches are within-trajectory repeats; trajectory remains the independent unit |
| Improvements depend on family | `trajectory_families.json`, family-paired intervals | `task_trajectory_families.csv`; Figure 5, Table 6 | Preserve every family, including flat or slower outcomes |
| Earlier DLS acceptance is not universally beneficial to tracking | DLS whole-trajectory records and completion UID differences | Table 3; `task_completion_uid_sets.csv`, `task_first_failures.csv` | UR5e loses strict-completed paths; cost savings do not erase this loss; cause is not identified |
| Deadline success differs from geometric completion | Per-frame outer latency and trajectory deadline counts | Table 3, trajectory CSVs | Nominal sampling interval is not a hard-real-time demonstration |
| No accepted return violates the implemented contract | `05_aggregate/verification.json`, saved returned configurations | Generated verification macro, witness indices | Kinematic software verification only |
| Learned routing does not generally accelerate witnessed successful solving | Frozen `revision_compute_allocation/reports/` | `task_allocation_*.csv`; Table 4, RQ3 | Historical case study, not a router newly attached to aligned TRAC/DLS |
| Aggregate routing savings can concentrate in common failures | Frozen decomposition and same-population records | `task_allocation_decomposition.csv` | Reject may avoid expensive unsuccessful work; no proof of mathematical infeasibility |

## Numerical authority and preservation

`paper/scripts/build_evidence.py` validates the stage-one delivery hashes before
writing `evidence_snapshot.json`, `paper_numbers.tex`, six table fragments and
`task_*.csv`. Empirical values in the text use generated macros. The figure
builder materializes exact frozen vector figures and records their hashes.
`paper/scripts/check_paper.py` checks sources, references, structure and build.

The current findings report is
`docs/TASK_CONTRACT_ALIGNMENT_FINDINGS.md`. Earlier findings, formal gates,
manifests, source identities, models and numeric solver/verifier settings remain
unchanged. Their original claims apply to their original protocols, not to an
expanded claim that learned routing is generally faster.

## Explicit non-claims

No new numerical IK solver, new Cartesian-bound capability, new task-tolerance
concept, learned verifier, universal alignment benefit, proven branch-drift cause,
collision or physical safety, hardware validation, or hard-real-time guarantee.
No failure without an exact-input witness is labeled mathematically infeasible.
No global pass/fail gate is imposed on this study.
