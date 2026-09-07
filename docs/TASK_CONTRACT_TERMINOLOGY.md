# Task-contract manuscript terminology

Stage-one evidence commit: `089a3fe7ec0456002aa0f3b9661119b99072fad7`, pushed before manuscript reconstruction.

| Canonical expression | Meaning | Avoid |
|---|---|---|
| solver-internal success | Original native convergence flag or return code | Unqualified success |
| task-command admissibility; task admissibility after definition | Unchanged final pose, finite-value, joint and rate verifier accepts | Safety certification |
| within-deadline admissible success | Task acceptance and measured outer latency at most the deadline | Hard real-time guarantee |
| task-contract-aware solver interface | Conservative native mapping, dynamic bounds, solve, independent verification | New TRAC-IK or DLS algorithm |
| witness-confirmed missed-admissible query | Exact input has independently saved verified command; tested call returns none | Solver failure proves infeasibility |
| reference-feasible trajectory | One continuous verified reference path exists | Every method-specific state remains feasible |
| excess iterations after task admissibility | Directly observed DLS main iterates after first admissibility | Unobserved TRAC over-solving |
| fully aligned TRAC-IK | Conservative component box in the task norm balls | Exactly equivalent tolerance sets |
| task-aligned DLS | Original numerical solver with public pose norm stops | Guaranteed better closed-loop tracking |
| CG-HIK | Frozen secondary adaptive-computation case study | Main universal acceleration method |

Argument: numerical convergence and executable command acceptance have different
semantics; mapping the fixed task contract into existing solvers can remove wasted
work and some trajectory failures, but the closed-loop benefit depends on the solver,
robot and workload. The experiment establishes this boundary before asking whether
allocation overhead has a useful role.

Use no development-version chronology in the body. Report larger legal residuals
alongside computation. Keep both positive TRAC results and the UR5e DLS completion
loss. A finite-horizon witness establishes existence only for its identical input.
No method, threshold, solver, trajectory or model changes follow the evidence.
