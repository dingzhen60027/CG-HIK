# Methods and Results writing scaffold

## Method summary

We formulate online inverse kinematics as a verified, variable-computation decision process. Given desired pose (x_t^d) and previous joint state (q_{t-1}), a bootstrap ensemble predicts bounded joint increments. Ensemble dispersion and seed-level geometric diagnostics are passed to a calibrated four-class difficulty model. Its probabilities select one of three numerical budgets. Every candidate is refined by the same adaptive damped-least-squares solver and checked by a common geometry and continuity verifier. Low-confidence failures invoke bounded optimization from retrieved workspace-neighbour seeds; otherwise the command is rejected.

## Claims permitted before full experiments

- The implementation jointly models seed proposal, query difficulty, budget allocation, verification, and fallback.
- The verifier prevents invalid candidates from being counted as successful under the declared tolerances.
- Risk labels correspond to observed numerical refinement outcomes, rather than a hand-written geometric difficulty score.

## Claims forbidden before full experiments

- First uncertainty-aware IK method.
- Formal safety, global convergence, or general unreachable-target proof.
- Cross-robot neural generalization.
- Real-time performance or superiority based on the toy smoke experiment.
- Collision avoidance, dynamic feasibility, or hardware reliability.

## Results placeholders

1. Risk discrimination and calibration on locked test data.
2. Verified success–function-evaluation Pareto curves.
3. Mean, P95, and P99 end-to-end latency.
4. Stress-category breakdown on both robots.
5. Cartesian trajectory continuity and spike rate.
6. Six ablations and fallback case analysis.
7. Three-seed paired confidence intervals and failure cases.

