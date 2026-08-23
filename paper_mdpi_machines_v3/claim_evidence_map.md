# Claim-to-evidence map

| Manuscript claim | Evidence source | Allowed wording | Boundary |
|---|---|---|---|
| The calibrated gate preserves feasible-query success while reducing mean function evaluations | Six frozen `paper_v2` runs, two robots x three training seeds | “preserved within the prespecified 1 percentage-point margin” and exact per-run rates | One locked test set per robot is reused across seeds; seeds are sensitivity replicates, not independent datasets |
| Explicit rejection avoids futile work | Six frozen `paper_v2` runs | Report rejectable-query rejection, mean FEV, and P95 latency | Query mixture is stratified and does not estimate deployment prevalence |
| The original eager learned gate incurred a fixed feasible-query overhead | Frozen `paper_v2` formal test | Report the failed feasible-latency claim gate | Do not hide or retune after test |
| Exact implementation optimization removes most fixed risk-inference overhead | `latency_pilot_v3`, validation only, seed 17 | “validation-only pilot” and exact P95 ratios | Not an independent test claim; `test_v3` pending |
| A verifier is operationally necessary | Frozen counterfactual interception counts | “intercepted converged but inadmissible candidates” | Kinematic verification only; no collision/torque/contact guarantee |
| Trajectory completion under the tested kinematic contract | Frozen 40-path tests per run | descriptive completion rates | No path-level equivalence claim; no physics, controller tracking, acceleration/jerk, or hardware inference |
| Accepted trajectory commands have zero recorded spike rate | Frozen query records and shared verifier | State that the spike predicate is the verifier's one-frame velocity predicate | This is an acceptance invariant, not independent smoothness evidence |
| Learned routing differs from a Cartesian-step guard and conventional entry points | Frozen query-level `paper_v2` records; `formal_comparator_runs.csv` and `formal_comparator_summary.csv` | Report feasible success, hard-valid success, FEV, eager P95, and descriptive trajectory completion | Do not claim fastest overall; previous-state DLS has lower eager P95 and the methods solve different contracts |
| Ensemble/calibration/uncertainty components are individually necessary | Diagnostic ablations | Report as mostly null operational ablations | Do not claim causal necessity where results are null |
| Bounded fallback contributes to reliability | No-fallback ablation | Report feasible-success/trajectory changes | Avoid universal robustness claims |

## One-sentence argument

In online manipulator inverse kinematics, a calibrated history-conditioned gate
can allocate a shared kinematically verified solver cascade's computation and
reject constructed futile queries while preserving feasible-query success; six
locked runs support this portfolio-specific allocation effect, whereas
validation-only runtime optimization does not yet independently establish the
optimized latency claim.
