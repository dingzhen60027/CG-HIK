# Final paper claim map

Updated: 2026-09-04

## Central claim

CG-HIK uses action-complete development observations to choose a sufficient
tail-latency-aware entry into a shared numerical IK cascade. Learning allocates solver
effort, numerical geometry generates joint commands, and deterministic verification
governs acceptance.

## Claim-to-artifact mapping

| Manuscript claim | Evidence role | Authoritative frozen source | Generated paper artifact | Boundary |
|---|---|---|---|---|
| Entry cost is heterogeneous across queries and families | Development mechanism | outputs/counterfactual_v4_bulk plus release-v4 policy-selection reports | paper/source_data/development_oracle_distribution.csv; development_family_oracle.csv | Not confirmatory |
| Router has finite empirical-oracle regret and pinball-trained latency heads with coverage diagnostics | Development policy selection | outputs/release_v4_candidate and outputs/release_v4_locked | development_routing_metrics.csv | Policy-selection split participated in freezing |
| Feasible verified success is preserved on fresh points | Fresh point test | outputs/test_v4_aggregate_repair_v1/aggregate_summary_v4.json | point_formal_results.csv; Table 2; Figure 3 | Primary seed 17 |
| Feasible mean FEV and P95 decrease | Fresh point test | same aggregate summary | point_formal_results.csv; Table 2; Figure 3 | P50 rises; Panda P99 nearly tied |
| Known-infeasible reject avoids computation | Fresh point test | aggregate_summary_v4.json, ood_and_abstention block | point_rejectable_results.csv; Figure 3 | Constructed known-infeasible population only |
| Point OOD is weak and defer has no recovery | Fresh point test | aggregate_summary_v4.json | paper_numbers.tex and evidence_snapshot.json | OOD is not a contribution |
| The legacy formal suite's six point and two trajectory-completion margins pass Holm, but its overall paper gate is false | Frozen formal suite | joint_holm_v4.json and paper_gate_v4.json | evidence_snapshot.json; Results/Supplement | Panda strict operational condition failed; the legacy trajectory arm is distinct from RQ3 |
| CG-HIK completion is not below hard on final fresh trajectories | Final one-shot test | outputs/fresh_transition_v4_test/main_table.json and completion_uids.json | trajectory_main_results.csv; Table 3; Figure 4 | Whole trajectory is completion unit |
| Aggregate latency and mean FEV decrease on both robots | Final one-shot test | main_table.json and final_gate.json | trajectory_main_results.csv; Table 3; Figure 4 | P50 increases |
| Among families completed by at least one trajectory on both robots, benefit is strongest near singularity and is not universal | Final one-shot test | family_table.json | trajectory_family_results.csv; Figure 5 | Panda joint-limit-skim regresses slightly; UR5e joint-limit has 0/20 completion under both methods |
| Accepted contract violation is zero | Final one-shot test | main_table.json, final_gate.json, raw-record audit | Table 3; evidence_snapshot.json | Contract is kinematic, not universal safety |
| Final trajectory gate passes | Final one-shot test | final_gate.json | evidence_snapshot.json | Frozen thresholds; results not used for tuning |
| Temporal shortcuts can alter closed-loop recoverability | Development-only mechanism | V5–V7 output manifests and summaries | Discussion only | No versioned method claim or formal comparison |

## Exact headline numbers and sources

### Fresh point-query primary system

Source: outputs/test_v4_aggregate_repair_v1/aggregate_summary_v4.json

| Number | Panda | UR5e |
|---|---:|---:|
| Feasible P95 ratio, CG-HIK/fixed | 0.753809 | 0.7427 |
| Feasible P99 ratio | 0.993101 | 0.8808 |
| Mean FEV reduction | 16.14% | 36.27% |
| P50 change | +16.47% | +16.45% |
| Known-infeasible reject recall | 95.65% | 93.95% |
| Known-infeasible FEV avoided | 95.686% | 93.916% |
| OOD AUROC / AUPRC | 0.430386 / 0.209706 | 0.503905 / 0.229905 |
| Defer recovery success | 0% | 0% |

Multiplicity and gate sources:

- outputs/test_v4_aggregate_repair_v1/joint_holm_v4.json
- outputs/test_v4_aggregate_repair_v1/paper_gate_v4.json

### Final fresh transition-rich trajectories

Source: outputs/fresh_transition_v4_test/main_table.json and final_gate.json

| Number | Panda | UR5e |
|---|---:|---:|
| Completion, CG-HIK vs hard | 41/80 vs 40/80 | 34/80 vs 34/80 |
| Aggregate cumulative latency reduction | 43.371% | 54.954% |
| Mean FEV reduction | 45.164% | 65.122% |
| P50, hard → CG-HIK (ms) | 1.651 → 1.844 | 1.558 → 1.731 |
| P95, hard → CG-HIK (ms) | 229.142 → 201.564 | 38.104 → 29.745 |
| P99, hard → CG-HIK (ms) | 353.570 → 353.672 | 50.821 → 39.085 |
| Accepted contract violations | 0 | 0 |

Family source: outputs/fresh_transition_v4_test/family_table.json

| Family | Panda cumulative reduction | UR5e cumulative reduction |
|---|---:|---:|
| smooth_fast_orientation_smooth | 29.62% | 9.44% |
| regular_near_singular_regular | 59.26% | 59.60% |
| central_joint_limit_skim_return | −0.68% | 79.26% |
| slow_high_curvature_high_speed_slow | 44.09% | 20.47% |

The negative Panda joint-limit value means CG-HIK is 0.68% slower, not a saving.
The nominal 79.26% UR5e joint-limit reduction accompanies 0/20 completed trajectories
under both methods and may reflect earlier failure; it is not evidence of cheaper
successful paths.

## Required wording boundaries

- Write “kinematically verified” or “verified against the implemented command
  contract,” never safety-certified or collision-safe.
- Write “action-complete pathway supervision,” not unobserved causal counterfactual.
- Write “P50 increased while cumulative latency and P95 decreased,” never all latency
  decreased.
- Write “conservative OOD defer with weak point discrimination,” never robust OOD
  detector.
- Write “same exogenous trajectory frame with method-specific closed-loop state,” not
  identical full IK query for every paired frame.
- Write “two sensitivity seeds on the same point test,” not three independent test
  replications.

## Reproducibility chain

paper/scripts/build_evidence.py validates frozen inputs and writes:

- paper/generated/evidence_snapshot.json
- paper/generated/paper_numbers.tex
- paper/generated/table_solver_rows.tex
- paper/generated/table_point_rows.tex
- paper/generated/table_trajectory_rows.tex
- paper/source_data CSV files

paper/scripts/make_figures.py reads those CSV files and produces the five main figures.
The manuscript references generated TeX macros and table-row fragments so numeric
claims are not independently hand-maintained.
