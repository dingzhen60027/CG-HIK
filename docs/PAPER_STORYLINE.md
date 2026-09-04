# CG-HIK final paper storyline

Updated: 2026-09-04

## One-sentence thesis

> **Learning allocates solver effort per query; numerical geometry generates joint commands; deterministic verification governs acceptance.**

CG-HIK treats online inverse kinematics as query-adaptive allocation over a shared
numerical solver portfolio. It changes where computation begins, not what constitutes
an admissible command.

## The closed argument

| Element | Final paper position |
|---|---|
| Problem | Sequential IK requests differ in initialization, conditioning, joint-limit proximity, branch history, and motion scale; they therefore require different numerical effort. |
| Established strengths | Analytical, Jacobian/DLS, trust-region, QP, and learned/generative IK each provide valuable geometry, constraints, or proposal coverage. |
| Gap | Hybrid systems often retain a fixed proposal count, budget, or execution order. A robust cascade may short-circuit, yet a fixed entry can still pay a redundant prefix. |
| Hypothesis | Robust coverage does not require every query to pay the worst-case computation cost. |
| Method | Execute every cascade entry for each development query; learn shared terminal success and entry-specific P50/P95; choose the sufficient entry with minimum predicted P95. |
| Governance | Numerical solvers generate commands. The deterministic verifier alone accepts them. Reject emits no command; uncertainty/OOD defers to the full fixed cascade. |
| Point evidence | Frozen fresh points preserve feasible success, reduce FEV and P95, and avoid most known-infeasible work; P50 rises, Panda P99 is nearly unchanged, point OOD is weak, and defer has no observed recovery. |
| Trajectory evidence | A final one-shot transition-rich evaluation matches or improves completion and reduces cumulative latency, FEV, and P95 on both robots; P50 rises and Panda P99 is effectively tied. |
| Conclusion | CG-HIK reallocates computation toward expensive requests. It is not uniformly faster and it is not a learned command generator. |

## Canonical names

| Artifact name | Paper name |
|---|---|
| fixed_robust_cascade | Fixed robust cascade |
| always_hard | Fixed hard-entry cascade |
| counterfactual_cghik_v4 | CG-HIK |
| categorical router | Categorical routing baseline |
| threshold guard | Cartesian-step threshold baseline |

Easy, medium, and hard are cascade entries, not true query-difficulty classes.
“Counterfactual” means that all alternative entries were actually observed during
development; the preferred wording is **action-complete pathway supervision**.

## Three research questions

1. **Computation heterogeneity and predictability.** How do observed entry costs vary
   across queries and families, and how closely does the router approach the empirical
   oracle?
2. **Fresh point-query system performance.** Does frozen CG-HIK preserve verified
   success while reducing FEV, P95, fallback work, and rejectable-query computation?
3. **Fresh transition-rich trajectory performance.** Relative to the Fixed hard-entry
   cascade, does CG-HIK preserve completion while reducing cumulative latency, FEV,
   and upper-tail frame latency?

## Evidence hierarchy

1. outputs/counterfactual_v4_bulk/: development-only mechanism and oracle analysis.
2. outputs/test_v4_aggregate_repair_v1/: fresh point-query evidence.
3. outputs/fresh_transition_v4_test/: final one-shot transition-rich trajectory
   evidence and final system gate.
4. V5–V7 output trees: historical, development-only temporal-shortcut analyses.

The point-study overall gate was false because a separate Panda operational condition
required strictly positive OOD-feasible false-reject improvement while both compared
methods had zero false rejects. The legacy suite's six point margins and two
trajectory-completion margins passed joint Holm correction; that trajectory arm is
not the final transition-rich evaluation. The final transition-rich gate passed for
both robots. Both facts are reported; neither replaces the other.

## Final result capsule

| Robot | Completion, CG-HIK vs hard | Cumulative latency change | Mean FEV change | P50 change | P95 change | P99 change |
|---|---:|---:|---:|---:|---:|---:|
| Panda | 41/80 vs 40/80 | −43.37% | −45.16% | +11.66% | −12.04% | +0.03% |
| UR5e | 34/80 vs 34/80 | −54.95% | −65.12% | +11.09% | −21.94% | −23.09% |

Family interpretation:

- near-singular trajectories show the largest cumulative savings;
- high-curvature/high-speed trajectories also benefit substantially;
- smooth/orientation transitions show moderate savings;
- Panda joint-limit-skimming is a disclosed counterexample: CG-HIK latency and FEV
  are slightly worse than hard;
- Panda high-speed completion remains low for every method.

## Claim boundary

The paper supports query-adaptive resource allocation under exact URDF/Pinocchio
kinematics and an explicit command contract. It does not support collision safety,
dynamics, torque/contact behavior, low-level tracking, real-robot reliability, a hard
real-time bound, strong point-OOD detection, or uniform latency improvement.
