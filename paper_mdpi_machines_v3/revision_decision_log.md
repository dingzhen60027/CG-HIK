# Revision decision log for the dual-abstention proposal

This document records how the supplied 23 August 2026 critique is handled. It
prevents a future-method proposal from leaking into the evidence claims of the
current `paper_v2` manuscript.

| Supplied recommendation | Decision for current manuscript | Evidence required before promotion to a result claim |
|---|---|---|
| Reframe IK as solver-resource allocation under a command contract | **Adopt now.** This is already supported by the shared-cascade comparison and verifier design | None beyond the frozen evidence already cited |
| Preserve and foreground the failed eager feasible-latency gate | **Adopt now.** Keep it in Abstract, Results, Discussion, and evidence notice | None |
| Distinguish FEV reduction from end-to-end latency reduction | **Adopt now.** Make this a central systems finding rather than a footnote | Frozen `paper_v2` and validation-only stage timing |
| Add modern analytic, constrained, and failure-detection literature | **Adopt selectively.** Add only independently verified primary sources and describe scope fairly | Verified bibliographic metadata and source reading |
| Present the exact TorchScript pilot as deployed test evidence | **Reject for now.** It remains validation-only | Six passing locked artifacts plus fresh one-shot `test_v3` |
| Claim that history, ensemble, uncertainty, or calibration causes the operational gain | **Reject for now.** Existing ablations are largely null | Targeted ambiguity/OOD experiments with a frozen causal comparison |
| Introduce command rejection versus model deferral | **Future v4 only.** Useful conceptual extension, absent from the current implementation | Implemented OOD detector, frozen deferral rule, ID/OOD independent test |
| Introduce counterfactual action-outcome labels | **Future v4 only** | All entry actions executed offline on training-only queries; labels and generation hashes frozen |
| Predict action success and P50/P95 latency | **Future v4 only** | Multi-head model, calibration, held-out quantile coverage and batch-one timing |
| Add route hysteresis | **Future v4 only** | Predefined transition rule, trajectory-level ablation, route-switch and completion results |
| Expand the verifier to acceleration and jerk | **Future v4 only** | Robot-specific limits and a frozen discrete-time definition; regenerated labels and fair baselines |
| Add collision verification | **Optional future subset.** Do not broaden the present claim | Collision model, environment protocol, continuous/discrete collision semantics, collision-aware baselines |
| Claim OOD robustness | **Reject for current paper** | Prespecified OOD families with non-overlapping parameter ranges and independent test |
| Add TRAC-IK, IKSel, IK-Geo, or QP baselines | **Needs new experiment.** Related-work acknowledgement can be added now | Reproducible implementation, same command contract, matched hardware and timing protocol |
| Add Panda and UR5e hardware trajectories | **Needs new experiment** | Hardware availability, controller configuration, trajectory-level replication, safety approval and logs |
| Treat encoder FK as ground-truth Cartesian tracking | **Reject.** It may be reported only as model-based tracking | Independent metrology is required for physical Cartesian ground truth |
| Tighten the future feasible P95 gate to ratio below 1.0 | **Do not retrofit to v3.** The locked v3 paper gate remains 1.25; any new v4 gate must be preregistered before v4 data | New protocol and untouched v4 test |

## Current-paper claim sentence

The current evidence supports a bounded conclusion. A calibrated action gate
reduced numerical work and rejectable-query cost under an otherwise identical
kinematic solver portfolio, but the eager batch-one implementation failed the
formal feasible-latency gate and the exact-export improvement remains
validation-only.

## Future-paper hypothesis sentence

The v4 study will test, rather than assume, whether counterfactual success and
tail-latency prediction combined with two-sided abstention can reduce verified
deadline misses while preventing uncertain OOD requests from being incorrectly
rejected.
