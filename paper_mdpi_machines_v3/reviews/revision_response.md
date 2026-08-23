# Post-review revision response

This response records changes made after the three isolated pre-submission
reviews. It is an internal audit artifact, not a journal rebuttal letter.

| Synthesized concern | Revision completed | Residual status |
|---|---|---|
| S-M1: optimized feasible latency lacks independent evidence | Title, abstract, evidence notice, provenance table, Results, Discussion, and Conclusion now identify the exact backend as seed-17 validation-only; “ready for release testing” was removed; the stopped UR5e/seed43 packaging gate is reported | **Open blocker.** Six unchanged artifacts and a fresh one-shot `test_v3` are still absent |
| S-M2: learned routing not distinguished from simple alternatives | Added frozen same-protocol results for previous-state DLS, previous-state TRF, learned-seed DLS, Cartesian-step guard, fixed cascade, and CG-HIK; added a functional positioning table for IK-Geo, viability QP, and generative IK | Internal comparator visibility resolved; matched external experiments remain future work |
| S-M3: component attribution exceeds ablations | Retained null ablations and explicitly declined individual necessity claims for history, ensemble, uncertainty, and calibration | Resolved for the bounded current claim |
| S-M4: rejectable queries may be easy | Added exact radial-unreachable and independently sampled large-step construction; removed “provably discontinuous”; stated that the strata contain generation cues and are neither natural prevalence nor OOD evidence | Resolved as a scope limitation; natural/OOD failures require new data |
| S-M5: online wording exceeds timing contract | Defined latency as empirical batch-one runtime, listed warmup/repeats/randomization/synchronization, separated the 20 ms kinematic interval from a hard deadline, and retained the 306.46 ms validation solver outlier | Hard real-time performance remains untested |
| S-M6: FEV and trajectories insufficiently defined | Added code-level DLS/TRF FEV accounting, cluster/bootstrap/sign-flip details, exact command-spike predicate, and the fact that zero accepted spikes are a verifier invariant; trajectory findings are now descriptive | Resolved for reproducibility; no path-level equivalence claim |
| Ambiguous use of “verified” | Changed title and central prose to “kinematically verified” and added an explicit acceptance-scope comparison | Resolved |
| Two-sided abstention/OOD proposal might leak into current evidence | Added a clearly prospective related-work/discussion paragraph and a separate executable v4 plan; no current Methods or Results claim OOD deferral | Resolved by evidence separation |

## Final review posture

The current manuscript is suitable for internal pre-submission circulation as a
bounded exact-kinematics study with an explicit negative eager-latency result. It
is not ready for journal upload until author metadata are supplied and the team
either completes the locked optimized release/test or deliberately submits the
paper as a negative runtime result without a positive optimized-latency claim.
