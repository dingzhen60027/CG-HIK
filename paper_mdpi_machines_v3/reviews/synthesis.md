# Cross-review synthesis

Post-review document. This synthesis was prepared only after all three reports
were finalized. It was not shown to the reviewers.

## Review setup

- **Input scope**  Current `main.tex`, the evidence snapshot, claim-evidence
  map, and release packaging diagnostic.
- **Assessment boundary**  Six frozen `paper_v2` formal runs and one
  validation-only seed-17 latency pilot. The six-artifact release is incomplete
  and `test_v3` is absent.
- **Isolation**  Three separate reviewer contexts received the same immutable
  packet. Their preassigned emphases were technical soundness, originality and
  significance, and interdisciplinary readability. No reviewer received
  another report or a concern list.

## Consensus strengths

All three reports independently identified the matched shared-cascade design as
the manuscript's strongest feature. The proposed and fixed methods share the
proposal ensemble, numerical stages, fallback, and kinematic verifier, which
supports attribution of within-portfolio FEV differences to the entry policy.
All three also valued the explicit negative formal latency result, the separation
of learned allocation from command acceptance, the distinction between training
seed sensitivity and independent testing, and the retention of null ablations
and scope limits.

## Consensus blocking concerns

### S-M1  Positive optimized feasible-latency evidence is absent

Mapped concerns are R1-M1, R2-M1, and R3-M1. Reviewers 1 and 2 marked this
blocking. Reviewer 3 treated it as nonblocking only for the narrower formal FEV
claim.

The formal eager implementation failed every feasible P95 gate. The favorable
exact-backend result was selected and measured on validation at seed 17. The
subsequent six-artifact release stopped at `ur5e/seed43`, and `test_v3` was not
started. The current paper therefore cannot establish positive optimized
feasible latency. Resolution requires either the unchanged locked release and a
fresh one-shot test, or a submission explicitly centered on the formal negative
runtime result with all readiness language removed.

### S-M2  Learned routing is not fully distinguished from simpler alternatives

Mapped concerns are R1-M2, R2-M2, and R3-M5. Reviewer 2 marked this blocking for
the originality case.

The matched fixed cascade establishes an internal routing effect, but results
for the declared threshold guard and conventional entry baselines are not
reported with the same visibility. The paper also lacks a compact comparison of
what closest warm-start, solver-portfolio, analytic, and path-level methods do.
Current prose can repair positioning, but empirical superiority to simple or
external policies requires new matched experiments. If those are unavailable,
the contribution must remain a controlled demonstration of portfolio-specific
learned routing and verified refusal.

## Other consensus major concerns

### S-M3  Mechanism attribution is broader than the ablations support

Mapped concerns are R1-M3 and R2-M3. Non-reject macro-F1 is modest, and history,
single-member, disagreement, and calibration ablations are operationally close
to the proposed system. The evidence supports an integrated frozen policy,
especially explicit rejection. It does not establish the individual necessity
of the ensemble, history, uncertainty, or calibration. The current paper should
narrow the mechanism claim and report action-resolved evidence if already
available.

### S-M4  Constructed rejectable queries may be easier than deployment failures

Mapped concerns are R1-M4, R2-M4, and R3-M3. The strongest saving occurs on
generated unreachable and discontinuous queries, yet their proof and generation
rules are not operationally detailed in the manuscript. Current claims must be
limited to the pooled constructed strata unless subtype generation and results
can be recovered from frozen artifacts. OOD or natural-workload robustness needs
new data.

### S-M5  Online language exceeds the reported timing contract

Mapped concerns are R2-M4 and R3-M2, with support from R1-m4. A 20 ms frame
interval and bounded solver evaluations do not constitute a wall-clock bound.
The 306.46 ms Panda validation outlier makes this distinction important. The
manuscript should call the reported latency empirical batch-one runtime and
state that deadline-miss frequency and recovery behavior are not established
unless they can be computed from frozen records without changing any gate.

### S-M6  Reproducibility details for FEV and trajectories are incomplete

Mapped concerns are R1-M5 and R1-M6. The central FEV counter needs an operational
definition across DLS, backtracking, TRF, caching, and verification. Trajectory
results use the correct whole-path unit but remain descriptive for forty locked
paths and should not be called equivalence. The paper should define the counter
and narrow trajectory wording.

## Where emphasis differed

Reviewer 1 placed greatest weight on FEV accounting, action-level calibration,
and trajectory inference. Reviewer 2 placed greatest weight on whether a learned
gate is needed relative to a threshold guard or adaptive portfolio and on the
limited deployment significance of exact kinematics. Reviewer 3 placed greatest
weight on a reader's ability to reconstruct evidence provenance, the meaning of
`verified`, the dependence across training seeds, and the lack of a wall-clock
deadline contract.

## Minor revision checklist

- Define command spike with formula, units, threshold, denominator, and its
  relation to the verifier. R1-m1 and R3-m2.
- Define `confidence` as in-distribution calibrated action probability and make
  no OOD confidence claim. R2-m3.
- Use `kinematically verified` where unqualified `verified` could imply broader
  safety or formal verification. R3-m1.
- State in the title or abstract that two robot-specific test sets are reused
  across three training-seed sensitivity runs. R2-m1 and R3-M4.
- Put absolute formal feasible P95 values or ratios in the main results table,
  rather than showing only favorable rejectable latency. R3-m4.
- Define cluster membership, sign-flip implementation, randomization, finite
  sampling correction, and Holm order. R1-m2.
- Report or explicitly mark missing hard-valid and declared comparator outcomes.
  R1-m3 and R1-M2.
- Add CPU, thread, affinity, and timing-repeatability details where frozen
  manifests support them. R1-m4.
- Resolve author, funding, repository, and data-release placeholders before
  submission. R2-m4.

## Broad-interest and significance readout

The strongest transferable finding is that numerical-work reduction and
end-to-end speed are distinct, and that a learned resource-allocation layer can
remain subordinate to an explicit kinematic acceptance contract. That principle
may interest adaptive-computation and runtime-assurance readers. The current
evidence remains a robot-specific exact-kinematics benchmark. It does not yet
establish OOD behavior, physical safety, hard real-time performance, hardware
reliability, or cross-robot model generalization.

## Most important issues before a strong engineering-journal case

1. Decide the evidential route. Complete the locked release and independent
   optimized test, or submit a deliberately negative-runtime paper.
2. Make the closest-alternative comparison visible, especially the simple
   threshold guard and declared conventional entry baselines.
3. Add one provenance table that separates formal test, validation pilot,
   failed release packaging, and absent `test_v3`.
4. Define FEV, rejectable subtypes, timing semantics, command spike, and
   trajectory inference precisely.
5. Narrow component, deployment, OOD, safety, and replication language to the
   evidence actually shown.

## Risk and unsupported claims

- Positive optimized feasible latency is unsupported by independent evidence.
- Hard real-time or 50 Hz guarantees are unsupported.
- The six seed-conditioned runs are not six independent runtime datasets.
- The present data do not establish natural workload prevalence, OOD refusal,
  or uncertainty-aware deferral.
- Physical collision, dynamic, controller, sensing, torque, contact, and
  hardware conclusions are outside scope.
- Individual necessity of history, ensemble disagreement, uncertainty, and
  calibration is unsupported by the current operational ablations.
