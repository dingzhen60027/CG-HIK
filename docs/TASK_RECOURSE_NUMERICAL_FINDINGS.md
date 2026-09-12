# TAR-IK: numerical completion of the unchanged joint objective

Baseline: `1a9db5071e6212bc25a9cdae9fe278328e47646a`. Development only.
No new algorithm name, objective, scenario, predictor, task contract or solver.
All new evidence is under
`outputs/task_recourse/numerical_completion_development/`; old outputs and paper
are not overwritten. The exact old kernel is loaded from the baseline Git blob
in memory for comparisons, rather than maintaining another versioned runtime.

## Protocol, before complete trajectory outcomes

1. Recheck all saved witness commands at each of the 80 mechanism states using
   the **actual internal backup of that state's old TAR call**. All 13 scenario
   targets must match exactly. A witness originally obtained from another
   previous state is only evidence after complete re-verification at this
   anchor; its source and previous-state equality are recorded. Also allow
   previously verified planned nodes at the same saved targets. No new witness
   searches are performed. Missing complete witnesses remain `unknown`.
2. A legal anchor plus legal commands for all nodes establishes J=0: q=anchor
   makes the command term zero and all future normalized pose excesses are zero.
   Nonnegativity of the unchanged objective supplies a global lower bound for
   this finite-node input only. No optimizer return code proves this property.
3. Improve initialization with the existing scaled least-squares nominal
   prediction step, then a common-Jacobian node correction. Clip guesses to
   the exact-anchor joint/rate interval. The map initializes z but never
   constrains the final free configurations. It is not thirteen IK solves.
4. Use the same sparse joint program with q's lower and upper bounds clamped to
   anchor; only after remaining real excess release q for joint updates.
   Recompute real FK, keep the lowest legal true objective, and return the exact
   unchanged anchor whenever its complete zero-cost plan is verified. A native
   current-pose interior margin must not exclude an already publicly legal
   fixed anchor; its bound uses the public unit radius in the fixed phase.
   This does not relax the public verifier or future joint/rate constraints.
5. Compare fixed-current update caps 2, 4, 6 on all 80 existing mechanism inputs,
   with two released updates and the existing 18 ms soft outer / 2 ms native
   limit. Three timing repeats per input. Select the budget with most known-zero
   states solved in all three repeats among those with reconstructed outer P95
   ≤20 ms, then the smallest cap; if none qualify select the lowest P95.
   This is a development numerical-budget choice, not a performance gate.
6. Same-input replay substitutes exactly the same saved backup in old and new
   kernels. Its numerical time is measured; adding the saved backup duration is
   labeled **reconstructed**, not a newly measured online TRAC call. Saved
   witnesses and actual next targets never enter online optimization.
7. Freeze the chosen budget before complete online runs. Compare old TAR,
   corrected TAR, task TRAC 5 ms and fixed Pink on all 40 existing trajectories
   per robot, three full repeats each, 150 frames. Interleave method jobs on the
   same resources. Accept before updating own state, hold after failure, advance
   every target and count every frame/time/timeout. No reference reset.

## Measurement and interpretation rules

The two success endpoints are TSR and DTSR20. Retain full outer latency,
per-trajectory cumulative time, accepted pose errors, commanded acceleration,
and all gained/lost UIDs. The additions concern only zero-cost completeness and
the initialization/fixed-current/joint-update/FK-check costs. Those timing
components are disjoint: fixed and joint categories contain linearization,
cone construction and solve; initial and nonlinear/final acceptance checks are
reported in FK checks. Backup, prediction and accounting remainder also remain
inside outer latency. Stage medians are not added to estimate a total median.

For complete trajectories the independent unit is UID (40/robot), with three
runs averaged inside UID before paired family-stratified bootstrap intervals.
Intervals are descriptive, unadjusted 95%, using the existing 4,000 resample
procedure. A zero-crossing interval does not establish equality. Same-input
comparisons use the same actual backup; across different closed-loop states,
aggregate percentages do not establish causal explanations.

Exact nonzero joint changes are distinguished from changes >1e−8 rad (the
existing reporting threshold); objective gaps are also shown against the old
1e−10 absolute comparison tolerance. Neither reporting threshold relaxes any
witness check. A tiny scalar objective does not by itself establish exact zero.

## Initial read-only finding

Saved commands establish zero-cost references for 39/40 Panda and 40/40 UR5e
inputs, with one Panda unknown. The old call changed all 79 anchors in exact
floating-point comparison, but no Panda change exceeds 1e−8 rad and none has an
objective gap above 1e−10. Only one UR5e input exceeds both reporting levels.
Thus most exact changes in these mechanism states are rounding-scale, not
evidence of a large unresolved optimization gap. Full raw configurations,
targets and re-verification results are saved; no unmatched source previous q
was silently reused.

The final conclusion will distinguish: (A) the old numerical implementation
did not solve the input's original objective well, (B) solving it better still
does not add task value, and (C) numerical completion yields actual task benefit.
None is assumed from the initial reference count.

## Numerical development log

The first same-input implementation used caps 2/4/6 and solved 78 of the 79
known-zero inputs exactly in all three repeats. Its provisional selection is
preserved as `selection.json`; no complete trajectory run used it. On UR5e
trajectory_22 the new initialized plan touched the sixth joint upper bound.
Although the cone solver returned `Solved`, reconstructed future commands
exceeded that bound by 1.2e−13 to 5.5e−13 rad. Every backtracked update was
therefore rejected, leaving true objective 8.10826 instead of the known zero.
This is an implementation-level numerical boundary issue, not evidence that
the recourse objective needs a different algorithm.

The existing q reconstruction rule (only clip violations ≤1e−8 rad, then check
the original nonlinear contract) is now applied to future z as well, using the
same representable interior bounds from the candidate current q. Larger
violations are not projected. This fixes arithmetic reconstruction, not the
verifier. The initial probe files remain in `same_input/`; the corrected probe
uses `same_input_roundoff_corrected/`, with a separate final selection file.
No objective/weight/trust-region/predictor change was made. A regression test
retains this exact state. Eleven tests pass after this correction.

Final same-input selection: **at most two fixed-current and two released joint
updates**, with the existing 18 ms soft limit and 2 ms native call limit.
All three candidate caps (2/4/6) solve all 79 known-zero inputs exactly in all
three repetitions after the roundoff correction; the smallest cap wins under
the recorded rule. Of the 79 states, 45 are zero at initialization, 33 after
one fixed-current update, and one after two. No released update is needed for
these known-zero inputs. One Panda state remains unknown/unavailable. These
results do not establish that the budget suffices on every online input.
`selection_roundoff_corrected.json` is final and precedes all complete runs.
