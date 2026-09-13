# Frozen task-set boundary supplementary validation

Baseline `8e4aa63975d465f3de8a1eef2f7773889252c91d`, same branch. No main solver, verifier, original outputs or paper edits. The supplied README, PROVENANCE and task instructions were read completely; both sequential local probes are archived unchanged and remain observed development evidence.

## Question and geometry

Can the fixed task minimax solver return more admissible current commands when a task set is witness-feasible but its exact center is not guaranteed reachable? This does not test whether the center is mathematically unreachable. An offset query's exact-center status is always `unknown`.

Independent ordinary / geometry-selected near-singular / near-upper-physical-limit anchors follow the original revision point sampler (64 geometric candidates for singularity selection). Per anchor, sample three witness displacements: original local scale .1–.6 with uniform components; original hard_feasible scale .75–.98 with signed .75–1 components; all-joint signed .999 velocity×dt, physically clipped. The latter is targeted boundary pressure, not natural prevalence. Record actual public velocity utilization and physical clipping separately.

Sample one independent isotropic 3D translation direction and one rotation direction per anchor, shared across all nine derived queries. Targets are p_w+alpha*epsilon_p*u_p and Exp(alpha*epsilon_R*u_R) R_w, alpha=0/.5/.95. Original 1 mm/.5 degree, .02 s, joint limits, velocities and velocity tolerance remain unchanged. Verify every witness using each exact input, with no solver screening. Alpha zero is an exact-FK control; retain every cell.

Generate and seal **both** development (10 anchors per geometric class/robot; 270 queries) and validation (80/class/robot; 2160 queries) before comparison. Seeds and all UID/query hashes are fixed in the config. Check against previous identity files, the other new split and supplied probe hashes. No replacement after results, no new trajectories.

## Comparators, controls and selection

Original GN κ1; frozen CompletedTaskBalanceGN κ1 forcing .25; its same-code tight mode; fixed-weight GN; same completion outer function with cached Clarabel epigraph direction; task-aligned TRAC 5/20 ms. No predictive or fallback chain runs.

The fixed-weight adapter copies the original GN solve function into an instance-local namespace, changes only its three task-cost evaluations and task H/g terms, and asserts exact source substitutions. Cost theta||ep||²+(1-theta)||eR||² gives factors 2theta and 2(1-theta) in H/g. At .5, operations equal original GN. Task norm checks, posture, damping, line search, caps and verifier stay original. Test exact decisions at .5; reject any upstream source mismatch.

Select **one** theta from {.2,.35,.5,.65,.8}, equal-robot mean verified success on the new development anchors, then equal-robot mean outer time, then distance to .5, then smaller theta only for a remaining exact tie. Three repeats nested per query. No main-method tuning. Save complete candidate scores and selection before validation calls. All main comparators also run on development for interface integration; results cannot alter the fixed design.

Clarabel uses the byte-identical completion outer function via an instance-local function namespace, substituting only the direction provider. First theta=.5 QP and native task eligibility, cached FK, final verifier, minimax backtracking and finite budgets remain identical. If more work is needed, one cached existing EpigraphReference solves the same regularized minimax box problem. It receives the same early legal first-step opportunity, not an additional IK fallback. Retain the best available feasible local direction; all returned current commands still require the original verifier. Existing Clarabel tolerances and max_iter=100 unchanged. Full-call clock includes conic conversion/update/quality. No global function mutation.

All methods receive only current pose, previous_q and dt. Store witness outside the online API. Each query resets to its own previous state; never use returned q as another query's initial state. CPU4, single-thread libraries and original NumPy path. Synthetic midpoint warmups, plus one synthetic conic structure warmup, occur before measured calls; initialize/warmup times separately stated. Three calls/method/query interleaved; all failures and >20ms retained. Primary timing is caller perf_counter_ns around the entire solve invocation, including final verification/result conversion; serialization and read-only replay are excluded for all methods equally.

## Statistical and causal boundaries

Anchor is the independent unit (240/robot), nine related queries per anchor, three nested repeat calls per query. Average repeats within query; paired, geometric-family-stratified cluster bootstrap resamples anchors with all nine queries together (4000 fixed-seed resamples). Report all 9 displacement×alpha cells and all 27 geometry×displacement×alpha cells, verified/within20ms success, gains/losses, paired differences and time ratios with descriptive unadjusted 95% intervals. No outcome-based gate, equivalence claim or selective significant-cell report. Report actual errors, >90% boundary use, iteration counts, .5-first acceptance, dual continuation and relative-stop fractions with explicit denominators.

For GN misses and relative successes, preserve exact shared inputs, all method commands/repeats and legal witnesses. Recompute final-point normalized gradient/projected residual diagnostics from saved q (no additional IK); local stationarity is not global infeasibility. If needed, diagnostic callbacks are separate from measured calls.

Old independent 2000/2000 point outcomes and unchanged GN-versus-balance trajectory completion sets remain authoritative. This finite supplemental query-domain validation cannot establish new closed-loop tracking capability or deployment prevalence. Do not add algorithms, weights, data, trajectories or paper writing after these results.

## Entry and evidence

`scripts/run_task_set_boundary.py`: verify → resolve → inputs → development (both robots) → select → validation (both robots). Tests run before new inputs/comparison. Outputs exclusively under `outputs/single_solver_evidence/task_set_boundary_evaluation/`. Commit fixed code and all input identities before validation; final report and raw records then commit/push to the same branch.

Mathematical discussion only reuses existing foundations: M=max(||ep||²,||eR||²)≤1 iff both pose blocks pass, conditional on finite joint/rate legality; convex epigraph two-block scalar dual; valid L/U and gap≤.25(P0−U), P0>U yield at least80% regularized local model decrease. Not an IK success/convergence/trajectory guarantee.
