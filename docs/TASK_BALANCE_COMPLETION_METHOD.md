# Task-balance numerical completion: fixed method and validation protocol

Baseline: `23f8555bd214eb4ac45b164a536ead55e142e44c`, same `codex/hierarchical-v5` branch. This is the finite completion package, not a new objective or runtime hierarchy. Supplied package instructions, provenance, local raw results and commands are archived in `outputs/single_solver_evidence/task_balance_completion/task_package/task_balance_completion/`.

## Local problem and limited progress property

For the current target and actual previously accepted command, retain normalized residual blocks e_p,e_R, residual Jacobian G=J_e S, and the fixed current-frame step box B. S uses the original per-joint speed × dt plus velocity tolerance. The objective is

`P(d)=max(||e_p+G_p d||²,||e_R+G_R d||²)+R(d)`;
`R(d)=λ||d||²/2+κ||c+Wd||²/2`, with d∈B.

Let `hθ=θ f_p+(1−θ)f_R+R`, θ∈[0,1]. The epigraph multipliers sum to one. At θ=.5, the weighted box QP has exactly the same H/g assembly as the frozen GN at the same κ: `H=GᵀG+λI+κW²`, `g=Gᵀe+κWc`. The first direction is not a second solver call. Joint/rate bounds and the public 1 mm / 0.5° verifier are unchanged.

For a feasible local step v, `U=P(v)` is an upper bound on P*. Strong convexity λ>0 gives

`Lθ=hθ(v)+min_{z∈B}{∇hθ(v)ᵀ(z−v)+λ||z−v||²/2} ≤ P*`.

Compute this minimum with `δ=clip(−∇hθ(v)/λ,lo−v,hi−v)`. It does not infer exact normal-cone membership from an approximately active coordinate. Retain the smallest feasible U and largest L across evaluated θ. In exact arithmetic, with `D=P(0)−U>0`, `E=U−L≥0`, and `E≤.25D`,

`D*=P(0)−P* ≤ P(0)−L = D+E ≤ 1.25D`, hence **D≥.8D***.

This is a fraction of the best regularized **local model** decrease. It is not an IK success rate, a nonlinear convergence theorem, or a trajectory guarantee. Floating evaluations are diagnostics, not interval-arithmetic certificates. Nonfinite bounds or `L>U` beyond `64 eps max(1,|P0|,|U|,|L|)` disable the condition. Numerically consistent tiny negative gaps are clamped only within that documented roundoff range.

## One implementation, distinct termination semantics

The historical `TaskBalanceGN` class stays unchanged. The appended generic completion core and `CompletedTaskBalanceGN` adapter use existing server kinematics, analytic SO(3) residual derivatives, original NumPy box-QP and original verifier. Cached tight and relative modes share the same code and differ **only** in `forcing=None` versus `.25`. Cached trial residuals/Jacobians and delayed block products remove repeated work; these are engineering changes, not mathematical novelty.

Native task eligibility may terminate internal work, but the final independent verifier alone accepts the command. Failed external validation remains failed, without an alternative IK solver. Otherwise the same nonlinear minimax Armijo test is required before adopting a non-admissible intermediate update. The internal statuses distinguish task eligibility, relative model progress, tight gap, cap, stationarity, numerical failure and deadline. All diagnostics retain actual bounds; an early task-eligible step has no invented optimality gap.

Unchanged budgets: κ0/κ1 development controls; κ1 independent candidate; damping .01 with the supplied original adaptation; 30 outer checks, 16 scalar-dual updates, 8 backtracking trials; one absolute 20 ms soft deadline starts before adapter conversion. The prototype count is corrected to include the already-executed first QP even if the dual loop immediately reaches the deadline. All work, including final verifier and result construction, is timed. Calls are not preemptible; late commands are retained, not claimed hard real-time.

## Locked finite evidence plan

- First accept the supplied vectors without any IK substitution. Compare exact source UID/frame/target/previous_q/dt; check all arrays for the package's previously observed reconstructed 160 Panda trajectories. Local package timings are not server timings or independent evidence.
- Run the supplied 48 convex checks against cold-start SLSQP epigraph solutions; regress fixed bounds, external rejection, shared deadline, identical first QP and invalid floating bounds.
- Reuse exactly 52 historical unique inputs, including a separately reported direct original-GN-source subset of 16. GN, historical balance, cached tight and relative are compared at both κ, three nested repetitions.
- Reuse all 40 development trajectories per robot, 150 frames, three repetitions, all four families. Compare the above eight methods plus the frozen Pink adapter. Do not select a new κ or forcing value from these outcomes.
- Reuse exactly 360 frozen real local problems. Compare historical tight, cached tight, relative and existing cached Clarabel epigraph. Tight-quality and relative-progress results are distinct; separately measure actual command verification, including a Clarabel solve followed by the same verifier. Clarabel does not expose the same task-eligible early stop; full optimality work is not mislabeled as matched early-acceptance latency.
- Before new outcomes, lock code, parameters, 2,000 independently drawn witness-feasible queries per robot (500 local / near-singular / near-limit / high-utilization) and 80 new trajectories per robot (20 per original family, 150 frames). Reuse the original geometry recipe unchanged; new seed/UID/query-hash identities only. Point queries do not feed back; trajectories update only their own accepted state and keep previous state after failure while advancing all targets. q_ref proves reference feasibility but is never an online seed.
- Independent methods, all κ1 where applicable: original GN, historical balance, cached tight, relative, aligned TRAC 5 ms and 20 ms. Three interleaved repeats, CPU 4 and single-thread numerical libraries. No new environment/backend or configuration search.

Queries or full trajectory UIDs are statistical units. Average three repeats within UID, then use the existing family-stratified paired 4,000-resample bootstrap for descriptive unadjusted 95% intervals. Report both robots separately, all families, failures, late calls, errors, accepted error utilization >90%, acceleration, and gained/lost UIDs. Zero differences or intervals containing zero do not demonstrate equivalence. No global gate or outcome-triggered extra experiment is defined.

The numerical-comparison wrapper was finalized before its first results to avoid duplicate Clarabel quality checking and to include all four actual-command timing controls. Development code/runtime hashes and measurements are retained; `protocol/measurement_seal.json` records the wrapper-only revision. No solver or setting changed after development.

## Prior foundations, not transferred guarantees

The `nature-academic-search` citation-verification workflow checked the three DOIs against Crossref and publisher/author pages; the academic MCP was unavailable and web Crossref access failed, so direct Crossref HTTP was used. Metadata is saved under `literature/`.

| Source | Established content and boundary here |
|---|---|
| Wang, Praveena, Rakita, Gleicher, **RangedIK**, ICRA 2023, 9700–9706 ([DOI](https://doi.org/10.1109/ICRA48891.2023.10161311), [author page](https://graphics.cs.wisc.edu/Papers/2023/WPRG23/)) | Ranged and preferred tasks are established; this task does not establish superiority over RangedIK or priority for tolerance-aware IK. |
| Boyd & Vandenberghe, **Convex Optimization**, Cambridge, 2004 ([author book](https://web.stanford.edu/~boyd/cvxbook/)) | Epigraph reformulation, duality and bound-based stopping are foundations, not newly invented principles. |
| Porcelli, **On the convergence of an inexact Gauss–Newton trust-region method for nonlinear least-squares problems with simple bounds**, Optimization Letters 7, 447–465 (2013; online 2011) ([publisher](https://link.springer.com/article/10.1007/s11590-011-0430-z)) | Controlled inexact subproblem solution already exists. Its trust-region convergence result does not transfer to this task-accepting finite minimax loop. |
| Walwil & Fercoq, **The smoothed duality gap as a stopping criterion**, Mathematical Programming Computation 17, 653–697 (2025) ([publisher](https://link.springer.com/article/10.1007/s12532-025-00284-0)) | Computable termination measures to avoid excessive work are established for a different affine-equality convex formulation. Its error-bound assumptions and guarantees are not claimed here. |

The candidate contribution to evaluate is the task-specific composition of the retained minimax target, low-dimensional dual and task-acceptance/model-progress-controlled effort. Novelty and practical value must follow the independent results, not the number of tests.
