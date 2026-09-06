# Supplementary computation-allocation evaluation

This protocol is a new supplementary evaluation, not a retroactive preregistration
of the existing results. Existing evidence and the released CG-HIK are immutable.
The previous story rewrite in the working tree is retained until the experiments
and findings report are finished. No new model, solver, verifier, or temporal
method is introduced.

## Sequence and population

A reads existing fresh trajectories and action-complete policy-validation data;
it makes no solver calls. Four completion groups retain all 150 frames and exactly
reconstruct total latency and FEV. Outcome-conditioned groups are descriptive.
The routing comparison uses one common successful, non-abstained policy-validation
population. Historical label timing is not substituted for unmeasured deployment
overhead; five-repeat empirical quantiles are noisy diagnostics.

B uses 3,000 fresh paired queries per robot, including 2,500 verifier-checked
witnesses, with three interleaved repetitions. Query state does not evolve. A
category-stratified 500-query subset is fixed before execution for two-fold
five-repeat entry-selection/evaluation swaps (ten repetitions per entry total).
All repetitions remain nested within the query in interval calculations.

C uses 40 fresh, geometry-selected reference paths per robot, four families,
150 frames each. All methods start from the reference initial state; only their
accepted commands update their own state. Failure holds state but target indexing
continues. Online methods receive no reference joints. Completion, acceptance
within measured 20 ms, and all-frames deadline completion are distinct endpoints.

## Exact method mapping

| Method | Reject | Defer | Eligibility | Entry |
|---|---|---|---|---|
| Always-hard | none | none | none | hard, with released escalation/fallback; no diagnostic routing features or predictor |
| Geometry rule | unchanged | full easy-entry cascade | original eligibility only determines abstention | easy if both geometric steps meet development-selected bounds, otherwise hard |
| Reject-only + hard | unchanged | full easy-entry cascade | unchanged | all non-abstained entries become hard |
| Routing-only | original reject becomes hard | full easy-entry cascade | unchanged | original P95 choice |
| P50-selection | unchanged | full easy-entry cascade | original success and P95 deadline criteria | minimum predicted P50 within eligible entries, same conservative-entry tie framework and margin |
| Full CG-HIK | unchanged | unchanged | unchanged | exact released policy/runtime |

Geometry features are the existing target-vs-previous position/orientation steps;
thresholds use calibration records only. Learned candidates remain part of all
internal numerical paths; only Always-hard omits unneeded routing diagnostics.
All returned commands remain subject to the original verifier.

TRAC-IK uses the official library, Speed mode, previous-state initialization,
5/20/100/400 ms budgets and the same URDF chain, limits and final verifier.
Its trajectory budget is selected from independent existing calibration queries
before opening B outcomes, never from a test winner. Internal tolerance and limit
handling are recorded with the adapter. External FEV is not equated to internal FEV.

## Timing, uncertainty and figures

Outer-call timing includes conversion, actually required prediction/decision,
candidate generation, numerical solution and acceptance verification. Initialization,
warmup, diagnostic replay and serialization are excluded. Quantiles come from
whole-call samples, not sums of stage quantiles. Paired, family-stratified bootstrap
95% intervals use queries or complete trajectories; repeats/frames are not extra
independent observations. Intervals are descriptive, without multiplicity-adjusted
significance claims or a composite pass/fail gate.

Figure contract: Python quantitative panels, editable PDF/SVG and PNG previews,
at least 7 pt labels. The cost-decomposition panel shows which completion groups
account for total savings (including negative contributions); mechanism panels
compare paired workloads; external curves count only commands actually verified
within each elapsed-time threshold. Data are never trimmed to improve a result.
