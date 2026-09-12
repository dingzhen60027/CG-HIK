# Fixed constrained-QP mechanism follow-up

Baseline be75d28ceadd519401c699cb9d5f125f3c5bc4fe. No new IK method, target,
seven-setting evaluation, or historical timing replacement.

1. The 971/1000 old development QPs remain a separate natural-bank cohort.
2. Diagnostic capture replays each saved GN kappa=1 repeat-0 **exact frame input**
   of the existing 160 trajectories/robot once. Each previous_q comes from that
   frame's original accepted history, not a new counterfactual trajectory.
   Capture overhead and its possible effect on the unchanged soft deadline are
   logged, including output discrepancy against the saved call. This is not a
   new online performance result. All captured matrices and chronological census
   metadata are retained. Post-solve diagnostics run outside the IK call.
3. Strong active constraints require proximity <=1e-8, correct multiplier sign
   and magnitude >1e-9, on independently KKT-qualified solutions. Weak contacts
   and unqualified results are separate. Physical, fixed-frame displacement,
   and per-iteration trust-step boundaries are separately attributed.
4. A targeted cohort takes constrained or multi-update QPs from that replay.
   If <=2000/robot retain all; otherwise round-robin UID allocation with fixed
   seed 2026091501, then restore original chronology. No speed/outcome selection.
   Working-set changes between adjacent original QPs and internal active-set
   additions/releases are distinct recorded concepts. Cohorts are not pooled
   into an overall mean or prevalence claim.
5. Official qpOASES releases/3.2.1, SQProblem changing-Hessian hotstart, cached
   double-buffered H, default double-precision options except PL_NONE, nWSR cap
   1000; no retry. Existing OSQP settings unchanged. Common quality: box<=1e-8,
   raw projected KKT<=1e-5, normalized projected KKT<=1e-8, scaled objective gap
   <=1e-8. Only quality can motivate native precision correction; preserve all
   attempts. NumPy original 50-update cap unchanged.
6. Five timing passes, method order rotated, each selected UID chronological;
   warm starts reset at UID boundaries. Time initial workspace and first solve
   separately, plus kernel and full conversion/update/quality-check call.
   All QP failures retained. Single logical CPU4 and one BLAS/OMP thread.
   Timing repeats average within QP, QPs within source UID; descriptive paired
   bootstrap at UID level (4000, seed 2026091502), no pseudo-independent calls.
7. Six prescribed gain/loss trajectories: earliest local direction difference
   with infinity norm >1e-6 in step-normalized coordinates. Compare bounded and
   clip on identical H/g/lo/hi and identical linearization state; independently
   evaluate nonlinear FK and original eight-level line search. Existing other
   method states/completion are linked but not substituted or claimed causal.
8. One Python quantitative-grid figure: local direction redistribution and
   actual nonlinear cost reduction, followed by historical outcomes of all six
   requested trajectories. Raw paired cases, no inferential error bars. Editable
   SVG/PDF and PNG; 183mm width, >=5pt text; scientific data, no generated artwork.
