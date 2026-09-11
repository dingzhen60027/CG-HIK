# Task-admissible recourse: development protocol and progress

Baseline: `e9b621a3cfb5b6b4d2d0a0f2588a3d118ff7f02e`. This is observed-data
algorithm development, not a fresh test or a novelty/performance conclusion.
Old paper, methods, configurations and results are immutable.

## Problem–Method–Evidence (protocol, before outcomes)

**Problem.** A specified inverse map measures one compensation policy, not all
task-admissible recourse when residual tolerance and redundant allocation remain
available. The supplied scalar counterexamples motivate a different object;
they do not identify the cause of every robot failure.

**Method.** One shared executable q and 13 free future z variables minimize
`0.5 ||S^-1(q-q_anchor)||² + 0.5 tau²`. Current pose, joint and rate constraints
remain hard; only future pose has tau slack. S is velocity*dt+epsilon_v.
The nodes 0, ±e_i perturb the constant-motion prediction in world coordinates,
using left SO(3) perturbations and public pose tolerances. Their convex hull is
the normalized L1 unit ball, not a probability region. All scenarios have one
common future expansion point and target-side SO(3) derivative. Each of at most
two updates is one sparse Clarabel SOCP. No z regularizer, gamma, demand window,
candidate pool, future truth, or inherited Elastic policy is used.

**Numerics.** Unchanged installed native tools, 0.5 normalized trust radius,
40 Clarabel iterations/2 ms native limit, 18 ms outer soft limit, backtracking
1/.5/.25. Current SOC uses the existing 1e-5 interior; future pose radius is
exactly 1+tau. Native rate box has a 1e-10 normalized roundoff interior; the
public verifier is unchanged. Merit acceptance requires true FK geometry and
improvement exceeding `1e-10 + 1e-8*abs(old_merit)`. Non-success native iterates
are recorded as such and can only be adopted after the same true checks.
No optimizer status grants acceptance. Native setup, sparse updates, backup,
prediction, FK and final public verification are all inside outer timing.

**Controls.** Task TRAC 5/20 ms, fixed Pink, same-core nominal (one node),
same-core fixed compensation (13 nodes, z_j=z_0+B w_j), and free recourse.
Fixed B is a scale-consistent least-squares inverse at the common expansion,
including deficient rank. Its initial mapped plan can violate future bounds;
then only a geometrically feasible program result establishes a merit reference.
Its backtracking reconstructs the fixed mapping rather than relaxing equality.
Free/nominal initial z_j=q_anchor always has finite slack if the anchor is legal.
Zero actual merit returns that exact anchor without creating a SOCP. A failed
optimization has only the already computed task TRAC 5 ms backup.

**Development inputs.** Reuse exactly the original 40 Panda and 40 UR5e targets,
four families of ten, 150 frames, dt=.02. Integration takes the first two entries
of each family in original file order, one run. Full comparison has three runs
of each whole trajectory per method, interleaved with a fixed ordering seed.
Pink repeats measure timing, not extra independent evidence. No failures or
timeouts are excluded. Methods receive their own actual accepted previous state;
failure holds it while targets advance. The input identity file is written
before any integration result. No existing fresh/locked evaluation is rerun.

**Same-input correction.** At fixed frame 80 for every development UID, take the
actual previous state of task TRAC 5 ms repeat zero. Retain even histories with
earlier failure. All six methods see this same previous q, current target, and
previous target; initialize policy history explicitly, with no future q or
method-specific warm-start advantage. If a current command fails, record it and
count its downstream availability as failed, rather than substitute a state.
An offline task TRAC 20 ms reference, three searches, tests 13 optimized nodes,
mixed ± directions on axis pairs (0,3)/(1,4)/(2,5) at L1 radii .75 and 1.5,
and the real next target. Mixed nodes were not optimized. All returned witness
commands are verified from that method's actual current command. This is a
bounded numerical reference, not an infeasibility certificate. Its time is
separate from online timings. Future truth is used only after online solving.

**Statistics.** Each robot has n=40 independent trajectory UIDs. First average
three runs inside UID, then form paired differences and cumulative-time ratios
with 4000 family-stratified bootstrap resamples (fixed seed). Descriptive 95%
percentile intervals are unadjusted for multiplicity; no global gate, equivalence
claim, significance-driven extra sampling, winsorization or favorable grouping.
Pooled frame quantiles describe runtime; frames/repeats are not independent n.
Main paired contrasts: free vs fixed, nominal, TRAC 5/20 and Pink. All family
rows, errors, losses, timeouts, and motion costs remain visible.

**Evidence still required.** The actual SOCP must reproduce both scalar ranking
reversals, rank-zero tolerance, common-current nonanticipativity, and affine
interpolation. Robot task gains and cost justification must come from the full
development table and same-input nonlinear records, not reduced tau alone.

## Progress

Task package extracted and read completely. Source archive SHA256:
`e775572642e21155abb8d99aed35503f90349a554aa6deecd6c6d47c33bcaa57`.
Actual joint-SOCP regressions and seven unit tests passed. The two-robot fixed
integration set completed (48 whole-trajectory calls per robot, 150 frames each).
All current outputs, causal feedback, future joint/rate bounds, final node
verifications, fixed-map equalities, and recorded merit were independently
recomputed without new IK calls. UR5e free recourse lost the selected
near-singular trajectory_10 in this integration run; it is retained. No numerical
solver, derivative, mapping, target or budget was changed based on integration
outcomes. Only a read-only validation entry was added before the full run.

The ignored `outputs/` policy required a separate explicit evidence commit:
code/protocol `30c34d47`, input identities and mathematical records `b48d17ee`.
Both precede the full development comparison. Original package bytes and newly
executed mathematical results live in separate directories.

Run with the already installed environment (no upgrades):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=tmp/crik_dependencies/python:src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_recourse.py development --robot panda
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_recourse.py development --robot ur5e
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_recourse.py mechanism --robot panda
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_recourse.py mechanism --robot ur5e
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_recourse.py report
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_recourse.py audit
```

Existing directories are never overwritten by these commands. Recorded runs use
separate equivalent performance cores (Panda 0/2, UR5e 4/6); all methods for a
robot run in the same process resources and interleaved order. Reporting and
read-only validation use another core.
