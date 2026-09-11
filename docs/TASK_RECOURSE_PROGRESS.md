# Task-admissible recourse: development protocol and findings

Baseline: `e9b621a3cfb5b6b4d2d0a0f2588a3d118ff7f02e`. This is observed-data
algorithm development, not a fresh test or a novelty/performance conclusion.
Old paper, methods, configurations and results are immutable.

## Final conclusion — one-page Problem–Method–Evidence

Completed 2026-09-11. **The numerical formulation is implemented and its
mathematical distinctions are reproduced, but this development comparison does
not establish a worthwhile online task advantage.** No numerical setting was
changed after the integration outcomes; no formal evaluation was started.

**Problem.** A fixed right-inverse policy need not represent the actual ability
to correct a task when pose residual and joint allocation are free. This is a
valid reason to test direct recourse, not evidence that direct recourse will
improve closed-loop tracking.

**Method.** TAR-IK directly optimizes one shared current command and 13 free
next-node configurations with a normalized future pose-shortfall penalty. Its
current command remains subject to the unchanged verifier. The nominal and
fixed-compensation controls use the same sparse numerical kernel. No learned
policy, scenario-specific current command, extra fallback chain, or future
target is supplied online.

**Evidence and the five requested answers.**

1. **Both proxy-ranking counterexamples are handled correctly.** In the actual
   SOCP, tolerance case A/B gives true shortfall 0.3/0; redundant-allocation
   case A/B gives 0/0.2. Rank-zero tolerance and shared-current regressions also
   pass. These establish implementation properties, not algorithm novelty.
2. **Free recourse does not consistently beat fixed compensation on robots.**
   Panda TSR is 87.50% versus 91.67% (paired difference −4.17 pp, descriptive
   95% interval [−9.17, 0.00]); UR5e is 95.00% versus 92.50% (+2.50 pp
   [−5.00, 10.00]). The UR5e mean gain coexists with near-singular losses.
3. **Multiple target changes do not show an overall completion advantage over
   nominal prediction.** Panda loses 4.17 pp; UR5e has the same observed mean
   TSR, with a paired difference interval [−8.33, 8.33] pp, not evidence of
   equivalence. Same-input nonlinear corrections also do not favor free recourse.
4. **The extra cost is not justified by these task outcomes.** Free recourse
   costs 13.58×/20.01× task TRAC 5 ms and 4.78×/6.37× Pink on Panda/UR5e.
   Its DTSR20 is 84.17%/91.67%, below those controls. Most frame calls fit the
   nominal deadline, but that does not imply whole-trajectory deadline success.
5. **Supported claims:** a functioning shared-command recourse kernel; the two
   mathematical counterexamples; real verified correction records; mixed,
   robot/family-dependent development outcomes. **Not established:** superior
   correction capability, generally better completion, worthwhile online cost,
   a nonlinear robust guarantee over a region, or novelty over tolerance-aware,
   predictive and adjustable-optimization work. The study stops here.

Source: [complete main table](../outputs/task_recourse/reports/main_table.csv),
[paired intervals](../outputs/task_recourse/reports/paired_comparisons.csv),
[same-input results](../outputs/task_recourse/reports/same_input_correction.csv).
The independent units are 40 trajectory UIDs per robot, not 120 independent
trials or individual frames; three runs are averaged within UID.

## Complete development results

Each completion entry lists the three counts out of 40. Cumulative time is the
mean total for a complete 40 × 150-frame sweep, **including failed and late
frames**. Frame latency quantiles pool the three runs descriptively. An equal
completion count does not imply the same completed UID set.

| Robot | Method | TSR counts | DTSR20 counts | Frame P50 / P95 / P99 (ms) | Cumulative time (s) |
| --- | --- | --- | --- | --- | ---: |
| Panda | Task TRAC 5 ms | 37 / 37 / 36 | 37 / 36 / 35 | 0.167 / 0.314 / 5.239 | 1.965 |
| Panda | Task TRAC 20 ms | 37 / 37 / 38 | 36 / 37 / 38 | 0.166 / 0.291 / 20.264 | 3.528 |
| Panda | Pink | 37 / 37 / 37 | 37 / 37 / 37 | 1.002 / 1.075 / 1.498 | 5.587 |
| Panda | Same-core nominal | 37 / 37 / 36 | 37 / 36 / 35 | 1.717 / 2.485 / 6.208 | 11.300 |
| Panda | Same-core fixed compensation | 37 / 36 / 37 | 36 / 34 / 36 | 4.347 / 5.941 / 7.376 | 25.571 |
| Panda | TAR free recourse | 35 / 35 / 35 | 33 / 34 / 34 | 4.453 / 7.055 / 7.576 | 26.676 |
| UR5e | Task TRAC 5 ms | 39 / 38 / 39 | 39 / 37 / 38 | 0.156 / 0.239 / 0.702 | 1.228 |
| UR5e | Task TRAC 20 ms | 39 / 38 / 39 | 38 / 38 / 39 | 0.156 / 0.238 / 20.216 | 2.327 |
| UR5e | Pink | 38 / 38 / 38 | 38 / 37 / 38 | 0.633 / 0.685 / 0.876 | 3.858 |
| UR5e | Same-core nominal | 39 / 37 / 38 | 38 / 35 / 38 | 1.646 / 1.957 / 6.193 | 10.751 |
| UR5e | Same-core fixed compensation | 37 / 38 / 36 | 37 / 37 / 35 | 4.103 / 5.948 / 7.269 | 23.364 |
| UR5e | TAR free recourse | 38 / 38 / 38 | 38 / 37 / 35 | 4.150 / 5.324 / 7.014 | 24.578 |

The cost ratios free/fixed are 1.043 [1.016, 1.072] on Panda and 1.052
[1.002, 1.104] on UR5e. Free/nominal costs 2.361 [2.215, 2.503] and 2.286
[2.054, 2.475]. Free/TRAC 5 ms costs 13.579 [9.368, 21.033] and 20.013
[16.446, 23.602]. Intervals are descriptive, family-stratified and unadjusted;
no equivalence, significance-driven selection or overall pass/fail gate is used.

### Every trajectory family

Counts are out of ten for each repeat. The full 48-row
[family table](../outputs/task_recourse/reports/family_table.csv) also retains
deadline completion, cumulative cost, latency, pose errors and joint motion.

| Robot / family | TRAC 5 ms | TRAC 20 ms | Pink | Nominal | Fixed | Free |
| --- | --- | --- | --- | --- | --- | --- |
| Panda / smooth | 10/10/10 | 10/10/10 | 10/10/10 | 10/10/10 | 10/10/10 | 10/10/10 |
| Panda / near-singular | 8/8/7 | 8/8/8 | 8/8/8 | 8/8/7 | 8/7/8 | 6/6/6 |
| Panda / joint-limit-return | 9/9/9 | 9/9/10 | 10/10/10 | 9/9/9 | 9/9/9 | 9/9/9 |
| Panda / high-curvature | 10/10/10 | 10/10/10 | 9/9/9 | 10/10/10 | 10/10/10 | 10/10/10 |
| UR5e / smooth | 10/10/10 | 10/9/10 | 10/10/10 | 9/9/8 | 9/9/8 | 10/10/10 |
| UR5e / near-singular | 9/9/9 | 9/9/9 | 9/9/9 | 10/9/10 | 9/10/9 | 8/8/8 |
| UR5e / joint-limit-return | 10/9/10 | 10/10/10 | 10/10/10 | 10/10/10 | 9/9/9 | 10/10/10 |
| UR5e / high-curvature | 10/10/10 | 10/10/10 | 9/9/9 | 10/9/10 | 10/10/10 | 10/10/10 |

Panda's stable loss against both same-core controls includes trajectory_16,
UID `40c3eae024871d71025c88c09b1bbffc1b50b311548d53112ed8d4118883654a`.
On UR5e, free recourse recovers smooth trajectory_02 against nominal/fixed in
all three runs (UID `f87cbb198e6aa95e16ea15160dd1d4f545d70b0d61622238ff991b3a2d27786a`),
but loses near-singular trajectory_18 against both controls (UID
`17d92c7319719a45d08d673e5023c3f00687e4f5a57bdebde3dccc6370b94c3a`).
The full [gained/lost UID table](../outputs/task_recourse/reports/gained_lost_uids.csv)
includes fractional repeat changes rather than forcing every UID into a binary
win or loss. [Per-repeat completion sets](../outputs/task_recourse/reports/completion_uids.json)
remain authoritative.

UR5e trajectory_15, UID
`bf166cd6db08076d8c85cfd7e7364d2446c3c1ec592ea82609b812c0faa204ea`, is **not
recovered**: all three free-recourse runs first fail at frame 125. In repeat 0,
the actual previous first joint is 6.279039176 rad, task TRAC reports
`native_no_solution` (−3), and the single attempted cone update ends `MaxTime`.
Its three nonlinear trials fail the current pose contract; one also violates a
joint bound. No command is emitted. The top-level `non_finite_or_wrong_shape`
label is the no-command sentinel, not proof that native arithmetic produced
NaNs or that this input is mathematically infeasible. This record does not
identify singularity, branch evolution or a shortcut as the unique cause.

### Accuracy, motion and actual cost

Free recourse's accepted position P95 is 0.976/0.979 mm on Panda/UR5e, versus
0.392/0.396 mm for TRAC 5 ms and 0.196/0.130 mm for Pink. Its orientation P95
is 0.005694/0.007588 rad; errors remain inside 1 mm / 0.5°, but using more of
that tolerance is not itself a tracking benefit. All methods have zero accepted
contract violations in the saved runs and independent reconstruction.

Free-recourse acceleration RMS averages 5.521/12.129 rad/s², versus
4.151/10.357 for TRAC 5 ms, 4.204/12.106 for Pink, and 4.992/12.725 for fixed
compensation. Failure holds are included; this is commanded-state finite
difference acceleration, not a measured physical robot response.

Free calls optimization on 100.00%/99.994% of frames and changes the legal
backup command on 63.04%/77.91%; mean normalized intervention is 0.02852/0.03554.
Its recorded decisions are: Panda 11,348 changed commands, 5,877 unchanged
backups, nine geometric recoveries and 766 failures; UR5e 14,023 changed
commands, 3,862 unchanged backups, three geometric recoveries and 112 failures.
These are frame counts within the 120 trajectory runs, not independent samples.

Most incremental time is not the cone solver itself: free-recourse median
nonlinear checks cost 1.434/1.399 ms and final command-plus-node verification
1.265/1.143 ms; cone build is 0.400/0.373 ms and cone solve 0.118/0.108 ms.
Backup, prediction, model construction and record assembly also remain inside
total timing. Stage medians are not additive estimates of the total median.
No stage is subtracted to claim online speed.

## Same-input real correction results

Every method sees the same saved TRAC-derived previous state at frame 80 for
each of the 40 UIDs/robot. Each target has three downstream TRAC 20 ms searches.
The table averages searches/directions within state and then all 40 states;
unavailable current commands stay in the denominator. Axial nodes are part of
the optimization; mixed directions are not. Mixed-outside nodes have normalized
L1 radius 1.5 and carry no in-set guarantee.

| Robot | Current-command method | Optimized axes (%) | Mixed inside (%) | Mixed outside (%) | Actual next (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Panda | TRAC 5 ms | 96.875 | 97.500 | 97.500 | 97.500 |
| Panda | TRAC 20 ms | 96.875 | 97.500 | 97.500 | 97.500 |
| Panda | Pink | 96.875 | 97.500 | 97.500 | 97.500 |
| Panda | Nominal | 96.875 | 97.500 | 97.500 | 97.500 |
| Panda | Fixed | 96.875 | 97.500 | 97.500 | 97.500 |
| Panda | Free | 96.875 | 97.500 | 97.500 | 97.500 |
| UR5e | TRAC 5 ms | 98.264 | 100.000 | 100.000 | 100.000 |
| UR5e | TRAC 20 ms | 98.333 | 100.000 | 99.861 | 100.000 |
| UR5e | Pink | 98.542 | 100.000 | 100.000 | 100.000 |
| UR5e | Nominal | 98.264 | 100.000 | 100.000 | 100.000 |
| UR5e | Fixed | 98.333 | 100.000 | 99.861 | 100.000 |
| UR5e | Free | 98.125 | 100.000 | 100.000 | 100.000 |

Panda has 39/40 available current commands for every method, UR5e 40/40.
The observed Panda correction rates coincide; this is not an equivalence test.
UR5e free-minus-fixed axial success is −0.208 pp [−0.556, 0.000], and
free-minus-nominal is −0.139 pp [−0.417, 0.000]. Tiny differences among
three stochastic reference searches do not establish improved correction.
The actual next target is inside the L1 set at 30/40 Panda and 28/40 UR5e
mechanism states; inside/outside strata are both retained. Across all online
frames with a next target, coverage is only 72.40%/72.50%, identical across
methods because it depends on targets, not outcomes.

There are 480 current calls, 37,440 downstream rows, 36,972 actual downstream
calls and 36,770 saved verified correction witnesses. Every witness was
rechecked from its own method's returned current command. Missing-current rows
are explicitly unavailable, not substituted with a reference joint state.
Witness configurations, targets, current inputs, residuals and reference times
are in [Panda records](../outputs/task_recourse/mechanism_panda/) and
[UR5e records](../outputs/task_recourse/mechanism_ur5e/); `witness=true` selects
the legal sequences. This offline reference time is separate from the online
comparison; failure only means that this reference search did not find a command.

### Affine prediction is not nonlinear acceptance

The [matched-model table](../outputs/task_recourse/reports/affine_nonlinear_summary.csv)
compares the same backtracked q/z trial under its original affine expansion and
true FK, with all trials and adopted trials separately retained. For adopted
free-recourse trials, the P95 absolute normalized-shortfall discrepancy is
0.00100/0.00149, with maxima 0.238/0.382. Four Panda and 13 UR5e adopted trials
have affine shortfall ≤1e−5 but actual shortfall >1e−5. These remain planning
variables, not zero-slack witnesses. Fixed-compensation rejected trials can
have very large nonlinear discrepancies; those are not deleted or interpreted
as actual executed commands.

Only 11.08%/15.02% of all free-recourse frames have all 13 planned nodes pass
the public verifier. Lower actual shortfall, even after a true merit decrease,
does not mean full recourse feasibility. Verifying the nodes also does not
prove nonlinear feasibility throughout their convex hull. This distinction is
why the mechanism measurements and complete closed-loop outcomes are reported
separately from the optimization objective.

## Delivery and verification

The sole entry is `scripts/run_task_recourse.py`; its `prepare`, `test`,
`integration`, `development`, `mechanism`, `report`, `audit` and `figures`
actions separate new solves from read-only reporting. Outputs are exclusive:
existing result directories are not silently replaced. A rerun is not needed
to inspect the delivered results.

- 1,440 complete development runs, 216,000 frames; zero dropped failures/timeouts.
- All 216,000 saved current outputs and feedback transitions independently
  checked; 96,989 saved plans and 100,170 adopted nonlinear trials checked.
  Public/native shortfall discrepancy is at most 6.54e−13 in final plans.
- Seven regression/unit tests pass, including real Panda/UR5e SO(3) derivatives,
  fixed-rank-zero behavior, sparse update stability and exact zero-cost return.
- Run manifests preserve the executed source hashes. Post-run additions only
  construct matched-model reports, statistics and figures; the numerical kernel
  and configuration remain identical to those used for full development.
- [Paired task/cost figure](../outputs/task_recourse/figures/paired_task_cost.pdf),
  editable SVG, PNG and plotted CSV are provided. Statistical reporting followed
  the nature-statistics skill (UID-level inference), and the nature-figure skill
  prompted an actual PDF glyph-size check and correction of small log exponents.

The executed native versions are Pinocchio 3.9.0, Clarabel 0.11.1, Pink 3.3.0,
and the existing TRAC-IK 2.2.0 build from upstream
`90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb`. NativeGeometry uses Pinocchio for
derivatives/FK inside optimization; the unchanged final verifier uses the actual
`URDFKinematics` backend, not a backend inferred from the environment name.
Its agreement with the native model is checked at adapter construction.

Figure QA: the final 183 × 116 mm PDF has minimum extracted text size 6.5 pt,
no undersized text, and all six panels were visually inspected. Source checks
have no failures. Their three warnings are documented: PNG is only a 360 dpi
preview (SVG/PDF are the vector deliverables), no TIFF was requested, and the
single-entry script's random job ordering is not simulated plotting data.
The plot consumes frozen aggregate estimates and intervals without resampling.

No old manuscript, model, runtime, configuration or frozen output was modified.
No next experiment, weight search, new solver or positive paper abstract follows
this report automatically.

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

## Related work: bounded comparison, checked 2026-09-11

| Primary source and reading scope | What already exists | Specific distinction / remaining burden here |
| --- | --- | --- |
| Wang, Praveena, Rakita, Gleicher, *RangedIK*, ICRA 2023, DOI [10.1109/ICRA48891.2023.10161311](https://graphics.cs.wisc.edu/Papers/2023/WPRG23/); [author manuscript §§III–V](https://arxiv.org/html/2302.13935) | Ranged and preferred goals enter a weighted multi-objective optimization through relaxed barrier losses; pose flexibility serves smooth motion, limits and other tasks. | Using pose tolerance to improve motion is not new. Here one shared current command and freely adjustable next-node configurations minimize actual normalized future pose shortfall. No claim of superiority to RangedIK: its inadequate historical adapter is not a performance baseline. |
| Schuetz, Buschmann, Baur, Pfaff, Ulbrich, *Predictive online inverse kinematics for redundant manipulators*, ICRA 2014, pp. 5056–5061, DOI [10.1109/ICRA.2014.6907600](https://portal.fis.tum.de/en/publications/predictive-online-inverse-kinematics-for-redundant-manipulators/) | The authors' institutional abstract describes moving-horizon optimal control via Pontryagin's minimum principle and conjugate gradients, motivated by high velocities from instantaneous IK. | Anticipation and online predictive IK are established. Our nominal control isolates only the scenario set in our own numerical kernel, not a reproduction of this paper. Full text was not retrieved; the abstract does not establish exactly how all its constraints compare with this contract. |
| Ben-Tal, Goryashko, Guslitzer, Nemirovski, *Adjustable robust solutions of uncertain linear programs*, Mathematical Programming 99, 351–376 (2004), DOI [10.1007/s10107-003-0454-y](https://www2.isye.gatech.edu/~nemirovs/MP_Elana_2004.pdf); author-hosted published paper §§1–3 | Decisions made before uncertainty and adjustable decisions afterward are distinguished. General adjustable counterparts can be intractable; affine decision rules give tractable cases/approximations. | Shared q and scenario-dependent z are an established nonanticipativity pattern, not our theorem. Fixing a pseudoinverse B is more restrictive than optimizing an affine decision rule; our fixed compensation control is NOT a general AARC baseline. Thirteen nonlinear FK-checked nodes do not prove all nonlinear realizations in their hull. |
| Stradovnik and Hace, *Task-Oriented Evaluation of the Feasible Kinematic Directional Capabilities for Robot Machining*, Sensors 22, 4267 (2022), DOI [10.3390/s22114267](https://www.mdpi.com/1424-8220/22/11/4267); [author article XML §2.4 via Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9185573/fullTextXML) | Decomposed Twist Feasibility couples translation/rotation directions and their task-dependent speed ratio under component joint-speed limits. The derivation in §2.4 assumes a nonsingular nonredundant serial arm. | Directional, component-constrained capacities and the drawbacks of unscaled scalar manipulability are already known. Our variables are finite configurations with pose tolerance and flexible redundant recourse, evaluated with nonlinear FK. This does not establish a generally better capacity characterization. |
| Zolotas, Long, Sagar, Padir, *constrained_manipulability*, JOSS 10(108), 7481 (2025), DOI [10.21105/joss.07481](https://joss.theoj.org/papers/10.21105/joss.07481.pdf); full six-page paper | C++/ROS library builds constrained velocity and allowable-motion polytopes from joint, velocity and environmental constraints, exposing vertex and half-space representations. | Feasible motion sets and polytope representations are not new. TAR is a sparse command-generation program; it does not enumerate polytopes or include the library's collision-world capabilities. |

Native tool documentation was checked against the frozen installed source:
[TRAC-IK upstream](https://github.com/traclabs/trac_ik/tree/90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb)
documents Speed, time limits and six Cartesian tolerances; Pink 3.3.0's
`ConfigurationLimit` and `VelocityLimit` docstrings/code define the displacement
inequalities used by its unchanged adapter ([official project](https://github.com/stephane-caron/pink/tree/v3.3.0)).
Clarabel's [data-update documentation](https://clarabel.org/stable/user_guide_data_updating/)
requires unchanged dimensions/sparsity and compatible presolve settings; our
fixed sparse template disables presolve/chordal structure changes. These are
existing tools, not algorithm contributions. Some web documentation endpoints
were unavailable; versioned local official code was read instead. No environment
upgrade or new solver was performed.

## Figure contract (before drawing)

1. Question: do the task outcomes gained by free recourse justify its cost
   relative to each of the five prespecified controls?
2. Signal: paired TSR/DTSR20 differences and cumulative-time ratios; the six
   panels retain both robots rather than average their directions together.
3. Evidence: frozen `reports/source_data.json`, 40 trajectory UIDs/robot, three
   runs averaged inside UID; no new fit, subset or solver call for the figure.
4. Encoding: point estimates and descriptive, family-stratified 95% intervals;
   zero difference / unit ratio reference lines; a labeled log scale for cost.
   All controls are displayed, and intervals spanning zero do not imply equality.
5. Delivery: original Python/Matplotlib drawing, 183 × 116 mm, readable vector
   text, editable SVG and PDF plus PNG, exact plotted CSV and source hash.
   Inspect the full figure and text extraction before delivery. This figure is
   a development result, not an illustration of proven robustness.
