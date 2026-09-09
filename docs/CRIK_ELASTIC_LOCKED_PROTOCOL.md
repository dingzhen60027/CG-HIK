# Elastic CR-IK: fixed-candidate independent evaluation

## Candidate decision (before new outcomes)

Freeze Elastic CR-IK **mu=0.25 for both robots**, as explicitly selected from the
already observed development comparison at `01d1dd1c82b97920f67cdf12b3abc539a5329523`.
This is a development-based choice, not an optimum or established superiority.
The UR5e development completion increment and its uncertainty, the absence of a
Panda completion increment over mu=0, and all other development results remain
unchanged. Those trajectories and the previously observed 160 formal trajectories
are not evaluation samples here. No old numerical file, model, paper or result is
edited. There is no new parameter search or universal performance gate.

## Frozen methods and the sole cost correction

Six settings: Elastic mu=0.25; `cr_ik_mu0_analytic`; original same-core ordinary
two-step predictive IK; task-aligned TRAC-IK 5 and 20 ms; fixed Pink. Every setting
runs three whole-trajectory repetitions on both robots. No old CR-IK, hard Minimal,
other mu, RangedIK or historical ablation is rerun.

`mu0_analytic` has the same nonnegative squared-intervention objective as frozen
mu=0. When the accepted backup q and initial nominal z form a legal pair, that
objective is exactly zero, a global minimum. It returns the identical backup q
and retains z without a cone solve, including when correction demand is unmet.
It preserves demand history and nominal-state updates. Otherwise its geometric
recovery is exactly the frozen mu=0 path. Shared-backup deterministic decision
tests and a source-equivalence guard cover the change; independent stochastic
TRAC trajectory runs are not asserted identical. Old mu=0 timings stay archived
and are not a speedup denominator.

Elastic's two-step extrapolation, W=20, development-frozen demand floor/startup,
SO(3) derivative, correction map, backup solver, real-objective improvement test,
two linearizations and budgets are unchanged. Original ordinary predictive IK
retains its own frozen three-update configuration. Neither nominal states nor
future references are executable seeds. Current and nominal commands use real FK
and the existing verifier. Empirical demand coverage is not a probability or
nonlinear feasibility guarantee; rank deficiency grants no positive 6D reserve.

## Inputs fixed without solver outcomes

Generate exactly 160 paths per robot, 40 per existing family, 300 frames, dt=20 ms.
Reuse `correction_reserve.data.reference_path` byte-for-byte: smooth, near-singular,
joint-limit return, and high-curvature/curvature-speed variation. The latter retains
four fixed abruptness levels, now ten paths per level. Seeds are
`983009000 + robot_index*10000 + family_index*100 + within_family_index`.
No outcome filtering, substitution, resampling or sample-size extension is allowed.

Reference paths are finite, range/rate valid, and checked frame-by-frame against
the unchanged 1 mm / 0.5 degree contract and original velocity tolerance. These
witnesses establish reference-state feasibility only. Save them separately from
online inputs (initial q and target poses only). Verify seed, UID, known canonical
query-hash and available target-sequence identities against prior metadata,
including development and observed formal inputs. A collision or invalid path
stops preparation; it does not authorize selecting a favorable replacement.

Preparation writes input identities, witnesses, dependency hashes, baseline Git
inventory, tests and a selection seal. Code, configuration, this decision and all
identities are committed and pushed before comparative solver calls. Preparation
uses only geometry and verification, not experimental solver outcomes.

## Execution and time

Each method starts from the shared q_ref(0), sees only current and historical
targets, updates its own previous_q only after acceptance, and holds it on failure
while the target index advances. Every one of 300 frames is retained, including
failures and deadline misses. One fixed stationary warmup per method is excluded
and disclosed; each measured trajectory resets online state. Jobs are complete
trajectories in a fixed randomized order, interleaving all methods and repeats.
Panda uses CPUs 0,2 and UR5e 4,6, with BLAS threads fixed to one and frozen installed
libraries. No simultaneous methods contend within a robot's CPU allocation.

Command-ready outer time includes conversion, backup, demand estimation, nominal
initialization, all cone construction/solves, nonlinear acceptance and final
verification. Serialization and post-trajectory diagnostic joins are excluded.
TRAC randomness has no controlled native seed; repetition indices do not imply
common random numbers. Pink repeats measure timing variability, not new samples.

## Prespecified analysis

TSR and DTSR20 are co-primary, with no all-metric gate. First average the three
repeats within each trajectory UID; then use 4000 paired bootstrap samples,
resampling UIDs within each fixed family. Robots remain separate. Report paired
differences and cost ratios with descriptive, unadjusted 95% percentile intervals.
An interval including zero is not evidence of equivalence or non-degradation.
Frame quantiles are descriptive pooled full-frame quantities, not independent
replicates; all calls enter cost. Accepted-error distributions are conditional on
acceptance with rejection rates separately visible.

Compare the primary against all five settings, emphasizing the fair analytic zero
control and mature TRAC/Pink. Report every family, repeat completion/deadline count,
cumulative outer time, P50/P95/P99, errors, acceleration RMS, optimization/partial
adoption rates, intervention and actual shortfall changes, first failure inputs,
and UID completion gains/losses. Gains/losses compare UID-mean completion fractions;
stable all-three versus none are additionally distinguished. Do not pair stochastic
search repetition numbers as matched seeds. Future-error joins run only after the
entire trajectory, never feeding the runtime.

All data, failures and timeouts remain. The final report evaluates replication,
robot-specific direction, independent value over analytic zero and cost/deadline
tradeoffs against mature solvers. No result triggers tuning, more samples, a new
algorithm or a paper rewrite.
