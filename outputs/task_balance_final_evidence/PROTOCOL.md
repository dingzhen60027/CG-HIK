# Frozen supplementary evidence protocol — before solver outcomes

Baseline: `79d3d9324647a51876dca2b2ec612d94599b6559`, existing branch.
No numerical method, verifier source, old output or paper is edited. Configuration,
new input identities and this protocol are committed before the measured calls.

## A. Public DROID common-input replay (Panda only)

Official `droid_100/1.0.0` has 100 episodes in 31 TFRecord shards. Only this
debug subset was downloaded for identity extraction, without decoding images.
Its original low-dimensional HDF5 objects were then located in `droid_raw/1.0.1`
(1.0.0 is an access fallback for the SAME episode, not a new data source).
Objects are size/MD5 checked and SHA256 recorded. No hardware-bearing DROID
module is imported, no controller/server is started.

Sort SHA256 of canonical raw episode path. Select first six field/time-usable
episodes for interface verification, next 24 from other lab/date recording-day
clusters for evaluation. Candidate order, access failures and every exclusion
are in `source_replay/selection.json`; selection uses no IK outcome or proximity
criterion. The known source dataset's success/failure labels do not filter it.
The 100-episode debug subset is not a probability sample of DROID.

Actual raw fields, verified against the official writer/collector:

| Meaning | Raw field / mapping |
|---|---|
| Requested target | `action/cartesian_position`, not observed FK and not the separate teleoperator target |
| Paired current state | `action/robot_state/joint_positions`, read inside `create_action_dict` |
| Source pose for FK checking | `action/robot_state/cartesian_position` |
| Optional witness only | `action/joint_position`; never solver seed |
| Earlier observation | `observation/robot_state/joint_positions`; retained to expose timing difference |
| Time diagnostics | `observation/timestamp/control/{step_start,control_start,step_end}` and skip flag |
| Pose convention | position in metres, extrinsic SciPy `xyz` Euler radians; base→panda_link8 |

Official source checked at `33ae6a67274f36d2e29525b86f23a56616ef43a7`;
Polymetis submodule `0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2` names
panda_link8 and ordered joints 0…6. Project frame mapping is identity, with no
fitting. All six interface episodes pass the predeclared 1e-5 m/rad maximum
software-FK discrepancy threshold (actual <4.20e-8 m / 8.04e-8 rad).
All remaining source states also pass a no-IK consistency check; none removed.

Input motion period is **1/15 s**, reconstructed from the official control and
IK configuration. Measured record intervals of 69–77 ms include host overhead;
they do not enlarge the motion allowance. No nonmonotonic time or >2-period gap
was found in the selected records. HDF5 `version_number=1.1` does not identify a
collection Git commit: that missing provenance is explicit, not silently claimed
to equal the current source commit. The software timing pattern and FK provide
consistency evidence, not historical checkout proof.

Public 1 mm / 0.5° acceptance is this study's contract, not a DROID task tolerance
or accuracy specification. Computation success within **20 ms** is distinct from
the 66.7 ms input motion period. Every raw record, including source skip/zero
commands and terminal record, is retained. Each query uses the same logged q for
all methods; commands do NOT alter subsequent queries. This is offline IK replay,
not hardware control, closed-loop TSR, or task success. No acceleration is inferred
from differences of offline commands at different logged states.

Five existing methods only: relative(.25), same-κ GN, fixed_qp2, same-outer
Clarabel, existing direct constrained SQP. Three calls/query, deterministic
seeded interleaving. Source command witnesses are verified independently; unknown
feasibility is not failure/infeasibility evidence. Report all queries and witness
subset, per-episode trigger/coverage/cost/errors, gains/losses and session clusters.

## B. Five fixed sensitivity instances, not model selection

Reuse all 270 already observed development queries/robot (30 anchors, nine cells).
Nominal η=.25/.10/.50 plus GN/SQP; position-half and orientation-half each .25
plus GN/SQP. Total **17,820 calls** (2×270×3×11), no historical baseline sweep.
Main remains .25 regardless of results. κ=1 and every other numerical setting
unchanged; same existing verifier class instantiated with stated tolerances.

Three η instances have byte-identical target and previous state. For each task
tolerance change, preserve witness and previous state and rescale target offset
from witness FK using the package formula (SO(3) world-frame left offset).
Each reconstructed witness is verified before any solver call. These are paired
reconstructions, NOT a fixed absolute target under different requirements.
SQP's existing 1e-7 normalized squared-constraint interior guard is unchanged.

## Execution, inference and stopping

Existing NumPy fallback path, original environment, CPU4; OMP/OpenBLAS/MKL=1.
Three full-call warmups/instance plus existing Clarabel conic-structure warmup.
Outer timer includes input conversion, numerical work and final verification.
Serialization, initialization/warmup and offline re-verification are outside.
Preserve all failures and late calls; soft budget is not hard real-time proof.

For B, average repeats inside query and bootstrap anchors within geometric class
(4000 paired resamples, descriptive unadjusted 95% intervals). Report all nine
cells. For A, first average repeats inside query, then aggregate complete episodes;
bootstrap lab/date session clusters preserving episodes, explicitly distinguish
equal-episode and pooled-query quantities. Frames/calls are not independent units.
No significance-based gate or sample expansion, no equivalence from zero intervals.

Only source-backed reports, scope and paper OUTLINE follow these two studies.
No paper body, algorithm revision, new data source or third experiment follows.
