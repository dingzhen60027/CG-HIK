# Isaac Lab external-validation protocol

This phase is deliberately separated from the frozen URDF-kinematic experiment.
The current ConFIK runtime does not become an Isaac Lab method merely because its
Conda environment contains Isaac Lab.

## When to run

Run this phase only if `outputs/paper_v2_aggregate/paper_gate_v2.json` reports
`paper_gate_pass: true`. If it is not run, manuscript claims must remain limited
to kinematic query sequences and verified command generation.

## Purpose

Test whether accepted joint commands remain trackable under a simulated
low-level controller. This phase does not retune the risk model, action gate,
solver budgets, verifier, or test paths.

## Locked setup

- Robot: Franka Panda Isaac Lab asset, matching the Panda joint ordering used in
  the kinematic run.
- Physics step: 0.005 s; one IK command every four physics steps (50 Hz).
- Controller: the same fixed joint-position PD controller for all IK methods.
- Methods: `fixed_robust_cascade`, `threshold_guard_cascade`, and `proposed_v2`.
- Paths: 20 independently seeded paths for each of smooth, orientation,
  near-singular, and joint-limit families; 200 command frames per path.
- Failure behavior: rejected command means hold the last accepted command; no
  reference state is injected after rollout begins.
- Order: randomized by complete path, with simulator reset to the same initial
  state before each paired method rollout.

## Outcomes

The whole path is the independent unit. Report:

1. path completion under the same pose tolerance;
2. end-effector RMS and P95 tracking error;
3. joint RMS and maximum tracking error;
4. maximum measured joint velocity and acceleration;
5. rejected-command/hold frequency and longest consecutive hold;
6. solver deadline misses at 20 ms, reported separately from controller error;
7. per-path paired differences with bootstrap confidence intervals.

The primary external-validation criterion is completion non-inferiority within
5 percentage points versus the fixed cascade, with no measured joint-velocity
violation. Latency superiority is not required because the kinematic pilot
already shows a small learned-gate overhead on ordinary feasible queries.

## Optional robustness blocks

After the nominal test is locked, repeat without retraining under one-factor
perturbations: ±10% link mass, ±20% joint damping, and one-command-period sensor
delay. These are sensitivity tests, not additional independent nominal samples.

## Claim boundary

Passing this phase supports simulated controller-trackability under the tested
conditions. It still does not establish collision safety, torque-limit safety,
contact-rich performance, or physical-robot reliability.
