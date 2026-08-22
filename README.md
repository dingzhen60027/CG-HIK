# ConFIK

ConFIK is a reproducible implementation of confidence-gated, verified hybrid
inverse kinematics. Learning predicts a cascade entry action and rejection risk;
adaptive DLS/TRF generates or refines solutions; a shared verifier alone accepts
commands.

## Current status

The legacy v1 run is retained only as developmental evidence. The corrected v2
protocol is frozen, and the independent UR5e pilot passed:

```text
outputs/pilot_v2_4/ur5e/results/claim_gate_v2.json
```

The two-robot, three-seed paper experiment has **not** been started. Its launcher
refuses to run if the pilot gate is absent or false.

## Run the protocol

Use the existing environment:

```bash
export CONFIK_PYTHON=/home/eric/anaconda3/envs/isaaclab_3/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Run the bounded pilot:

```bash
./scripts/run_pilot_v2.sh
```

Run tests without loading unrelated ROS pytest plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$CONFIK_PYTHON" -m pytest -q
```

Only after inspecting the pilot artifacts, run the complete experiment:

```bash
./scripts/run_paper_v2.sh
```

That script runs UR5e and Panda with seeds 17, 29, and 43, then creates the
cross-run gate at:

```text
outputs/paper_v2_aggregate/paper_gate_v2.json
```

The design is specified in:

- `docs/experimental_protocol_v2.md` — frozen estimands, splits, thresholds, and stop rules;
- `docs/experiment_matrix_v2.md` — each paper claim mapped to a comparison and ablation;
- `docs/isaaclab_external_validation.md` — optional post-gate controller-tracking protocol;
- `docs/artifact_schema.md` — artifact and query-log semantics.

## What one full run evaluates

There are six primary methods:

1. previous-state DLS;
2. learned-seed DLS;
3. fixed robust cascade;
4. validation-constrained Cartesian threshold guard;
5. previous-state TRF;
6. proposed learned action gate.

The other seven methods are diagnostic ablations: no history, single ensemble
member, no uncertainty features, no calibration, no reject, no fallback, and
fixed damping. Thus the full output contains 13 method names, but only six belong
to the main comparison table.

The runtime benchmark separates:

- independently generated feasible point queries for budget routing;
- unreachable/discontinuous points for rejection;
- whole closed-loop paths for sequential completion.

It does not pool post-failure path frames into point-query success. Point-query
inference uses one unique generating unit per query; trajectories are analyzed as
whole-path clusters.

## Reproducibility safeguards

- `risk_train`, model-validation, calibration, policy-validation, classifier-test,
  and runtime-test data have disjoint roles.
- The learned and threshold policies are tuned only on policy-validation under
  the same false-reject and reject-recall constraints.
- Every output directory stores hashes of its resolved config and `src/confik`
  source tree. Changed code/config cannot reuse locked artifacts under the same
  experiment name.
- Method order is randomized per query; CUDA is synchronized; batch-one timing
  records CPU/GPU/software provenance.
- Query effects use paired cluster bootstrap intervals. Training seeds are
  sensitivity replicates, not extra independent test datasets.
- No solver result is accepted without pose, finite-value, joint-limit, and
  per-frame velocity checks.

## Scope

The current paper pipeline uses exact URDF kinematics and local Isaac Sim robot
descriptions. It runs inside an environment that also contains Isaac Lab, but it
is **not an Isaac Lab physics experiment**. It does not establish collision
safety, torque/dynamic feasibility, low-level controller tracking, or hardware
real-time performance. Those require a separately reported Isaac Lab or physical
robot validation layer.

The old `configs/paper.yaml`, `scripts/run_paper.sh`, and unversioned result files
belong to v1 and should not be used for the paper claim.
