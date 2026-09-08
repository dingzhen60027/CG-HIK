# Official RangedIK baseline adapter

This is a baseline implementation record for Correction-Reserve IK development,
not a comparative performance result or a claim of a new RangedIK algorithm. No
old solver, verifier, configuration, result, model, or manuscript was modified.

## Source and build identity

The source is the authors' [RangedIK repository, `ranged-ik` branch](https://github.com/uwgraphics/relaxed_ik_core/tree/ranged-ik),
pinned to `1c48d2ae408b4e024ee037641aac1e728267984e`, under its MIT license.
The [authors' publication page](https://graphics.cs.wisc.edu/Papers/2023/WPRG23/)
identifies the ICRA 2023 paper and DOI `10.1109/ICRA48891.2023.10161311`.
The algorithm is the official weighted multi-objective RangedIK core, not a
Python reimplementation of a different IK method.

The pristine checkout is `tmp/crik_dependencies/ranged_ik/`. The positive-range
copy is `tmp/crik_dependencies/ranged_ik_positive_range/`; it is generated from
that exact revision, and the build checks every tracked source file. The only
change to this copy is the two-line patch saved at
`src/confik/correction_reserve/native/ranged_positive_range.patch`:

```diff
-        if (bound <= 1e-2) {
+        if (bound <= 0.0) {
```

It applies once to `MatchEEPosiDoF` and once to `MatchEERotaDoF`. All other 129
tracked upstream files are byte-identical. The patch SHA-256 is
`0ef16658394005bd3436ffd0dc089450cd5e56d9b6cb46e7b9010124be97f91e`.
The upstream/patched objective file hashes are, respectively,
`652511941a79787a1a2087b3fd90924ee712db22823e8d582c09a4e4b76d0dca` and
`c3f71b22fd3f081e870aa16db43ed54865620d906e0a69a38637886939b36dcf`.

Rust 1.85.1 was installed only under `tmp/crik_dependencies/rust_home` and
`tmp/crik_dependencies/cargo_home`; no existing Python environment, shell startup
file, or robot dependency directory was changed. The checked-in bridge
`Cargo.lock` fixes dependencies, including `optimization_engine=0.7.7` and
`nalgebra=0.30.1`. Builds use Rust release optimization level 3. The generated
manifest records native library, source, lockfile and bridge hashes:
`tmp/crik_dependencies/ranged_adapter_build/manifest.json`.

## Why the range adaptation is explicit

Upstream's two hard-coded `bound <= 1e-2` conditions are not configurable. At
the public contract of 1 mm position and 0.5 degrees orientation, conservative
component ranges are approximately 0.000577 m and 0.005038 rad. Both are below
the native cutoff. Merely passing these ranges to an unmodified checkout would
therefore exercise its ordinary groove goal, not its ranged-goal objective.

The adapted mode is named **RangedIK (positive-range adapter)**. It activates
the exact existing `swamp_groove_loss` whenever a bound is positive, retaining
the original loss constants and all non-collision objective weights. Zero ranges retain the
original ordinary goal. It does not enlarge task tolerance or native bounds.
The original behavior remains separately callable as `range_mode="upstream"`.
Neither mode may be reported as an unmodified full reproduction of every choice
in the RangedIK paper. In particular, a comparison must not infer superiority
over ranged-goal optimization from a baseline in which ranged mode is disabled.

The existing ranged loss uses normalized distance within its given range, but
its numerical behavior at sub-centimeter/sub-degree scales is not assumed to be
identical to its behavior at the authors' larger examples. This adapter change
is an implementation adaptation, not an algorithmic contribution of CR-IK.

## Native solve and task-contract interface

| Item | Fixed implementation |
|---|---|
| Numerical solve | Official objectives and gradients, PANOC 0.7.7 |
| Native iteration and stopping settings | At most 100 iterations; PANOC fixed-point residual tolerance 0.0005 |
| Native cache | Tolerance 1e-14; L-BFGS memory 10, as upstream |
| Primary native time cap | 18 ms, fixed before comparative outcomes; `time_limit_ms=None` exposes uncapped development behavior |
| Pose representation | Position in m; quaternion `xyzw` at FFI; native principal rotation-vector components in rad |
| Ranges | `nextafter(epsilon / sqrt(3), 0)` on each position/rotation component |
| Native residual frame | `R_target.T @ (p_actual - p_target)` and `Log(R_target.T @ R_actual)` |
| Hard joint bounds | Original URDF interval intersected with the actual previous command's single-frame rate interval, rounded inward by the existing routine |
| Native joint-limit penalty | Original static URDF bounds, not the narrowed dynamic interval |
| Final acceptance | Existing `SolutionVerifier`, unchanged |
| Feedback history | Actual accepted or held command for each prior frame, not raw rejected outputs |
| Self-collision objective | Disabled by default because collision is not in the common task; `collision_objective=True` restores the exact native objective tail |

The component box is a conservative subset of the task's norm ball, not an
equivalent representation. More importantly, RangedIK's Cartesian terms remain
**soft weighted objectives**, not hard pose constraints. PANOC convergence does
not imply that its output satisfies the task contract. Raw native status and
verifier outcome are recorded separately; a geometrically admissible native
iterate can be accepted even when the internal return reports an iteration/time
limit, whereas a converged but task-rejected iterate cannot be submitted.

The primary 18 ms native cap is not a hard 20 ms outer deadline guarantee.
PANOC checks elapsed time between steps and can overrun a cap within a step.
The command-ready outer duration includes target conversion, history
synchronization, dynamic interval construction, the native call, final
verification and the actual accepted/held history update. The timer stops only
after that update. Offline metric calculation and JSON/list record packaging
are outside that interval and separately reported in
`offline_packaging_latency_ns`. `history_update_latency_ns` is included in
`total_latency_ns`, and `accepted_within_20ms` uses that command-ready duration.
This scope aligns with the other online adapters rather than charging one
method for more verbose offline logging.

Retained official weights are unchanged: each position term 50; each rotation term 10;
each joint-limit term 0.1; velocity 0.7; acceleration 0.5; jerk 0.3;
manipulability 1. The official constructor then appends a separate tail of
self-collision terms, each with weight 0.01. These use distances between native
link segments with a hard-coded 0.05 m offset and a soft distance loss; they are
not part of the public pose/joint/rate command contract.

To avoid imposing an extra task on this baseline, the adapter's default
`collision_objective=False` truncates only that tail from the objective and
weight vectors at construction, without modifying the pinned core. It does not
merely set those weights to zero: their geometry and gradients are not evaluated.
The retained prefix is 17 terms for Panda and 16 for UR5e. Setting
`collision_objective=True` restores the complete 32- and 26-term native objective
sets, respectively. Actual active weight arrays and the explicit boolean are
stored in metadata and tested against the native constructor. This is a
pre-evaluation task-alignment choice, not a weight fit using smoke performance.

## Necessary FFI and state changes

The official Python/C wrapper discards PANOC's status, returns a leaked `Vec`
allocation, and updates the method history before any external acceptance
decision. Its `reset` also replaces all history with one configuration. A
small Rust bridge therefore exposes the official objective and PANOC solve
using caller-owned result buffers, actual native status, a hard per-call
Rectangle and explicit four-state command history. These are interface and
measurement changes; they are not a replacement objective or candidate search.

The bridge does not commit a result. Python verifies it and then records either
the accepted command or the held previous configuration. A failed target still
advances time, so the held configuration enters the velocity/acceleration/jerk
history for that frame. An externally changed `previous_q` triggers an explicit
history resynchronization recorded in `history_resynchronized`.

The original URDF text is passed to the official `Robot::from_urdf`, with the
same Panda `panda_link0`–`panda_link8` and UR5e `base_link`–`tool0` chain selection
as the common backend. No substitute robot settings, meshes, future target or
reference joint path is used. The actual common backend is
`confik.kinematics.urdf.URDFKinematics`; native FK uses the official Rust
`spacetime::Robot/Arm`, with `k` URDF parsing and nalgebra transforms. Three
predetermined nonspecial configurations per robot are checked at construction
to catch frame or joint-order mismatch. Other joint-axis conventions or
continuous-joint models are outside this two-robot adapter and rejected.

`native_objective_calls` and `native_gradient_calls` count actual PANOC callback
calls. They are **not** claimed to equal FK/FEV counts because the official
gradient evaluates several objective/FK operations internally. The generic
`solver_function_evaluations` field is therefore null rather than an invented
FEV estimate.

## Build, API and bounded checks

```bash
python src/confik/correction_reserve/native/ranged_build.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src CRIK_RANGED_NATIVE_SMOKE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_crik_ranged.py
```

The builder never changes the pinned checkout, and construction/solve never
downloads or compiles dependencies. To make a fresh build, install Rust or use
the already installed task-local toolchain. Both compiled modes and complete
build logs are placed under the new `ranged_adapter_build` directory.

```python
adapter = RangedAdapter(kin, verifier, source_config, urdf_path,
                       robot="panda", range_mode="positive_range",
                       time_limit_ms=18.0, collision_objective=False)
adapter.reset(initial_actual_q)
record = adapter.solve(position, rotation, previous_actual_q, dt=0.02)
if record["accepted"]:
    previous_actual_q = np.array(record["q"])
adapter.close()
```

The bounded smoke suite passed four tests after the contract-only adjustment.
One unit test injects artificial offline metric cost into a deterministic clock
and confirms it is excluded while actual history update time is included. The
native test also checks both collision configurations have exactly the same
non-collision objective weights and the expected native tail.

Each native smoke makes four calls per robot and
mode (16 calls total), not a trajectory evaluation: current pose, two tiny
current-target changes, then a deliberately unreachable target. All raw returns,
rejections, histories and timings are retained in
`tmp/crik_dependencies/ranged_adapter_build/smoke_contract_only.json`. The initial
with-collision smoke remains unchanged at `smoke.json` in the same directory;
it is not silently replaced with the task-aligned adapter record. Native/common FK
differences were below 6e-16 in both m and rad at the construction checks.

Actual native component evaluations confirm activation. At half of the nominal
mapped position bound, the original and positive-range loss values are about
-0.9999950000 and -0.9689475388; at half of the orientation bound, they are
-0.9996192786 and -0.9689474761. The tests compare these against the exact
respective native loss formulas, not against a homemade IK solver.

One unfavorable initial, with-collision smoke outcome is deliberately retained: the third small Panda
target in positive-range mode returned `PANOC_converged` but had a 4.14 mm
position residual and was rejected. The corresponding original-cutoff smoke
command was accepted. The adapter passed because it correctly rejected the raw
output and held the actual accepted history, not because all test targets were
solved. No loss constants, budgets or candidate choices were subsequently tuned
to remove this observation. The later collision-tail removal and timing endpoint
alignment are explicitly motivated by common-task fairness and are recorded
before complete comparative measurements. These few selected smoke inputs do not estimate
either method's trajectory success or online timing distribution.

No complete dataset was run by this adapter task; no old evidence was rewritten.
