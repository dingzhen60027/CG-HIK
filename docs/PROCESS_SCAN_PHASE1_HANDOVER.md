# Process-scan Phase 1 handover

This is a development-baseline handover, not a new-method result or a paper draft.
The accepted historical source is `bf9ccc4e0314c0d289101d89ff4e40577de392f9`.
The final result totals and comparison conclusions are in
`PROCESS_SCAN_BASELINE_REPORT.md` and its generated CSV source tables.

## Reproduce on the recorded host

Use the isolated `.venv-process-scan` environment. It inherits the existing Python
3.12 numerical environment; only CasADi 3.7.2 and TOPPRA 0.6.3 were installed in
the new environment. The old IK environments were not upgraded. MuJoCo 3.10.0,
Coal 3.0.2, NumPy, SciPy and the frozen URDF paths are recorded in provenance.
This is not yet a self-contained cross-platform distribution of all robot assets.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-process-scan/bin/python scripts/run_process_scan.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-process-scan/bin/python scripts/run_process_scan.py run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-process-scan/bin/python scripts/run_process_scan.py report
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv-process-scan/bin/python scripts/run_process_scan.py render
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-process-scan/bin/python -m pytest tests/test_process_scan.py -q
OPENBLAS_NUM_THREADS=1 .venv-process-scan/bin/python scripts/verify_process_scan_delivery.py
```

`prepare` refuses to replace the existing seal. `run` checks its code/config and
input hashes, and skips only already completed condition records. It refuses an
unreviewed unfinished run directory; exceptions are preserved as implementation
errors, never silently replaced by a more favorable method result. The first two
commands above describe the original execution order, **not an instruction to
overwrite or rerun the frozen evaluation**. `report` and `render` are read-only
with respect to inputs and numerical/physical records.

Official dependency commits:

- BoundMPC `b286df18641145bb97b31f29387227e4089a1ba4`;
- UR description `18e6f603b3ebc2ec479fecb62d6be544b15755e9`;
- inspected, unused Menagerie `8161bba264d7fa7c99ca301e91e7fb44737676ad`.

BoundMPC's real ROS messages were built with system Python3.10 and ROS2 Humble.
The headless check uses `src/confik/process_scan/boundmpc_check.py`, the upstream
experiment1 mathematics, and bundled MUMPS. No real robot controller was started.
It does **not** implement or benchmark the requested Panda/UR5e scan adaptation.

## Files and record semantics

Root: `outputs/process_scan/phase1_baselines/`.

- `inputs/`: all24 fixed development scene identities, construction records,
  reference configurations, model XML, and pre-comparison numerical source seal.
  Three unsuccessful Panda cylinder reference constructions remain in this set;
  they were not replaced or reclassified as mathematical infeasibility.
- `runs/<slot>/<method>_r<repeat>/`: condition manifests, geometric initialization,
  independent dense validation, timed plan, optional optimizer checkpoints,
  actual torque-actuated states, raw/quality-filtered ray hits, and metrics.
- `shared_b1/`: the **repeat0, verified10 s checkpoint** used unchanged to start B2.
  Missing B1 paths are distinguished from a B2 optimization failure.
- `calibration/`: failed as well as successful interface/control checks. These
  are not mixed into the development performance table.
- `official_boundmpc/`: three genuine headless official updates, not scan results.
- `reports/`: all-condition index, all scene results, main/family/bridge/bottleneck
  tables, paired intervals, editable figures, video index and integrity summary.

`metrics.json` is a flat, nullable measured record. It does not silently populate
the draft schema's unavailable integer/time fields with zeros: unexecuted
physics, unavailable B3 timings and missing initialization have explicit status
and null measurements. `raw_record_index.json` lists missing artifacts and why
no path/execution file can exist for an unexecuted condition. The provided draft
schema is retained as the design reference; the actual CSV headers and file
indices are the implemented data dictionary.

HDF5 stores actual q/dq/qacc/torque/reference/TCP/contact samples at5 ms. Maxima,
rate violation durations and contact checks are accumulated at every1 ms
physics step. `scan_samples.npz` contains81 actual ray endpoints per sampled
profile, ROI/occlusion-aware validity, range, hit geometry and coverage mask.
No ideal FK path is substituted for these measurements. Videos replay these
saved states in separate rendering data; they do not rerun IK or overwrite the
physical execution stream.

The review videos are640×480 at10 fps, in real simulated time, with the full
200 Hz feedback/ray files retained. One completed30 fps preview and a rendering
interrupted solely to lower preview cost are preserved in
`reports/videos/initial_30fps_render/`; neither is a new physical trial or a
replacement measurement. `reports/videos/INDEX.md` indexes the completed
10 fps delivery. `verify_process_scan_delivery.py` runs the interface tests,
checks video duration and source hashes, audits figure text, and produces a
delivery manifest. It runs no planning comparison or physical dynamics.

`actual_clearance_sampled_5ms_min_m` evaluates the saved actual configurations
with Coal and the same conservative CAD envelope used for planning; it is not
an exact height-field distance or a continuous-time minimum. Actual MuJoCo
contacts against the height field and robot geometry are counted at every1 ms
step. These two measurements have different semantics and remain separate.

The shared robot/model construction and three zero-residual geometry-adapter
calls precede each scene's measured condition loop. The supplied pose is the
midpoint's own FK, so these calls take the immediate-accept exit: they warm the
geometry path, **not** every local-QP or NLP backend. They are common fixtures,
not included in per-condition planning time. Method-specific initialization,
remaining lazy backend startup, graph construction,
optimization, retiming, dense checks and incumbent bookkeeping are included.
The B2 total also charges the recorded cost of acquiring its common B1 path.

## Calibration changes and their scope

The following occurred before the original comparison seal; previous attempts
remain available:

1. Corrected URDF RPY conversion to extrinsicXYZ; explicitly checked oldFK,
   generatedMJCF and the opticalTCP transform.
2. Replaced unreliable observed `mj_geomDistance` zero/height-field values in
   planning with the already installed Coal backend and a conservative CAD
   envelope. Physical collision and actual rays continue to use MuJoCo.
3. Distributed fixed64 spline capacity over prescribed turns. Both robots use
   the same deterministic knot rule;64/128 checks are calibration, not a new
   performance sweep or a selectable method setting.
4. Used computed-torque/inertia-scaled PD, rather than unstable torque-space
   wrist gains; all methods share400 s⁻²/40 s⁻¹ and the same torque limits.
5. Corrected the MuJoCo state timestamp: refresh derived poses after integration
   before reading the5 ms scanner. Alternating stale4/6 ms pose ages were an
   interface error, not a scanner performance result.
6. The time-parameterization grid includes spline knots and task joins;
   derivative verification separately uses6401 uniformly spaced samples.
   The common retimer records rawTOPPRA time and uniformly dilates by
   `max(1, velocity_ratio, sqrt(acceleration_ratio)) / .98`. The2% reserve was
   fixed from common reference calibration after1 ms extrema exposed a small
   violation missed by5 ms subsampling. Public limits were not widened.
7. Stored actual cubic-basis structural zeros sparsely for automatic
   differentiation. This fixes graph-construction cost; it does not choose a
   subset of optimization variables. Late feasible paths cannot be backdated.

These are interface/numerical correctness corrections, not algorithmic research
contributions. The frozen control reserve does not guarantee every later path's
physical acceleration; subsequent violations count as failures and remain in
the tables.

## Numerical-grid correction and the active seal

An initial partial comparison exposed roundoff-duplicate time-grid nodes in the
v-direction: seven pairs separated by approximately 2.8e−17 in normalized path
progress. These are identical intended knots, not distinct task targets. They
produced near-zero denominators in the B2 time-acceleration discretization.
The corrected union merges representations within 1e−12. The u-direction grid
is bit-for-bit unchanged; the fixed v-path timing equivalence check changed
duration by approximately 0.3 microseconds. No targets, process constraints,
objective, iteration/time budget, controller or acceptance threshold changed.

The original source, partial run records and shared B1 files are preserved under
`numerical_interface_revision/pre_grid_merge/`. They are **not** pooled into the
reported batch. The original `inputs/seal.json` remains unchanged. The complete
batch uses `inputs/numerical_seal_grid_repair.json`, which retains every input
identity/hash and records the source-only revision and original seal hash.
The grid equivalence and fixed-input B2 interface checks accompany that record.
This is a numerical interface correction, not a new optimization method.

Three unavailable reference records have a separate metadata erratum: their
early-return q-prefix was originally paired with a linspace stretched to 1.
No baseline used those records. The original files and seal are preserved;
`provenance/partial_reference_indices/` supplies the correct original index/400
coordinates, and `partial_reference_metadata_errata.json` identifies all three.

## Scope left open, not automatically continued

B3 still lacks a verified robot/state and coupled scanner-domain adaptation.
Its requested online delay accounting therefore has no scan measurements.
This is the specific unfinished system comparison, not evidence against the
published BoundMPC method. B2 is a local, finite-budget full-variable reference,
not an oracle; its residuals, iteration counts and no-improvement cases matter
when assessing how strong that comparison actually is.

Formal test scene rules remain design-only. No proposed block optimizer,
additional scene, new IK objective, paper text, perturbation study or hardware
execution is authorized by completion of this handover. Stop for user review.
