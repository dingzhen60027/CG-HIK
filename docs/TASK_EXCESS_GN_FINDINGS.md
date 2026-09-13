# Task-excess bounded GN: development findings

Baseline: `c2a136b2bc0531998354f91c22d795fbe9d9e8c1`, branch `codex/hierarchical-v5`.
This is an implemented current-frame algorithm experiment on **previously observed development inputs**, not an independent evaluation or a submission claim. The old GN, verifier, robot configuration, paper and evidence are unchanged.

## One-page Problem–Method–Evidence–Conclusion

**Problem.** A squared sum of normalized position and orientation errors can have a constrained stationary point outside one task tolerance, despite the same input admitting a command. A return code or a small sum does not establish the two separate pose conditions.

**Method.** The supplied prototype is now a runnable single-solver mode. It minimizes squared distance to the product of the two task balls, uses their radial/tangential curvature in a damped bounded GN step, and checks actual nonlinear merit and the original verifier. The fixed internal radius is `1−10⁻⁶`; physical acceptance is unchanged. No prediction, extra seed, posture penalty, fallback or new QP backend was added.

**Evidence.** The package command passes the original verifier. Across 52 exact, deduplicated historical GN-family failure inputs, the new method accepts 21/41 Panda and 6/11 UR5e inputs in all three repeats, versus 4/41 and 1/11 for frozen κ=0. This inventory includes clipping/OSQP GN-source states; the **direct frozen κ=0/κ=1 source subset** contains 13 Panda and 3 UR5e inputs, of which the new method recovers 5 and 1, versus zero for the frozen methods. These are selected mechanism inputs, not an occurrence-rate estimate.

On all original 40 development trajectories per robot, the new solver completes **40/40 Panda and 39/40 UR5e**, identically across three runs. Frozen κ=0 completes 39/40 and 40/40; κ=1 completes 40/40 on both. Same-Φ L-BFGS-B completes 35/40 and 28/40. Thus there is an actual local-solver increment and a Panda gain over κ=0, **but also an UR5e loss and no completion advantage over frozen κ=1**. New cumulative computation is 2.67×/3.24× κ=0. About 99.3% of its accepted commands exceed 90% of at least one pose tolerance, and UR5e acceleration RMS is 64.7% higher than κ=0.

**Conclusion.** The task-objective mismatch is supported by legal commands and same-input comparisons, rather than only by a toy example. The analytic metric also adds local value beyond the selected isotropic and generic references. However, this implementation is **not yet a demonstrated replacement for the frozen bounded GN**: more same-input recoveries do not guarantee better closed-loop tracking. The evidence supports a specific candidate numerical mechanism, not general solver superiority or established novelty. This work package stops here; no parameters, fallback or follow-up formal experiment were added.

## 1. Immediate original-environment verification

The four supplied vectors were checked **without running IK**, against their exact original target, actual previous accepted configuration and `dt=0.02`. The source CSV was matched by robot, UID, frame and method, and by exact target/previous arrays. The selected input is Panda `trajectory_094`, frame 53, UID `c2ebae0d24e306f3744e9ac2a38add8502705448e7eda234f19e3082f56fa52c`.

| Supplied vector | Position error (mm) | Orientation error (°) | Original verifier |
|---|---:|---:|---|
| Frozen GN κ=0 | 0.260779286 | 0.545854361 | Reject: orientation |
| Frozen GN κ=1 | 0.260604188 | 0.545875289 | Reject: orientation |
| Prototype loop, point loss | 0.260779277 | 0.545854362 | Reject: orientation |
| Task-excess prototype | **0.999999679** | **0.499999634** | **Accept** |

All four are finite and satisfy the original joint and single-frame rate checks. The accepted vector's maximum rate utilization is `0.9999999999990011`. This is roundoff-close to the rate bound and the pose tolerance; it is not a robustness margin.

The repository implementation then independently regenerated a legal command from the real previous state, without the supplied successful vector or failed endpoint as seed. In measured repeat 0 it took 1.522 ms and 8 residual/Jacobian evaluations; the saved results contain all three calls. The frozen κ=0 endpoint has normalized projected-gradient infinity norm `2.19×10⁻⁷`, supporting a numerical near-stationary interpretation for this exact example. L-BFGS-B and the isotropic reference did not recover this input within their fixed settings. Their failures do not prove infeasibility.

Sources: [package verification](../outputs/single_solver_evidence/task_excess_development/package_verification/original_verifier.json), [same-input records](../outputs/single_solver_evidence/task_excess_development/local/same_input_results.json), [supplied task package](../outputs/single_solver_evidence/task_excess_development/task_package/task_excess_gn/README.md).

## 2. Numerical definition and checks

For normalized residual `e=[e_p;e_R]`, let `C_a=B₃(a)×B₃(a)`, with `a=1−10⁻⁶` in the implementation. Use `Φ=½ dist²(e,C_a)`. Outside a block ball, `v=(r−a)u` and `W=(1−a/r)I+(a/r)uuᵀ`; inside, both vanish. With `G=J_e S`, the update is the frozen box-QP with `H=GᵀWG+λI`, `g=Gᵀv`. In particular, the Hessian model uses **W, not WᵀW**. At a boundary the inside generalized derivative is selected.

The entire frame uses the same interval formed from the actual previous accepted state and `S_i=velocity_i·dt+velocity_tolerance`, intersected with the URDF range using the existing representable inward boundaries. Iteration does not recenter or enlarge this interval. The original NumPy active-set function is imported without modification. There are at most 30 outer checks and 8 backtracking trials per update, starting with λ=0.01, floor 0.0001, halving on an adopted update and multiplying by ten on rejection. True nonlinear Armijo decrease is required unless the trial is already accepted by the original verifier. The final verifier is always called, including on timeouts.

The following are standard mathematical properties, not proposed new theorems:

1. With radius **1**, `Φ=0` exactly matches the two public pose conditions; finite values, joint limits and velocity still require their separate checks. With the implemented inward radius, the zero set is slightly smaller. An admissible command may therefore return with a tiny positive internal Φ.
2. An attained block has zero gradient. An unattained block has radial eigenvalue 1 and tangential eigenvalues `1−a/r`. Thus W is positive semidefinite and λ>0 makes H positive definite.
3. For an **exact** QP minimizer d with zero feasible, first-order optimality against zero gives `(Hd+g)ᵀ(−d)≥0`, hence `gᵀd≤−dᵀHd`. A nonzero exact step is a descent direction. This is a local statement; it is not global convergence, existence of a feasible command or closed-loop stability.
4. The implementation checks direction finiteness, box violation and true nonlinear decrease. Detailed diagnostics retain `gᵀd`, `dᵀHd`, actual step and actual new Φ rather than treating a QP status as a proof.

Tests cover the zero set, inactive blocks, residual-space gradient/Hessian finite differences, positive definiteness and constrained descent, the scalar counterexample, and the actual Panda/UR5e SO(3) Jacobians. The scalar example remains a counterexample to a **point-loss optimum**, not a proof that every point-loss query fails.

### Same-Φ standard reference and timing

Standalone SciPy 1.17.0 L-BFGS-B uses the same Φ, analytic gradient and dynamic box, in joint variables normalized by S. It starts at the same previous state. The original verifier is checked on every evaluated configuration, and an admissible point terminates immediately. Settings were fixed before the comparisons: maxiter 30, maxfun 240, maxls 20, gtol `10⁻⁸`, ftol 0, and a 20 ms soft outer deadline. Disabling relative-function termination avoids accepting numerical stagnation of a squared tiny positive excess as task success; it does **not** require convergence after a legal command has been found. No alternative settings were scanned. These controls are consistent with [SciPy's documented projected-gradient and relative-function stopping semantics](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-lbfgsb.html).

Every method's complete outer conversion, bounds, numerical work and verification is timed. Three common nonstationary warmups and the original runner's stationary startup are outside timing. CPU affinity is logical core 4; BLAS/OpenMP each use one thread. Existing NumPy, SciPy, Pinocchio, TRAC-IK and Pink versions were retained, without enabling Numba. The numerical Jacobian backend is **Pinocchio**, while independent final verification uses the repository **URDFKinematics** implementation. The 20 ms checks are soft: late calls remain recorded.

Detailed traces were captured in separate diagnostic calls, not primary trajectory timing. L-BFGS-B's early-abort `nit` is unavailable (the raw zero is a sentinel), so summary iteration averages are intentionally omitted for it; actual function/gradient evaluations and full latency are retained. An initial report pass encountered TRAC's nullable iteration field; only the report aggregation was corrected and the saved commands reverified. No solver run was repeated for this reporting fix.

## 3. All historical first-failure inputs

The inventory contains 108 source aliases, deduplicated by **exact** robot, target position/rotation, previous_q and dt into 52 inputs: 41 Panda and 11 UR5e. The aliases include κ=0, κ=1 and the old clipping/OSQP GN controls. Tiny but real differences in previous states are retained; related inputs and three repeats are not treated as independent trajectories. There are 13 Panda and 3 UR5e source trajectory UIDs in the full inventory. No current-run failures were added to the sample after observing new outcomes.

| Method | Panda stable recovery / 41 | UR5e stable recovery / 11 | Direct κ0/κ1-source Panda / 13 | Direct κ0/κ1-source UR5e / 3 |
|---|---:|---:|---:|---:|
| Frozen κ=0 | 4 | 1 | 0 | 0 |
| Frozen κ=1 | 4 | 1 | 0 | 0 |
| Same loop, point loss | 4 | 1 | 0 | 0 |
| **Task-excess GN** | **21** | **6** | **5** | **1** |
| Same-Φ L-BFGS-B | 14 | 6 | 2 | 1 |
| Same-gradient isotropic metric | 16 | 6 | 3 | 1 |

“Stable” means all three repeated calls accepted; here their acceptance decisions agree. Compared with rerun κ=0, the new method gains 17 Panda and 5 UR5e inputs and loses none. Of these gains, 12 Panda and 5 UR5e endpoints also satisfy the diagnostic raw normalized-coordinate projected-gradient threshold `10⁻⁵`. Their separately saved legal commands establish **numerically near-stationary point-loss endpoints with missed admissible commands**. Other recovered cases are not relabeled as proven stationary. The threshold is a reporting diagnostic, not a changed solver setting or a mathematical proof of a local minimum.

The full W recovers five more Panda inputs than the isotropic metric, with no reversed cases in this sample; on UR5e their recovered sets coincide. Same-input first-step directions, damping, bounds and nonlinear Φ changes are saved separately. The local comparison therefore supports some value for radial/tangential curvature on Panda, not its universal necessity. Compared with L-BFGS-B, the full W gains seven Panda inputs and has the same UR5e accepted set. This does not establish superiority over all optimizers of Φ.

Sources: [source aliases](../outputs/single_solver_evidence/task_excess_development/protocol/failure_source_aliases.csv), [source-separated recovery table](../outputs/single_solver_evidence/task_excess_development/reports/local_source_groups.csv), [stationarity and witnesses](../outputs/single_solver_evidence/task_excess_development/reports/stationarity_and_witnesses.csv), [same-input metric directions](../outputs/single_solver_evidence/task_excess_development/reports/same_input_first_metric_update.csv), [all actual legal commands](../outputs/single_solver_evidence/task_excess_development/local/legal_witnesses.json).

## 4. Full closed-loop development comparison

Each robot has the original 40 trajectories, four families of ten, 150 frames each. All six methods run three complete sweeps from frame 0. A method only sees its actual previous accepted state and the current target; accepted commands update feedback, failures hold state and the target index advances. No reference reset or future target is used. All **216,000 calls** enter cost statistics. Replaying the stored commands through the original verifier found **212,347 accepted commands and zero accepted contract violations**. All successes, failures and late returns remain in the raw records.

Numbers below are new server measurements, not historical timings. Completion and deadline columns give the three counts out of 40; cumulative time is one whole 40-trajectory sweep, averaged over repeats. Frame percentiles include all frames, including failures.

| Robot | Method | TSR counts | DTSR20 counts | P50 / P95 / P99 (ms) | Cumulative (s/sweep) |
|---|---|---|---|---|---:|
| Panda | Frozen κ=0 | 39 / 39 / 39 | 39 / 39 / 39 | 0.266 / 0.285 / 0.450 | 1.706 |
| Panda | Frozen κ=1 | 40 / 40 / 40 | 40 / 40 / 40 | 0.267 / 0.284 / 0.378 | 1.607 |
| Panda | **Task-excess GN** | **40 / 40 / 40** | **40 / 40 / 40** | **0.765 / 1.104 / 1.348** | **4.555** |
| Panda | Same-Φ L-BFGS-B | 35 / 35 / 35 | 34 / 35 / 35 | 2.455 / 4.068 / 5.972 | 15.619 |
| Panda | TRAC task 5 ms | 36 / 37 / 37 | 35 / 37 / 37 | 0.170 / 0.269 / 5.214 | 1.977 |
| Panda | Pink | 37 / 37 / 37 | 37 / 37 / 37 | 1.000 / 1.062 / 1.121 | 5.517 |
| UR5e | Frozen κ=0 | 40 / 40 / 40 | 40 / 40 / 40 | 0.255 / 0.270 / 0.348 | 1.532 |
| UR5e | Frozen κ=1 | 40 / 40 / 40 | 40 / 40 / 39 | 0.255 / 0.270 / 0.361 | 1.551 |
| UR5e | **Task-excess GN** | **39 / 39 / 39** | **39 / 39 / 39** | **0.713 / 1.045 / 2.273** | **4.964** |
| UR5e | Same-Φ L-BFGS-B | 28 / 28 / 28 | 28 / 28 / 28 | 2.276 / 4.350 / 6.690 | 14.803 |
| UR5e | TRAC task 5 ms | 37 / 38 / 38 | 37 / 38 / 38 | 0.157 / 0.242 / 5.199 | 1.738 |
| UR5e | Pink | 38 / 38 / 38 | 38 / 38 / 38 | 0.641 / 0.707 / 0.831 | 3.901 |

### Paired trajectory effects

The inferential unit is trajectory UID. Repeats are first averaged within UID; 4,000 paired resamples preserve each family count. Intervals are descriptive, unadjusted 95% intervals conditional on these observed repeats. Frame quantiles are descriptive, not independent frame observations. No equivalence or universal guarantee is inferred from a zero-width or zero-containing interval.

| Comparison (new / reference) | Panda completion difference (pp; 95% interval) | UR5e difference | Panda cumulative ratio (95% interval) | UR5e ratio |
|---|---|---|---|---|
| Frozen κ=0 | +2.5 [0, 7.5] | −2.5 [−7.5, 0] | 2.670 [2.334, 2.909] | 3.240 [2.738, 4.135] |
| Frozen κ=1 | 0 [0, 0] | −2.5 [−7.5, 0] | 2.834 [2.766, 2.906] | 3.201 [2.693, 4.103] |
| Same-Φ L-BFGS-B | +12.5 [5, 20] | +27.5 [12.5, 42.5] | 0.292 [0.279, 0.304] | 0.335 [0.279, 0.433] |

The new method costs less and completes more than this same-Φ generic reference. It also completes more than TRAC/Pink on these development trajectories, but costs 2.30×/2.86× TRAC and 0.83×/1.27× Pink, respectively. Relative to the already effective frozen GN, its median and tail cost are higher. The fixed-reference comparison does not support a general speedup claim.

### Families, gained/lost trajectories and error use

Panda new GN completes all ten trajectories in every family. Against κ=0, it gains near-singular `trajectory_14`, UID `4e626ba28da6a8d45d22f023c3732e2d796c83a0e74df2aa41168fe26f6dccab`; κ=1 already completes it. UR5e new GN completes ten smooth, ten near-singular, ten joint-limit-return and **nine high-curvature** trajectories. It loses high-curvature `trajectory_30`, UID `382307fdf25e31f8b1ae5c14a7545e81bf0c54489223c4626934c3dbe31a195e`, against both frozen GN configurations. Its first failure is frame 32 in all repeats, with both position and orientation rejected. No favorable previous state from another method was substituted. Different closed-loop previous states prevent this comparison from proving one unique causal explanation for the loss.

| Robot / method | Accepted position P95 (mm) | Accepted orientation P95 (°) | Either pose error >90% tolerance | Mean trajectory acceleration RMS (rad/s²) |
|---|---:|---:|---:|---:|
| Panda κ=0 | 0.426 | 0.0315 | 0.99% | 3.093 |
| Panda κ=1 | 0.433 | 0.0316 | 0.93% | 3.214 |
| Panda new | 1.000* | 0.500* | **99.27%** | **3.805** |
| UR5e κ=0 | 0.370 | 0.0283 | 0.67% | 5.744 |
| UR5e κ=1 | 0.369 | 0.0298 | 0.67% | 5.796 |
| UR5e new | 1.000* | 0.500* | **99.26%** | **9.457** |

*Rounded for display; actual P95 values are strictly below the public limits and are preserved in source-data. This is use of allowed error, **not improved accuracy**. RMS uses the actual accepted/held command sequence, including failures, not a smooth reference. Compared with κ=0, mean trajectory acceleration RMS rises 23.0% on Panda and 64.7% on UR5e; the UR5e paired ratio interval is [1.256, 2.222]. The new method has 0 Panda and 15 UR5e late frames; all 15 UR5e late calls are retained, including those in the failed trajectory. Full family tables, both error components and all six methods are in the linked source-data, not filtered from the report.

Sources: [main table](../outputs/single_solver_evidence/task_excess_development/reports/main_table.csv), [all family results](../outputs/single_solver_evidence/task_excess_development/reports/family_table.csv), [paired intervals](../outputs/single_solver_evidence/task_excess_development/reports/paired_comparisons.csv), [gained/lost UIDs](../outputs/single_solver_evidence/task_excess_development/reports/gained_lost_uids.csv), [saved successful trajectories](../outputs/single_solver_evidence/task_excess_development/reports/successful_trajectory_index.json), [original-verifier replay](../outputs/single_solver_evidence/task_excess_development/reports/verification.json).

## 5. Method boundary against prior work

The comparisons below use original methods sections, not only abstracts: RangedIK §§III–IV, Moe et al. §4 and §6, and Drusvyatskiy–Lewis §5. This is a focused boundary check, not a proof of literature-wide originality. The dedicated academic-search connector was unavailable; primary author/publisher full texts and author publication metadata were used instead. Accessed 2026-09-13.

| Work | Established content | Difference of this implementation and evidence boundary |
|---|---|---|
| Wang, Praveena, Rakita & Gleicher, **RangedIK**, ICRA 2023, DOI 10.1109/ICRA48891.2023.10161311 ([author full text](https://graphics.cs.wisc.edu/Papers/2023/WPRG23/2023_ICRA_RangedIK.pdf)) | Per-instant nonlinear optimization combines range/equality/preference tasks using weighted parametric losses, including Swamp/Swamp-Groove range terms. | Here the allowed pose set is exactly two normalized norm balls; its squared-distance derivative and radial/tangential metric form one dynamic-box update. Tolerance use and current-pose optimization are not new. No successful head-to-head RangedIK evaluation was added, so relative superiority is unestablished. |
| Moe, Antonelli, Teel, Pettersen & Schrimpf, **Set-Based Tasks within the Singularity-Robust Multiple Task-Priority Inverse Kinematics Framework**, Frontiers in Robotics and AI 3:16 (2016), DOI 10.3389/frobt.2016.00016 ([full text](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00016/full)) | Set-task activation, tangent-cone checks and null-space task priority let inactive attained tasks cease constraining motion. | This experiment uses one joint-space bounded update without a task hierarchy, with residual-space curvature for both pose blocks. Ceasing to chase zero error inside an allowed set is already established. No claim to inherit that paper's stability analysis is made. |
| Drusvyatskiy & Lewis, **Error bounds, quadratic growth, and linear convergence of proximal methods**, Mathematics of Operations Research 43 (2018), 919–948, DOI 10.1287/moor.2017.0889 ([author manuscript](https://people.orie.cornell.edu/aslewis/publications/18-error.pdf), [publication record](https://people.orie.cornell.edu/aslewis/publications/2018.html)) | The prox-linear framework minimizes a convex outer function composed with a smooth map plus a convex term, linearizing the map and adding quadratic regularization. | The current code instead uses the local quadratic residual-space metric in a box-QP; it is not an exact solve of that paper's full prox-linear subproblem. Projection, composite/GN modeling, damping and constrained descent are mature foundations, not new theorems or transferable global guarantees. |

The supported candidate increment is **this task-ball geometry combined with the existing bounded update and actual command verification**, with extra local recoveries relative to a same-loop point control and some local value beyond isotropic curvature. The full-trajectory evidence remains mixed; it does not yet establish the complete new solver as the best deployment choice or a universally necessary task metric.

## 6. Reproduction and stopping point

Entry: `scripts/run_task_excess_development.py`; fixed configuration: `configs/task_excess_development.yaml`. In the existing environment:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=tmp/crik_dependencies/python:src
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_excess_development.py verify
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_excess_development.py prepare
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_excess_development.py local
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_excess_development.py trajectories --robot panda
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_excess_development.py trajectories --robot ur5e
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_excess_development.py report
```

These commands are the recorded stage order, **not an instruction to rerun into the populated output tree**: every stage uses exclusive output creation and refuses overwriting. A deliberate reproduction requires an isolated copy with an empty result root, preserving the supplied task package. The archive, original inputs, numerical sources, job order and completed-record hashes are recorded. Tests include the frozen GN regression suite; the test count is not used as performance evidence.

Final answers: **yes**, the package vector is admissible; **yes**, same-input numerical near-stationary point-loss misses with legal witnesses recur; **yes, locally**, the full task metric adds recoveries over selected same-objective references; **yes, a trajectory loss remains**, on UR5e alongside higher error use and motion fluctuation; **no**, broad novelty, universal improvement and replacement of frozen GN have not been established. No paper, new model, additional layer, fresh trajectory set or follow-up parameter search was created.
