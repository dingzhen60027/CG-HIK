# Frozen experimental protocol

## Research question

Does a calibrated estimate of learned-seed difficulty improve the verified success–computation trade-off of hybrid IK by assigning candidate count, DLS iterations, and fallback only where needed?

## Robots and models

- UR5e, six revolute joints, end frame `tool0`.
- Franka Panda, seven revolute joints, end frame `panda_link8`.
- Models are trained independently for each robot. Results must not be described as cross-robot model generalization.

The seed input is normalized previous joint state, desired position, and the first two columns of desired rotation. Each of five bootstrap MLPs predicts a bounded joint increment. Training combines joint Huber loss with differentiable FK position and orientation losses.

The risk feature vector is fixed as:

1. seed position residual;
2. seed orientation residual;
3. ensemble mean variance;
4. ensemble maximum variance;
5. minimum Jacobian singular value;
6. minimum normalized joint margin;
7. proposed joint-step norm;
8. Cartesian position step;
9. Cartesian orientation step.

Difficulty labels are generated from the best learned seed with a maximum of 50 adaptive-DLS iterations: `easy=0..8`, `medium=9..25`, `hard=26..50`, and `fail=no verified convergence`.

## Locked runtime policy

| Risk decision | Candidate policy | Per-candidate budget | Fallback |
|---|---:|---:|---|
| `P(easy) >= 0.80` and `P(fail) < 0.05` | best learned seed | 8 DLS | no |
| not easy and `P(fail) < 0.25` | top 3 learned seeds | 15 DLS | no |
| otherwise | top 5 learned seeds + previous state | 25 DLS | 3 KDTree seeds with bounded TRF |

All attempts stop at the first verified solution. If every attempt fails, the system rejects the query instead of returning the lowest-residual invalid configuration.

## Acceptance contract

- position error ≤ 1 mm;
- orientation error ≤ 0.5 degrees;
- finite joint vector inside URDF limits;
- per-joint displacement ≤ URDF velocity limit × 0.02 s + numerical tolerance.

Minimum singular value is reported and used for damping/risk prediction, but it is not itself an acceptance condition.

## Data splits per robot

| Split | Size | Use |
|---|---:|---|
| Seed train | 200,000 | five bootstrap regressors |
| Seed validation | 20,000 | loss-weight/model checks only |
| Risk train | 60,000 | difficulty classifier |
| Risk validation | 20,000 | MLP vs gradient-boosting selection |
| Calibration | 20,000 | isotonic probability calibration only |
| Risk test | 20,000 | locked calibration and discrimination metrics |
| ID query test | 20,000 | runtime evaluation |
| Five stress categories | 5,000 each | runtime evaluation |
| Cartesian trajectories | 25 × 4 types × 200 frames | continuity evaluation |

Risk splits mix local transitions with large joint-space changes and joint-limit targets so that solver difficulty is not dominated by trivial adjacent frames.

## Main comparisons

1. previous state + 50-iteration DLS;
2. five random starts × 25 DLS;
3. three KDTree starts × 25 DLS;
4. best learned seed × 25 DLS;
5. three learned seeds × 15 DLS;
6. bounded TRF from previous state;
7. confidence-gated method.

The ablation suite removes history, ensemble diversity, uncertainty features, calibration, fallback, and adaptive damping separately.

## Primary outcomes and claim gate

Primary outcomes are verified acceptance, function/Jacobian evaluations, and end-to-end P95 latency. Secondary outcomes include P99 latency, fallback rate, trajectory spikes, pose error, and unreachable rejection.

The intended paper claim is allowed only if both robots meet at least one matched comparison:

- at matched success, ≥15% lower mean computation and ≥10% lower P95 latency; or
- at matched mean computation, ≥3 percentage-point higher hard-query success.

The risk component additionally requires fail AUROC ≥0.75 and ECE ≤0.10 on the locked risk test. Effects must have the same direction on both robots. Failed criteria are reported; thresholds are not retuned on the test set.

Each complete pipeline is repeated with seeds 17, 29, and 43. Query-level differences use 10,000 paired bootstrap resamples. Multiple primary comparisons are to be Holm-corrected in the manuscript analysis.

Timing is measured at batch size one with PyTorch restricted to one CPU thread. Every method receives 1,000 warm-up calls, after which each locked query is executed five times and its median wall time is recorded.
