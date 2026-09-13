# Task-excess-driven bounded GN — selected-input research prototype

## Status

This package proposes a narrowly scoped extension of the existing BoundedGN.
It is NOT a demonstrated new optimization principle or a validated robot policy.
It has been run on ONE previously observed Panda input, not a fresh dataset.
No original-server TRAC/Pink run, full trajectory run, or runtime comparison was
performed for this package. The original GitHub repository was not modified.

## Motivation

The command contract requires separate position and orientation norms to be
within their limits. The frozen solver minimizes the sum of their squared
normalized residuals. These objectives are not equivalent under joint/rate
constraints: minimizing one already-acceptable residual can offset a still
unacceptable residual in the other block.

A scalar exact example is e_p=x and e_R=1.4-0.5*x. The least-squares minimizer
x=0.56 has e_R=1.12>1, even though every x in [0.8,1] is task-admissible.
This is an algebraic illustration, not a claim about all IK problems.

## Candidate method

Let e=[e_p;e_R] be normalized by the unchanged physical tolerances and C be the
product of two 3D unit balls. Minimize Phi(q)=0.5*dist(e(q),C)^2 within the
same fixed current-frame joint interval.

For a block with r=||e_b||>1 and u=e_b/r:
  v_b=(r-1)u
  W_b=(1-1/r)I+(1/r)uu^T.
Inside a block's acceptable ball use v_b=0 and W_b=0.
The actual demonstration uses radius 1-1e-6 as a FIXED inward numerical margin;
the final verifier still uses radius 1. This is slightly stricter, not looser.
The mathematical zero-set equivalence Phi=0 iff pose acceptable applies to
radius 1; radius 1-1e-6 has a slightly smaller zero set.

Use one bounded generalized-GN step:
  H=G^T W G+lambda I, g=G^T v, G=J_residual*S;
  min 0.5*d^T H d+g^T d with the original current-frame displacement bounds.
Perform a true nonlinear Phi Armijo check and original command verification.
The prototype has no posture cost, no predictor, no fallback and no scenarios.

W is the residual-space generalized Hessian of the squared-distance loss, not
its square. It has eigenvalue 1 radially and 1-1/r tangentially outside a ball.
This is derived from existing projection mathematics. Originality must be
judged against set-based IK, ranged-goal optimization and generalized/proximal
Gauss-Newton, NOT inferred merely because the code is new.

## Actually computed selected-input results

Source commit: c2a136b2bc0531998354f91c22d795fbe9d9e8c1
Source path: outputs/single_solver_evidence/reports/first_failure_inputs.csv
Source row: Panda, single_gn_k0, trajectory_094, frame 53, repeat 2.
UID: c2ebae0d24e306f3744e9ac2a38add8502705448e7eda234f19e3082f56fa52c

All four solvers start from exactly the recorded previous_q with the same target.
The failure iterate is NOT used as the main solver's seed.

- Frozen BoundedGN kappa0: rejected, position 0.2607793 mm, orientation 0.5458544 deg.
- Frozen BoundedGN kappa1: rejected, position 0.2606042 mm, orientation 0.5458753 deg.
- Same-loop point loss: rejected, position 0.2607793 mm, orientation 0.5458544 deg.
- Task-excess generalized GN: locally accepted, position 0.9999996794 mm,
  orientation 0.4999996344 deg, 5 outer checks (4 accepted updates).

An independent homogeneous-transform FK and scipy Rotation logarithm validate
local pose/rate/limit acceptance. Original repository verifier acceptance is
still required. The local FK constants were taken from the user-provided
single_solver_completed_results.zip package. Its orientation scaling uses the
package's rounded 0.00872664626 rad, whereas independent acceptance uses exact
numpy.deg2rad(0.5). The fixed inward margin exceeds this rounding difference.
The task-excess command is close to the acceptance boundary: this is not evidence
of desirable precision or long-horizon behavior.

The frozen methods are given a generous OFFLINE time allowance so this comparison
isolates numerical behavior; no run time is reported as an online benchmark.
The same-loop control also checks that changing the loop alone does not explain
this input's outcome. This is still one selected input, not a success-rate study.

## Reproduce

    OPENBLAS_NUM_THREADS=1 python check_selected_input.py

The script writes selected_input_results.json and algebra_example.json. It also
checks the projected loss gradient and generalized metric away from the ball
boundary against finite differences. It does not fetch data from the internet.

Files:
- task_excess_gn.py: proposed small numerical kernel, callback-based geometry.
- bounded_gn_reference.py: frozen supplied reference solver, not a new algorithm.
- panda_model_local.py: local Panda model from the earlier supplied package.
- selected_input_source.json: exact selected input and source identifiers.
- selected_input_results.json: the actually computed comparison and commands.
- CODEX_TASK.md: bounded next implementation/evaluation task.

## Literature boundary

- Wang et al., RangedIK, ICRA 2023, DOI 10.1109/ICRA48891.2023.10161311:
  ranged/preferred goals and weighted multiobjective barrier losses already exist.
  Author page: https://graphics.cs.wisc.edu/Papers/2023/WPRG23/
- Moe et al., Set-Based Tasks within the Singularity-Robust Multiple Task-Priority
  Inverse Kinematics Framework, 2016, DOI 10.3389/frobt.2016.00016:
  set-task activation and keeping tasks inside acceptable regions already exist.
- Squared distance to a convex set, its projection gradient, generalized Newton
  metrics, box QPs and Armijo descent are established mathematical ingredients.

A viable new contribution would require a task-specific numerical mechanism
and demonstrated value beyond simply changing a loss function or calling a
general optimizer. That evidence is NOT supplied by this one-input package.
