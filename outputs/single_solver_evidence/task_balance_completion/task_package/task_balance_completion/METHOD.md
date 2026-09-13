# Same objective; a computable inexact model-decrease condition

This is a candidate completion of the existing two-block task minimax GN, not a new task, loss, robot model, fallback, or claim of established publication novelty.

Let normalized task residuals be e_p and e_R, let G be the correctly scaled residual Jacobian, and let d stay in the current-frame box B, fixed relative to the actual previous accepted joint state. The local objective remains

P(d) = max{||e_p+G_p d||², ||e_R+G_R d||²} + R(d),
R(d) = lambda/2 ||d||² + kappa/2 ||c+W d||².

The numerical task still returns a command only after the original nonlinear verifier accepts it. It need not minimize the nonlinear minimax objective before returning an admissible command.

## Weighted subproblem and bounds

For theta in [0,1], define h_theta(d)=theta f_p(d)+(1-theta) f_R(d)+R(d). Its Hessian is at least lambda I, with lambda>0. Any candidate in B gives U=P(d)>=P*. For a feasible weighted candidate v, strong convexity supplies a valid lower bound:

L_theta = h_theta(v) + min_{z in B} [grad h_theta(v)^T(z-v)+lambda/2 ||z-v||²].

The last minimization is separable. Set delta=clip(-grad/lambda, lo-v, hi-v), then evaluate the expression. Since h_theta(z)<=P(z), L_theta<=min h_theta<=P*. Maximum lower bounds collected across theta remain valid. This bound does not assume that an approximately active coordinate is exactly in the normal cone. Computed floating-point bounds are not interval-arithmetic certificates; material L>U inconsistencies must disable the claimed condition.

## Relative decrease stop

Let P0=P(0), D=P0-U>0, and E=U-L>=0. Instead of requiring a tiny absolute optimality gap at every nonlinear iterate, allow stopping the local solve when

E <= eta D,  0<eta<1.

Then, in exact arithmetic with valid U and L:

D* = P0-P* <= P0-L = D+E <= (1+eta) D,

hence D >= D*/(1+eta).

With the prototype eta=0.25, a stopping point has at least 80% of the best regularized local-model decrease. This is not 80% of the nonlinear improvement, success probability, tracking completion, or future feasibility. True nonlinear sufficient decrease and final acceptance are still checked. No unproved global or closed-loop convergence statement is added.

When the task is already accepted, the solver returns without claiming local optimality. At the cap or deadline, it reports an inexact or timed-out result rather than inventing a certificate. If no descent step exists, failure is numerical, not mathematical infeasibility.

## What is and is not changed

1. The theta=1/2 initial weighted QP remains algebraically the frozen GN step for the same kappa. A legal first proposal returns directly.
2. If that proposal is not yet legal, the scalar dual computes a sufficiently useful step, not necessarily a highly accurate local minimizer.
3. A cache retains each actual q's residual/Jacobian, so backtracking or final result assembly does not reevaluate it unnecessarily. The original independent verifier still determines final acceptance.
4. Tight and relative modes share exactly the same cached implementation, box QP, data, damping, and geometry. Only the inner stopping test differs. This separates algorithmic effect from software overhead.

The caching and delayed matrix construction are implementation improvements. Duality, inexact GN, minimax, and active-set QP are established methods. The proposed research contribution is a task-specific composition and numerical acceptance/progress rule; its novelty and independent utility remain to be evaluated against closest work.

## Closest reference boundaries

- Wang et al., RangedIK, ICRA 2023. https://doi.org/10.1109/ICRA48891.2023.10161311 — ranged tasks and preferred points already exist.
- Boyd & Vandenberghe, Convex Optimization, 2004. https://web.stanford.edu/~boyd/cvxbook/ — epigraph, duality, and gap bounds are prior foundations.
- Porcelli, On the convergence of an inexact Gauss–Newton trust-region method for nonlinear least-squares problems with simple bounds, Optimization Letters 7, 447–465 (2013). https://doi.org/10.1007/s11590-011-0430-z — controlled inexact subproblem solution is not new.
- Walwil & Fercoq, The smoothed duality gap as a stopping criterion, Mathematical Programming Computation 17, 653–697 (2025). https://doi.org/10.1007/s12532-025-00284-0 — using computable stopping measures to avoid excess iterations is established; that paper treats a different convex formulation and its guarantees cannot be transferred here.

These are prior-art boundaries, not a claim that all closest IK-specific methods have been exhaustively excluded.
