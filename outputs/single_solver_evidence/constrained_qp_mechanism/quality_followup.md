# One quality-only cap correction (before follow-up calls)

The original cached OSQP settings reached max_iter=20000 on 19 Panda and 2 UR5e
targeted inputs and failed the common KKT checks in all five passes. The initial
benchmark remains intact. No trajectory setting, target, or main solver changes.

Increase only this offline reference's max_iter to 200000; keep eps_abs=1e-8,
eps_rel=0, polishing, adaptive-rho and the original cached structure. This is a
quality-driven resource ceiling, not selection for speed or trajectory outcomes.
Repeat the same local-bank timing for all four methods in the same interleaving,
writing benchmark_quality_cap/ separately. Do not count any remaining faster
quality failures as wins and do not continue cap searches. Report original-cap
failures, final quality, and the changed cap explicitly. Both timing rounds stay
available; primary tables use the common final timing round, not the faster of
two results for each method.
