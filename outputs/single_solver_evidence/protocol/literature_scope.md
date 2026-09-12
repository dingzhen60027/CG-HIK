# Focused source check (not a new literature-review experiment)

The academic-search MCP tools are not mounted in this session. Primary papers,
author repositories and official documentation were read through web access,
and checked against the installed source where version-specific behavior matters.
Retrieved 2026-09-12. No citation-count or priority-of-invention claim is made.

| Source | Established capability relevant here | Boundary for this experiment |
|---|---|---|
| Flacco, De Luca and Khatib, *Control of Redundant Robots under Hard Joint Constraints: Saturation in the Null Space*, IEEE T-RO 31, 637–654 (2015), [author manuscript](https://iris.uniroma1.it/retrieve/e383532a-4954-15e8-e053-a505fe0a3de9/Flacco_Postprint_Control-of-Redundant_2015.pdf) | SNS reallocates motion among unsaturated joints, maintains hard motion bounds, incorporates task scaling and discusses QP optimality/variants. | Bound-aware redistribution is established. This study does not run SNS or establish superiority over it. Our fixed target timing and final pose contract are not SNS task scaling. |
| [Pink official implementation and differential-IK formulation](https://github.com/stephane-caron/pink), [QP and limits API](https://stephane-caron.github.io/pink/inverse-kinematics.html) | Weighted task errors, configuration and velocity constraints form a QP; numerical joint commands are obtained from differential IK. | The measured installed version is **pin-pink 3.3.0**, not the current website version. The frozen repository adapter performs one differential step; this is not a comparison against every possible Pink-based iterative controller. |
| Wingo, Sathya, Caron, Hutchinson and Carpentier, *Linear-time Differential Inverse Kinematics: an Augmented Lagrangian Perspective*, RSS 2024, [DOI 10.15607/RSS.2024.XX.110](https://www.roboticsproceedings.org/rss20/p110.html), [paper](https://simple-robotics.github.io/publications/loik-solver/static/paper/loik2024rss.pdf) | LoIK exploits kinematic-tree structure with augmented-Lagrangian/ADMM machinery for constrained differential IK and linear-complexity structured subproblems. | A dense 6/7-variable active-set benchmark provides no comparable complexity theorem or performance comparison with LoIK. No LoIK adapter is added. |
| [TRACLabs official repository](https://github.com/traclabs/trac_ik), also local pinned README and source at upstream `90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb` | A Newton-style branch and a bound-aware SQP branch run concurrently; Cartesian bounds and different solution objectives are established functionality. | TRAC is already constraint-aware, not the clipped-GN ablation. Tests use the previously verified native task-tolerance adapter, Speed mode, 5/20 ms. |
| [OSQP algorithm](https://osqp.org/docs/solver/index.html), [Python update and warm-start API](https://osqp.org/docs/interfaces/python.html), inspected installed OSQP 1.0.5 Python source | Convex QPs, cached numeric matrix updates, warm starts and polishing are existing solver capabilities. | We replace only the local QP, retain its mathematical objective/bounds, and assess raw and projected returned steps independently. Active-set and ADMM iteration counts are not commensurate. |

Constrained Gauss–Newton/SQP local models, damping, line searches, box active
sets and joint centering are mature ingredients. The current code is a particular
finite-budget combination, with a current-frame interval fixed around the actual
previous accepted state. Its line search decreases **task residual only**, not
the complete posture-regularized quadratic objective. It is neither a new SQP
theorem nor a globally convergent constrained optimizer.
