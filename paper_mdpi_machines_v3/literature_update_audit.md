# Literature update audit

Audit date: 23 August 2026

## Scope and source limitations

The supplied research note listed 45 papers across numerical IK, learned IK,
motion planning, risk calibration, and OOD detection. This manuscript does not
import the list wholesale. Only papers needed to repair a current positioning
gap are eligible for citation. Bibliographic metadata must be independently
verified, and preprints must remain labelled as preprints.

The academic-search MCP was unavailable in this session. The OpenAlex fallback
returned HTTP 429. The audit therefore used direct Crossref metadata for the
entries below. This is adequate for bibliographic identity, not a substitute
for full-text methodological reading.

## Newly verified primary records

| Role in manuscript | Verified record | DOI | Status |
|---|---|---|---|
| Modern analytic/geometric IK comparator | Elias, A. J. and Wen, J. T. “IK-Geo: Unified robot inverse kinematics using subproblem decomposition.” *Mechanism and Machine Theory* 209, 105971 (2025) | `10.1016/j.mechmachtheory.2025.105971` | Crossref title, authors, venue, year, volume, article number verified |
| Constraint-rich numerical IK boundary | Zhang, Y. and Kikuuwe, R. “Ensuring Viability: A QP-based Inverse Kinematics for Handling Joint Range, Velocity and Acceleration Limits, as Well as Whole-body Collision Avoidance.” *Journal of Intelligent & Robotic Systems* 112(1) (2026) | `10.1007/s10846-025-02335-z` | Crossref title, authors, venue, year, volume, issue verified |
| Runtime uncertainty and model escalation context | Xu, C. et al. “Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies.” *Robotics: Science and Systems XXI* (2025) | `10.15607/rss.2025.xxi.073` | Crossref title, authors, venue, year verified |

## Already verified and retained

The current bibliography already covers DLS, TRAC-IK, variable step sizes,
IKFlow, GGIK, CppFlow, CycleIK, MimicIK, RelaxedIK, IKSel, calibration, deep
ensembles, PyTorch, scikit-learn, SciPy, and Pinocchio. These sources support the
current paper's proposal, refinement, calibration, and implementation context.

## Deliberately not promoted into the current bibliography

- Papers used only to motivate the prospective dual-abstention v4 method.
- Entries whose metadata or full technical scope was not independently checked.
- External baselines that were not executed under the present command contract.
- Collision-aware and hardware-deployment studies when the current manuscript
  reports exact URDF kinematics only.

Those works may remain in the v4 planning bibliography, but they cannot be used
to imply that the current method already supports OOD deferral, acceleration or
jerk constraints, collision avoidance, or hardware validation.
