# Final paper changelog

Date: 2026-09-04
Branch: codex/hierarchical-v5
Evidence baseline: 244a216caed4367bf90c5d1eba5a0e1a88997019

## Scope

This revision performs no training, solver execution, threshold selection, query
generation, or frozen-output modification. It converts the completed V4 point and
fresh transition-rich evidence into one publication manuscript.

## Narrative changes

- Replaced the development-version storyline with one method name: **CG-HIK**.
- Adopted the title “CG-HIK: Query-Adaptive Tail-Latency Routing for Kinematically
  Verified Online Inverse Kinematics.”
- Fixed the organizing sentence:
  “Learning allocates solver effort per query; numerical geometry generates joint
  commands; deterministic verification governs acceptance.”
- Rebuilt the Introduction as eight logical paragraphs.
- Reorganized Related Work into four contribution-based chains rather than a paper
  list.
- Separated the online query, command contract, entry-cost objective, and three RQs.
- Rewrote the method around the shared numerical portfolio, action-complete pathway
  supervision, shared success/action-specific latency heads, tail routing,
  reject/defer, deterministic verification, and exact batch-one deployment.
- Removed stale claims that fresh testing had not begun or that optimized latency was
  validation-only.
- Renamed all artifact-facing methods to reader-facing canonical names.

## Evidence changes

- Added the final one-shot transition-rich evaluation as RQ3.
- Preserved the fresh point-query evidence as RQ2, including the false point-study
  overall gate and weak OOD/defer results.
- Added the final transition completion, cumulative latency, FEV, P50/P95/P99,
  contract, and family results.
- Retained the Panda joint-limit-skim latency/FEV regression and low Panda
  high-curvature completion.
- Kept V5–V7 only as a brief development-only analysis of temporal shortcuts; no
  version sequence appears in the Introduction or main method.

## Reproducibility changes

- Rebuilt paper/generated/evidence_snapshot.json from frozen JSON/NPZ inputs.
- Rebuilt paper/generated/paper_numbers.tex and generated table rows.
- Replaced stale source-data tables with development, point, trajectory, family, and
  representative-trace CSVs.
- Retained superseded manuscript-derived CSVs under `paper/historical/source_data/`;
  no frozen experiment output was moved or modified.
- Rebuilt the figure script and five main figures in SVG, PDF, and PNG.
- Archived superseded manuscript figures under `paper/historical/figures/`; the active
  `paper/figures/` directory contains only the five main figure groups.
- Updated the paper README with the exact evidence/figure/PDF build sequence.
- Added FINAL_PAPER_CLAIM_MAP.md and synchronized PAPER_STORYLINE.md,
  CLAIM_EXPERIMENT_MAP.md, and RESEARCH.md.

## Literature changes

- Retained 40 highly relevant references; 24 are from 2024–2026.
- Cross-checked all 40 records against authoritative metadata: 38 are field-verified,
  while CppFlow's conflicting page range and one forthcoming issue assignment remain
  explicitly disclosed checks rather than silently guessed fields.
- Corrected bibliographic metadata for Wampler, CycleIK, HJCD-IK, and SciPy.
- Preserved explicit preprint/forthcoming status and the unresolved CppFlow page-range
  boundary rather than guessing metadata.

## Claim changes

The final manuscript claims:

- heterogeneous queries benefit from query-specific solver allocation;
- CG-HIK preserves fresh point success while reducing FEV and P95;
- the final transition-rich test matches or improves hard-entry completion and reduces
  cumulative latency, FEV, and P95 on both robots;
- all accepted commands satisfy the implemented kinematic contract.

It does not claim:

- every latency quantile improves;
- strong OOD detection or defer recovery;
- collision, dynamics, contact, torque, controller, or real-robot validation;
- hard real-time guarantees;
- temporal shortcuts are part of the final method.

## Submission-specific fields remaining

- replace anonymous author and affiliation fields;
- add final CRediT roles and funding statement;
- provide a citable archival DOI;
- apply the target journal's official LaTeX template without changing evidence.
