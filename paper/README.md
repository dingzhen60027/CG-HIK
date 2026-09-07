# Task-contract-aware online IK manuscript

**Task-Contract-Aware Online Inverse Kinematics: Aligning Solver Convergence with
Command Admissibility**

Main source: `main.tex`. Compiled manuscript: `main.pdf`.
The paper distinguishes native convergence, public command admissibility and
deadline-qualified admissibility. CG-HIK is a secondary historical allocation case
study, not the principal algorithm.

## Build without running experiments

From the repository root, with Python and LaTeX available:

```sh
python paper/scripts/build_evidence.py
python paper/scripts/make_figures.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
python paper/scripts/check_paper.py
```

The first script uses only the Python standard library and validates the frozen
stage-one delivery before generating numbers, tables and source-data CSVs.
The second materializes the exact reviewed frozen PDF/SVG/PNG figures; the editable
rendering source is `src/confik/task_contract_alignment/reporting.py`.
The final check uses standard-library checks plus Poppler's `pdfinfo`.

These commands do not import IK solvers, train models, construct new inputs,
resample experimental statistics or write to `outputs/`.
Do not rerun the experiment launcher to build the paper.

## Authoritative evidence

Measurement commit: `089a3fe7ec0456002aa0f3b9661119b99072fad7`.

- `outputs/task_contract_alignment/`: protocol, raw new point/sensitivity/UR5e
  records, aggregate reports, witnesses, vector figures and delivery hashes.
- `outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/`:
  historical authoritative Panda trajectories, included by read-only aggregation.
- `outputs/revision_compute_allocation/`: historical allocation boundary analysis.
- `docs/TASK_CONTRACT_ALIGNMENT_FINDINGS.md`: the five prespecified answers.

The source manifest and figure manifest identify every active evidence artifact.
No previously frozen result, query identity, solver, verifier or model was changed.

## Active package

- `generated/evidence_snapshot.json`: sources, hashes, tables and formatted values.
- `generated/paper_numbers.tex`: empirical-number macros used by the manuscript.
- `generated/task_*_rows.tex`: six tables (three main, allocation, status, family).
- `source_data/task_*.csv`: active source-data tables and complete supplementary
  families, intervals, errors, first failures and completion UID sets.
- `figures/figure1_taxonomy.*` through `figure5_family_effects.*`: five main
  figures in PDF, SVG and PNG.
- `figures/supplement_dls_excess_iterations.*`: direct DLS trace figure.
- `references.bib`: 38 references, 22 dated 2024–2026; verified identities and
  explicit publication-status/optional-metadata qualifications.
- `generated/final_qa.json`: source, preservation, citation and compilation checks.

Older non-`task_` CSVs, older figure names and old generated fragments are retained
but unused by the current paper. The complete previous manuscript is also preserved
in `history/cghik_paper_02aa287.tar.gz` from the incoming baseline. Existing
`historical/` contents are historical as well.

## Interpretation and submission status

Read `docs/FINAL_PAPER_CLAIM_MAP.md`, `docs/FINAL_PAPER_CHANGELOG.md`,
`docs/TASK_CONTRACT_PAPER_GUIDE.md` and
`docs/REFERENCE_VERIFICATION_TASK_CONTRACT.md`.

This is a complete generic engineering-journal research manuscript, not an actual
journal submission or a hard-real-time/hardware demonstration. Author identities,
affiliations, CRediT, funding/conflict declarations and the chosen journal's final
formatting require author input. No archival DOI is invented; the public repository
and frozen commit provide the present reproducibility reference.
