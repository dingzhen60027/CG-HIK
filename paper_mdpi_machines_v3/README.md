# Machines manuscript package

This directory contains the reproducible pre-submission manuscript for the
confidence-gated hybrid inverse-kinematics study.

## Status

- Target journal: **Machines** (MDPI), section *Robotics, Mechatronics and
  Intelligent Machines*.
- Scientific evidence: frozen `paper_v2` test outputs plus the strictly
  validation-only `latency_pilot_v3` implementation study.
- Submission blocker: an independent optimized `test_v3` has **not** been run.
  The manuscript therefore labels optimized latency as validation evidence and
  does not present it as an independent test result.
- Release status: the first locked packaging attempt retained five passing
  robot/seed combinations but stopped on the strict deployment-equivalence gate
  for `ur5e/seed43`. A validation-only diagnostic reproduced maximum risk
  probability and score differences of `1.259825577e-12` and
  `1.144973005e-12`, slightly above the locked `1e-12` limits, while every
  runtime action/outcome agreement remained exact. The incomplete package is
  preserved outside this manuscript directory, and no `test_v3` process was
  started.
- Author names, affiliations, funding, repository DOI, and corresponding-author
  details are explicit placeholders that must be supplied before submission.

## Rebuild

From the repository root:

```bash
/home/eric/anaconda3/envs/isaaclab_3/bin/python paper_mdpi_machines_v3/scripts/build_evidence.py
/home/eric/anaconda3/envs/isaaclab_3/bin/python paper_mdpi_machines_v3/scripts/make_figures.py
cd paper_mdpi_machines_v3
/home/eric/.local/bin/latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The evidence script reads the frozen experiment outputs and writes only inside
this manuscript directory. It records SHA-256 hashes for every JSON evidence
source used by the paper.

## Submission-template note

`main.tex` is a self-contained, reliably compilable MDPI-structured review
draft. Before uploading to the journal, copy the content into the latest
official MDPI LaTeX template downloaded from the MDPI author portal. This avoids
silently claiming that a locally cached class file is the current publisher
version.

## Package map

- `main.tex` and `main.pdf`: complete English pre-submission manuscript and its
  compiled PDF;
- `references.bib`: verified bibliography used by the manuscript;
- `figures/`: editable PDF/SVG figures plus PNG/TIFF review copies;
- `source_data/`: figure- and table-level CSV evidence;
- `source_data/formal_comparator_*.csv`: streamed same-protocol results for all
  six declared formal comparators;
- `scripts/`: read-only evidence extraction and deterministic figure generation;
- `generated/evidence_snapshot.json`: source paths and SHA-256 hashes;
- `claim_evidence_map.md`: boundary between formal-test, validation-only, and
  pending claims;
- `reviews/`: three isolated pre-submission reviewer reports, their post-review
  synthesis, and an evidence-bounded revision response;
- `future_v4/dual_abstention_research_plan.md`: prospective two-sided
  abstention/OOD study; it is deliberately not represented as current evidence;
- `terminology_ledger.md` and `revision_decision_log.md`: controlled terms and
  the decision record for the evidence-bounded revision;
- `qa_report.md` and `submission_checklist.md`: technical QA and remaining
  author/release actions.
- `release_v3_packaging_diagnostic.md`: exact reason Phase A stopped.
