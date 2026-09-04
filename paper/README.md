# CG-HIK paper package

This directory contains the final evidence-backed manuscript:

> **CG-HIK: Query-Adaptive Tail-Latency Routing for Kinematically Verified Online
> Inverse Kinematics**

The paper's organizing statement is:

> Learning allocates solver effort per query; numerical geometry generates joint
> commands; deterministic verification governs acceptance.

## Build

From the repository root:

    conda activate isaaclab_3
    PYTHONPATH=src python paper/scripts/build_evidence.py
    python paper/scripts/make_figures.py
    cd paper
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The first two commands are read-only with respect to outputs/. They regenerate:

- generated/evidence_snapshot.json
- generated/paper_numbers.tex
- generated table rows
- source_data CSV files
- five figures in SVG, PDF, and PNG

The compiled manuscript is main.pdf.

## Evidence sources

- Development heterogeneity and predictability:
  ../outputs/counterfactual_v4_bulk/
- Frozen exact predictor and policy:
  ../outputs/release_v4_locked/
- Fresh point-query results:
  ../outputs/test_v4_aggregate_repair_v1/
- Final fresh transition-rich trajectories:
  ../outputs/fresh_transition_v4_test/

All reported numbers are regenerated from frozen JSON/NPZ artifacts. No manuscript
build command trains a model, runs an IK solver, creates a test query, or modifies a
frozen output.

## Package layout

- main.tex / main.pdf — manuscript source and compiled paper
- references.bib — 40 relevant references: 38 field-verified records and 2 records
  with explicitly disclosed metadata checks
- scripts/ — evidence and figure builders
- generated/ — machine-generated snapshot, TeX macros, and table rows
- source_data/ — active figure/table CSVs regenerated from frozen evidence
- figures/ — five main figures in SVG/PDF/PNG
- historical/figures/ — superseded manuscript figures, retained but not cited
- historical/source_data/ — superseded manuscript-derived CSVs retained for traceability;
  these are not inputs to the final paper

Claim boundaries and provenance are summarized in:

- ../docs/FINAL_PAPER_CLAIM_MAP.md
- ../docs/FINAL_PAPER_CHANGELOG.md
- ../docs/PAPER_STORYLINE.md

The manuscript is a blinded generic-journal draft. Author names, affiliations,
funding, and archival DOI remain the only submission-specific fields to supply.
