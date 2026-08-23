# Figure contract

Core conclusion: a calibrated gate reallocates a shared verified solver
cascade's computation and rejects futile queries without degrading the locked
kinematic acceptance contract, while implementation-only optimization removes
the learned gate's fixed overhead on validation.

Figure archetype: schematic-led composite for Figure 1; quantitative grids for
Figures 2-4.

Target journal/output: Machines; 183 mm maximum width; editable PDF/SVG plus
600-dpi TIFF and PNG preview.

Backend: Python (matplotlib), used exclusively for all manuscript figures.

Panel map:

- Figure 1: proposal, calibrated routing, shared numerical cascade, verifier,
  and reject/fallback logic.
- Figure 2a: feasible-query function evaluations in all six frozen runs.
- Figure 2b: feasible-success paired differences in percentage points.
- Figure 2c: rejectable-query P95 latency in all six frozen runs.
- Figure 2d: the prespecified formal-test feasible-latency failure.
- Figure 3: diagnostic ablations, retaining the mostly null ablations.
- Figure 4: validation-only P95 ratios, eager-to-exact reduction, stage P95
  profile, and percentile distributions.

Evidence hierarchy: Figure 2 is the primary locked-test evidence; Figure 3 is
diagnostic; Figure 4 is validation-only readiness evidence and is visually
marked as such.

Statistics: paired query-level formal inference remains in the manuscript
tables/text. Figure 2 shows all three training-seed sensitivity runs for each
robot; the seeds share one test set per robot and are not treated as independent
datasets. Figure 3 bars are mean +/- one seed standard deviation over the same
three sensitivity runs with all raw points shown; this is a sensitivity summary,
not an inferential sample. Figure 4 uses 750 paired validation feasible queries
per robot.

Source data: generated CSV files under `source_data/`, built from frozen JSON
outputs with SHA-256 provenance in `generated/evidence_snapshot.json`.

Reviewer risk: the formal v2 feasible-latency gate failed and the optimized
backend has validation-only evidence. The figures must not imply independent
test confirmation before `test_v3`.
