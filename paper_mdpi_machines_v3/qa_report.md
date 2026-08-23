# Manuscript QA report

QA date: 23 August 2026

## Evidence and claim controls

- Formal efficacy claims use only the six frozen `paper_v2` test runs.
- Optimized-inference latency is labelled validation-only throughout.
- The failed eager feasible-P95 formal gate is reported as a negative result.
- No `test_v3` result is claimed; the manuscript states that it has not started.
- Three training seeds are treated as sensitivity repeats on a shared test set,
  not as independent query datasets.
- Null ablations are retained and do not support component-necessity claims.
- The exact-URDF scope is explicit: no collision, torque, controller-tracking,
  contact, or physical-robot safety claim is made.
- Thirty-six frozen evidence inputs, including the six streamed formal
  query-level comparator files, are path- and SHA-256-recorded in
  `generated/evidence_snapshot.json`.

## Reproducibility artifacts

- Four scientific figures have source CSV files and a deterministic generation
  script.
- Figure deliverables exist as PDF, SVG, PNG, and 600-dpi TIFF.
- Figure source preflight: 20 passed, 0 warnings, 0 failures.
- Figure text audit: minimum detected text size 6.2 pt; all panels passed.
- Bibliography contains 23 cited references in the compiled manuscript. The
  2025 IK-Geo article, 2026 viability-QP article, and RSS 2025 runtime-failure
  paper were verified against primary publisher/proceedings records before use.

## LaTeX and PDF checks

- `latexmk` compilation completed without undefined citations, undefined
  references, overfull boxes, or undefined control sequences.
- PDF: 15 A4 pages, PDF 1.7, unencrypted.
- Extracted page-text lengths are nonzero on all 15 pages.
- Extracted text contains no `??`, Unicode replacement character, or
  `undefined` marker.
- Visual inspection covered all 15 rendered pages and the four figures; no
  clipping, overlap, or illegible panel was observed.
- Six red `AUTHOR INPUT NEEDED` markers intentionally remain and block
  submission.

## Experiment-release status

- The locked backend remains `torchscript_exact`; no alternative backend was
  selected after the validation pilot.
- The first `release_v3_locked` attempt stopped at strict equivalence for
  `ur5e/seed43`; a read-only reproduction isolated the failure to maximum stored
  validation risk-probability (`1.2598255771933964e-12`) and risk-score
  (`1.144973005295924e-12`) differences, both just above the locked `1e-12`
  tolerances. All 3700 paired runtime records retained exact route, accepted,
  fallback, FEV, executed-stage, verification-reason, and command agreement.
  The incomplete output was retained for audit.
- `test_v3` was not generated or started.
- Frozen `paper_v2` and `latency_pilot_v3` outputs were read but not modified.

## Interpretation

The PDF is a complete, evidence-bounded pre-submission manuscript, not yet a
submission-ready final. Submission requires author metadata, completion of the
locked release, the preregistered one-shot `test_v3`, an evidence-status update,
and migration into the latest official MDPI LaTeX template.
