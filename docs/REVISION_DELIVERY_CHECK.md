# Supplementary revision delivery check

Completed on 2026-09-06 in `codex/hierarchical-v5`. This is a delivery record, not an additional experiment or an acceptance gate.

## Completed scope

- A: read-only decomposition of original trajectories, with additive time/FEV totals, and matched-population five-entry diagnostics.
- B: 3,000 fresh point queries per robot, ten measured configurations with three interleaved calls; 500 predefined oracle-diagnostic queries per robot, three entries with ten interleaved calls.
- C: 40 witnessed reference trajectories per robot, 150 frames each, four methods; every fixed target frame is retained.
- Findings and supplementary tables were completed before the manuscript rewrite. Query- and trajectory-paired intervals preserve the appropriate sampling unit.
- Four publication figures and the revised manuscript are exported. The manuscript is 21 pages, including the unchanged original numerical results in an appendix.

The interrupted first adapter attempt remains in `outputs/revision_compute_allocation/diagnostic_adapter_attempt_01/`, excluded from analysis. Its correction and the measurement sequence are disclosed in the protocol, measurement notes, findings, and manuscript. No outcome-driven model or threshold adjustment was made.

## Focused checks

- `tests/test_revision_compute_allocation.py`: 20 tests passed in the final check.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: successful; no undefined references, overfull boxes, or LaTeX warnings in the final log.
- All manuscript pages were visually reviewed; affected pages were re-rendered after the final edits. Tables, figures, references, and page boundaries have no observed clipping or overlap.
- Publication figures retain editable PDF/SVG output. The figure validator reported 18 passes, two preview-format warnings, and zero failures; standalone PDF text checks found a minimum 7 pt font. PNG files are previews, not the submission masters.
- Selection seal, configuration, dataset identities, frozen release files, and final measurement-source hashes all match their recorded values.
- Supplementary source tables, paper-export tables, and paper figures match the evidence manifest. The findings file still matches the hash recorded before the manuscript rewrite.
- `git diff --check`: passed. Existing tracked changes are confined to `paper/main.tex` and `paper/main.pdf`; the new supplementary module, configuration, tests, documents, and paper exports are separate additions.
- No existing tracked solver, verifier, configuration, original output, bibliography, or original paper source-data/generated/figure file was changed. The pre-existing `docs/STORY_FLOW_AUDIT.md` was left untouched.

## Delivered artifact hashes (SHA-256)

| Artifact | SHA-256 |
| --- | --- |
| `paper/main.tex` | `57ca001ef6702e69e2ba5288a014f3734b70a1be4e70a7e892b291ba59411d5f` |
| `paper/main.pdf` | `b6ca1407aac104d53c9a5c22728e5934cb97529bda7ee721de02b4bfa0541cb0` |
| `docs/REVISION_EXPERIMENT_FINDINGS.md` | `7c56df4e591166ffa1dfd6c4a78736fdbf3f0ffac5db84547ec012b21c0d7c3e` |
| `outputs/revision_compute_allocation/reports/SUPPLEMENTARY_MAIN_TABLES.md` | `02f9ab10b4831e182053bf5efa651e72e67823ac2ca5a5d113dbaafab0dd546d` |

Detailed source/figure hashes are in `paper/revision_generated/evidence_manifest.json`; complete call-count and additivity checks are in `outputs/revision_compute_allocation/reports/analysis_delivery_manifest.json`.

No further experiment is required for this delivery. No new temporal method, network, or timing parameter was introduced. The initial delivery was left in the local worktree. The subsequent explicit user request authorizes committing and pushing this supplementary package; the enclosing Git history identifies that publication. The pre-existing, earlier-stage `docs/STORY_FLOW_AUDIT.md` is outside this package and remains untouched.
