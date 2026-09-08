# CR-IK final artifact checks

This check documents the completed experiment; it does not authorize another
solver run, parameter choice, or algorithm branch.

## Numerical evidence and preservation

- The pre-evaluation code/configuration/identity commit is
  `e13f32f53a451eb19879292dbfa8e382edb4bc39`. It was pushed before the formal run.
- Both complete trajectory runs and the fixed-site perturbation study finished
  once. Their exclusive output directories and raw manifests remain intact.
- The FK-only replay checked 1,008,000 frame records, 813,615 accepted commands,
  and 389,998 nominal-next pairs. There were zero acceptance, feedback, or
  deadline-label discrepancies. The replay did not invoke an IK solver.
- The 34 pre-evaluation tests passed, including native adapter smoke. Two
  additional postprocessing-only tests check unavailable-site retention and
  identical-input requirements for paired probes.
- The aborted development serialization run and the completed replacement are
  both retained and explicitly separated from formal results. No formal solver
  rerun or post-outcome change of a numerical setting was made.
- The earlier paper and tracked evidence at `276f3d311aafc67786c916c5fec596dd860cc5dc`
  were not modified. The unrelated user file `docs/STORY_FLOW_AUDIT.md` is not
  part of this work and must not be staged.

## Figure QA

Primary delivered figures are in `outputs/correction_reserve_ik/formal_figures_final/`.
Earlier plot layouts remain as drafts; the final version only moves a trace
legend outside the data area and labels the residual axis transformation.
No observation, aggregation, or uncertainty calculation changed in that edit.

All four figures were inspected at rendered size. Labels, shared legends,
uncertainty bars and axes are readable. The PDF text audit reports a 7 pt minimum
for each file, with zero glyphs below the 5 pt check threshold. SVG/PDF text is
editable; PNG previews are 600 dpi. Figures are 183 mm wide.

The static source preflight has 17 passes, 3 warnings and no failures:

| Warning | Resolution |
|---|---|
| No TIFF | These line plots are delivered as editable PDF/SVG; PNG is explicitly a preview, not a promised journal submission format. |
| Random sampling detected | The RNG resamples observed trajectory UIDs for bootstrap intervals; it generates no synthetic observations. |
| Log positivity guard not recognized | The source explicitly asserts that all plotted P95 values are positive. No pseudocount is added. The residual trace is linear to 1 and logarithmic above 1, stated on its axis. |

Completion intervals resample whole UIDs within family after averaging nested
TRAC searches. P95 latency is a descriptive pooled-frame quantile, not an
inferential mean. Family comparisons retain all 20 UIDs. The mechanism scatter
requires an admissible current command and an available nominal/unperturbed next
step; missing states remain in the tables and the all-input failure denominator.
Its apparent radius is limited to the tested signed axes and capped at four
tolerance units, not a nonlinear robustness certificate. Three search repeats
and multiple states of one trajectory are not independent trajectories.

The main comparison includes every failure and late return. Raw CRLF CSV line
endings are preserved; they are standard CSV output, not whitespace defects in
the scientific records. The code/doc whitespace check is separate.

## Interpretation checks

The findings report presents both positive and negative evidence: development
UR5e, fresh paired uncertainty, added computation, rare calls beyond 20 ms,
larger but admissible residuals, competitive Pink/TRAC-IK configurations, and
the narrow-contract RangedIK adapter limitation. It does not infer a global
nonlinear guarantee from gamma or mistake solver failure for infeasibility.

The statistical and figure skills influenced whole-trajectory inference,
explicit uncertainty/availability labels, readable vector exports, and the
separation of mechanism examples from independent performance evidence.
