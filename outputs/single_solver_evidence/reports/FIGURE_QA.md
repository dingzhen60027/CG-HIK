# Source-backed figure checks

Python/matplotlib only; all figures use the saved results, no invented points.
SVG text remains text; PDF uses embedded TrueType (type 42). Exported PDF audit:
task_cost 92 text runs, paired_contributions 77, qp_timing 22; minimum 6 pt,
zero runs below the 5 pt floor. Figures are 183 mm wide.

Source preflight: 17 PASS, 0 FAIL, three reviewed warnings. PNG at 300 dpi is a
preview, not a TIFF submission image (no submission manuscript requested).
The width warning is a static-parser limitation: it reads the expression
183/25.4 as 183 inches; actual exported dimensions are 183 mm. The source font
and explicit SVG/PDF export checks pass. Matplotlib used a temporary cache
because the home config cache is read-only; font rendering and exports succeeded.

| Figure/panel | Question and statistic | Unit/variation | Rendered check |
|---|---|---|---|
| task_cost a | Panda geometric completion | 160 UIDs, mean and three nested repeats | All seven labels; no collision |
| task_cost b | Panda within-20ms completion | Same UIDs/repeats | All late calls retained; labels clear |
| task_cost c | Panda cumulative cost | Whole-sweep mean and repeat range | No truncated time values |
| task_cost d | UR5e geometric completion | 160 UIDs, nested repeats | All seven settings visible |
| task_cost e | UR5e within-20ms completion | Same UIDs/repeats | Labels clear; not hard-real-time claim |
| task_cost f | UR5e cumulative cost | Whole-sweep mean and repeat range | Range does not collide with labels |
| paired_contributions a | Panda TSR paired differences | UID means, 4000 stratified bootstrap, unadjusted 95% | Zero reference and all six controls clear |
| paired_contributions b | Panda DTSR paired differences | Same definition | No uncertainty omitted |
| paired_contributions c | Panda paired cost ratio | Same definition | Ratio-one reference clear |
| paired_contributions d | UR5e TSR paired differences | Same definition | Negative/zero/positive effects preserved |
| paired_contributions e | UR5e DTSR paired differences | Same definition | No sign-based exclusion |
| paired_contributions f | UR5e paired cost ratio | Same definition | Adverse ratios above one visible |
| qp_timing a | Panda full local call distribution | 971 fixed QPs, 5 nested repetitions | Dot=P50, orange dash=P95, triangle=P99 |
| qp_timing b | UR5e full local call distribution | 1000 fixed QPs, 5 nested repetitions | Same notation; Clip is not a quality-equivalent solver |

Task-cost ranges represent search/timing repetition, not confidence intervals.
QP quantiles are distribution summaries, not uncertainty intervals. Source data
and complete four-family tables remain beside the plots. The title-case
"Ur5e" in plot titles denotes the same UR5e robot, not a separate model.

The paired panels are necessary because near-ceiling marginal completion plots
alone hide paired gains/losses. The narrow, sometimes degenerate intervals are
conditional on the observed UID/repeat sample and are not equivalence proofs.
