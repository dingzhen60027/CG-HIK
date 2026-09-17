# Phase 2 figure and statistics QA

These are development-report figures, not submission figures. Source data are
the frozen Phase 2 measurements reduced by `phase2_reporting.py`; no simulated
display data or replacement outcomes were inserted. All 24 scenes remain in
completion counts. Cost comparisons explicitly identify jointly qualified subsets.

The nature-statistics and nature-figure checks were used to keep scene-level
pairing, nested timing repeats, inspectable source data and visual verification.
Three planning runs are averaged within scene, not treated as three independent
physical trials. Only the preassigned repeat 0 has physical evidence. The CSV
intervals resample CAD-by-placement clusters with direction nested, separately
for each robot; they are marginal descriptive intervals, not simultaneous tests.

## Panel audit

|Figure / panel|Question and summary|Unit / variability|Rendered check|
|---|---|---|---|
|quality_cost, left / Panda|How many of all 12 scenes pass quality?|Counts; no sampling-error bar|10 for A1, 9 for Proposed; failures not hidden|
|quality_cost, left / UR5e|Does UR5e completion persist?|Counts; no sampling-error bar|B2-common 10; remaining settings 12|
|quality_cost, middle / both robots|What actual cycles are observed?|All qualified scene points and black median; not a paired causal comparison|All values visible; readable category labels|
|quality_cost, right / both robots|How much total planning work was charged?|All 12 within-scene means and black median, including failures|No optimization or verification stage subtracted|
|paired_changes, left / both robots|What is the within-scene cycle change?|Paired scene values, black median, exact n below categories|Moved n labels away from extreme points; no masking|
|paired_changes, right / both robots|What is the within-scene turn-time change?|Same paired subset as left; raw points and median|Unfavorable positive changes retained|
|graph_to_path, left|Does graph connectivity yield a verified C² fit?|All 24 scenes; fraction of three timing repeats|Zero-fit cases marked by crosses; sparse tick labels do not omit bars|
|graph_to_path, right|When do accepted updates reduce real retimed duration?|Every repeat-0 accepted-history curve; not physical success or an interval|No extrapolation; B2/A1/Proposed labels and colors consistent|

All three final PNGs were visually inspected. The SVG/PDF exports preserve text.
PDF glyph minima: 7 pt, 8 pt and 7 pt; no glyph below 5 pt. Source preflight:
16 PASS, 4 WARN, 0 FAIL. `automated_qa.json` retains the machine-readable checks.

Warnings are intentional for this deliverable: PNG is a 300 dpi preview while
SVG/PDF are the primary editable exports; no journal-size/TIFF submission is
being prepared. The RNG flag is the declared cluster bootstrap, not fake data.
Comparable plot panels show raw scene distributions consistently; confidence
intervals and small-cluster limitations are in the companion CSV/report.

## Recorded-state previews

The predeclared 36-entry video index contains 27 rendered executions and nine
explicit not-executed entries, without replacing failed selections. All videos
were checked with ffprobe, have 640×480 frames at 10 Hz, and retain hashes of their
immutable 1 kHz source histories. Two representative frames were visually checked
and retained as `videos/qa_panda_frame.png` and `videos/qa_ur5e_frame.png`.

Rendering uses separate MuJoCo visualization data only; it does not run another
controller or write the saved physical records. A rendering-only performance
fix materializes compressed ray arrays once instead of decompressing them for
each displayed ray. A partial preview was restarted, without changing any
planning, dynamics, measured state, ray return or quality statistic.
