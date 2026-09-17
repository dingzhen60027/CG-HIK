# Figure and reporting review

The four generated PNGs were visually inspected at full-panel resolution. Axes,
units, legends, panel labels, missing-result treatment and paired subsets are
readable. All values come from the Phase 1.5 records, not schematic examples.

PDF text-operator audit with a 6 pt floor:

|Figure|Minimum text size|Text runs|Below floor|
|---|---:|---:|---:|
|core_baselines|7 pt|82|0|
|calibration_closure|6 pt|72|0|
|one_ms_calibration_traces|6 pt|106|0|
|bottlenecks_and_paired_times|8 pt|58|0|

All PDF audits were auditable with no warnings. SVG text remains editable and
PDF exports are vector outputs. PNG is a 300 dpi review preview, not a claimed
journal production deliverable. Static figure preflight had no failures; TIFF,
600 dpi raster, and journal column-width warnings are intentionally inapplicable
to this review packet. No journal submission figure specification was imposed.

The calibration before/after comparison changes the common route and spline
resolution as well as the speed reserve. It is not an isolated causal effect of
one safety factor. Both settings passed all six calibration cases. Near-limit
occupancies are nonexclusive and are not additive time attribution.

The report generator initially stopped on an indentation error in a figure-axis
label line. Only this read-only reporting issue was corrected. No solver or
physical run was repeated. Subsequent reduction completed from all 288 records.

Time partition review found that the original classification counters include
the final timestamp, adding 1 ms to their sum. `bottlenecks.csv` now partitions
the saved adjacent timestamp intervals with left-endpoint labels and asserts
that the parts sum to actual elapsed time. The original counters are preserved
in separate columns and in raw metrics. Cycle timestamps, controller behavior,
acceptance, and all executions are unchanged.

Panda and UR5e plane/B1 replay frames at 5 s were visually inspected. Robot,
workpiece, ray fan, actual-time label, and simulation/quality labels are visible.
These previews are extracted from recorded-state replay videos, not new physics.

Statistical review uses complete CAD-placement clusters, timing means within
scene, and explicitly conditional paired cycle-time comparisons. Failure yield
retains all scheduled scenes. The literature check is confined to the two
specified prior works; no novelty conclusion is inferred from these plots.
