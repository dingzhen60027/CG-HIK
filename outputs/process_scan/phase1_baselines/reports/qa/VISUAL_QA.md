# Review-artifact quality check

These are Phase-1 review sheets, not paper submission figures. The plots use
the Python/matplotlib backend, measured source CSV/NPZ/HDF5, editable SVG/PDF
and300 dpi PNG. No synthetic performance values or image-generation model was
used. Each of the eight assembled figures and its panels was visually inspected.
The five physical-detail sheets use row-shared external legends so labels do
not conceal data. The minimum PDF glyph size is6 pt (overview/status8 pt).

`panel_audit.csv` records each panel's question, statistic, unit and uncertainty
scope. Overview points are scene-level values, with timing repeats averaged
inside the scene; median bars are descriptive, not inferential error bars.
The separate paired figure uses the declared CAD/placement cluster bootstrap.
The status map retains unavailable/missing cases, not just executed successes.
Physical sheets show all four offline settings for the predetermined placement0,
u-direction examples; the unavailable Panda cylinder is explicitly indexed.

Source-preflight warnings were reviewed: no TIFF is required for this review;
300 dpi previews accompany vector originals; large review-sheet widths are
intentional rather than journal-column layouts; `default_rng` is used for
bootstrap resampling, not invented measurements; `rotation=` in the source is
a saved CAD matrix, not a rotated text label. No automated FAIL was reported.

Visual inspection is distinct from `artifact_verification.json`: the latter
checks360 condition identities,84 physical source files and63 exact common B1
initializations without executing any solver or dynamics. The saved records
include physical failures. A5 ms displayed trace is not substituted for the1 ms
maximum used for acceptance; the latter is printed separately on each motion
panel.

Video previews sample existing measurements at10 fps in real simulated time.
Their source hashes, frame counts, durations and beginning/middle/end decoding
are independently checked in `video_checks.json`. The contact sheet is a
midpoint visualization only, not evidence of full completion; actual completion
comes from the complete recorded measurement and metric files.
