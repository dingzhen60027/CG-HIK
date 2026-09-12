# Scientific figure QA

Contract: Python quantitative grid, 183 ×104 mm; one figure links same-input
local numerical effects to six prescribed historical outcomes. No synthetic
data, no selected favourable robot/case, no causal claim from differing future
states. Editable SVG/PDF; 600 dpi PNG preview, not a submission TIFF requirement.

| Panel | Unique evidence | Statistic / unit | Visual and evidence checks |
|---|---|---|---|
| a | Bounded minus clipped direction, all six common inputs | Raw dimensionless joint step; no interval | Both directions use identical H/g/lo/hi and q. Black outlines are strong KKT-active bounds, not command proximity. UR5e joint 7 is absent (blank), not zero. Symmetric log colour normalization is explicit in source. |
| b | True nonlinear task residual after the same full step | Raw FK task cost / pre-step cost; six pairs | All values positive; log axis has plain-text ticks. Full step can be an unaccepted internal trial, not final command. No averaging across independent inputs. |
| c | Subsequent original completion or first failure | Raw successful prefix, repeat 0; all three repeats agree | Bounded-loss UR5e 131 included. This is historical follow-up, not causal isolation after the commands diverge. |

No error bars: panels show exact selected input-level values and identical
historical discrete outcomes across repeats, not stochastic aggregate estimates.
Timing bootstrap is separately defined in the tables at source-UID level; the
RNG in the shared reporting file generates bootstrap indices, not figure data.

Static preflight: no FAIL. Warnings reviewed: PNG is a preview with PDF/SVG
delivery, not a required TIFF; random generator is the table bootstrap only;
log positivity is explicitly asserted. PDF final glyph audit minimum 7 pt.
Panel-by-panel and combined PNG review checks titles, labels, active outlines,
axis ticks, legend, colour bar, and footer clearance. Earlier label collision
and 4.9 pt automatic log exponents were corrected without changing data.

Sources: `mechanism/six_same_input_cases.json`, original copied history JSONL,
and `reports/local_to_trajectory.csv`. Figure generated solely by
`src/confik/constrained_qp_reporting.py::figure`.
