# Figure QA — minimal-intervention CR-IK development

Conclusion: the large cost reduction is visible together with, not instead of, the completion losses.

Backend: Python/matplotlib only. Each figure is 183 mm wide. Editable SVG/PDF and 600 dpi PNG previews are provided. No old figure was replaced.

## Statistics and panel audit

Independent units are the 40 complete development trajectory UIDs per robot. Three TRAC search repeats are nested within each trajectory; Pink has one deterministic run. This is an observed development comparison, not a fresh test. There are no biological replicates, training seeds, p-values, or multiplicity-adjusted claims. All categories and failures are retained.

| Figure / panel | Unique question | Center and uncertainty | Unit | Visual audit |
| --- | --- | --- | --- | --- |
| Completion/cost a | Panda completion retained? | Paired mean TSR difference, 95% family-stratified UID bootstrap CI | 40 trajectories | Labels, zero line, points and intervals clear |
| Completion/cost b | UR5e completion retained? | Same as a | 40 trajectories | Same scales, labels and uncertainty as a |
| Completion/cost c | Panda end-to-end cost reduced? | Ratio of aggregate cumulative times, paired 95% UID bootstrap CI | 40 trajectories | Positive log-axis quantities explicitly asserted; reference ratio 1 visible |
| Completion/cost d | UR5e end-to-end cost reduced? | Same as c | 40 trajectories | Same scales and uncertainty as c |
| Trace top left | Panda causal demand versus actual forecast error and found reserve | One raw trajectory, no aggregate CI | First sorted high-curvature UID, repeat 0 | Linear-to-1/symmetric-log scale stated; missing reserve left as gaps |
| Trace top right | UR5e demand under different observed motion | Same fixed UID selection rule | One trajectory | Same legend and scale interpretation |
| Trace middle left | When Panda actually invokes SOCP | Raw call count per frame | Same trajectory | Discrete 0/1/2 counts visible |
| Trace middle right | When UR5e actually invokes SOCP | Raw call count per frame | Same trajectory | Same discrete scale |
| Trace bottom left | Panda complete-call cost at those target times | Raw old/new latency, 20 ms reference | Same target UID, different method-specific states | Costs not cropped to successful frames; no smoothing |
| Trace bottom right | UR5e complete-call cost at those target times | Same as left | Same target UID, different method-specific states | Common method colors and timing definition |

Trace selection uses family and sorted UID, not method completion. The actual next-target prediction error is joined only after online execution; the plot does not imply future-target access by the algorithm. Frame traces are descriptive and are not treated as independent replicates.

## Automated checks and exports

- Source preflight: 18 PASS, 2 WARN, 0 FAIL.
- PDF text: minimum rendered glyph size 7 pt in both figures, zero runs below the 5 pt audit floor.
- Warning 1: PNG rather than TIFF is intentional; these are development previews, and editable vector outputs are primary.
- Warning 2: the static log-guard detector misses the explicit positive center/CI-endpoint assertion; no offset or exclusions were used.
- Both assembled PNGs were visually inspected panel by panel. No label/data collisions, clipped uncertainty extents, or hidden legend entries were found.
- Orange/blue distinguish original and minimal methods; lines, panel position, and labels provide additional structure. No red/green encoding or rainbow map is used.
- No image edits, cropping, interpolation, or contrast adjustment were applied.

Source data: ../reports/source_data.json and trace_source_data.json. Raw trace paths and SHA-256 values are recorded in trace_source_data.json. Complete export metadata and selection definitions are in manifest.json.
