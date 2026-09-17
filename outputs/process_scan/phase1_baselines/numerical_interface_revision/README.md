# Floating-point duplicate grid repair

During the first development batch, an input-grid audit found seven pairs of
v-direction timing nodes differing by about2.8e−17 rather than representing
distinct path locations. B2 forms `u=diff(x)/(2 diff(s))`; these numerical
zero-length cells can damage conditioning. This is an implementation defect,
not a method, budget, quality or scene selection change.

The initial partial batch, its B1 snapshots, and its exact source implementation
are retained under `pre_grid_merge/`. The initial input seal remains untouched
at `inputs/seal.json`. No scenes, targets, source q, or identity are regenerated.

A common numerical-grid function now merges only representations separated by
less than1e−12 in normalized progress. Every original node lies within1e−12 of
a retained point; the u grid is bit-for-bit unchanged. Both TOPPRA and the full
joint-time NLP use that function. No algorithm parameter, objective, constraint,
geometry initializer, physical controller or acceptance threshold changes.

All360 scheduled conditions are rerun as one comparable batch under
`numerical_seal_grid_repair.json`; preliminary results are not mixed into the
main table. The interrupted condition remains incomplete in the preserved
partial batch and is not labeled solver failure. The time and unfavorable
outcomes of that batch are not deleted.
