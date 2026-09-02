# Hierarchical V5-Lite policy-validation pilot

| robot | method | success | P50 ms | P95 ms | P99 ms | mean FEV | seed invocation |
|---|---|---:|---:|---:|---:|---:|---:|
| panda | always_local | 0.779361 | 0.529549 | 0.603334 | 0.609800 | 3.213268 | 0.000000 |
| panda | always_hard | 0.998034 | 1.662015 | 2.061657 | 2.515428 | 4.156757 | 1.000000 |
| panda | counterfactual_cghik_v4 | 0.998034 | 1.886723 | 2.300567 | 2.781588 | 4.156757 | 1.000000 |
| panda | hierarchical_cghik_v5 | 0.998034 | 0.964983 | 2.712867 | 3.298753 | 4.181818 | 0.266830 |
| panda | hierarchical_cghik_v5_lite | 0.998034 | 1.868631 | 2.272608 | 2.721042 | 4.156757 | 1.000000 |
| ur5e | always_local | 0.649585 | 0.486401 | 0.551513 | 0.558480 | 3.344558 | 0.000000 |
| ur5e | always_hard | 1.000000 | 1.549174 | 1.916241 | 2.238751 | 3.738409 | 1.000000 |
| ur5e | counterfactual_cghik_v4 | 1.000000 | 1.766122 | 2.134542 | 2.435280 | 3.738409 | 1.000000 |
| ur5e | hierarchical_cghik_v5 | 1.000000 | 2.104673 | 2.529705 | 2.961177 | 3.757443 | 0.586628 |
| ur5e | hierarchical_cghik_v5_lite | 1.000000 | 1.742213 | 2.105574 | 2.521601 | 3.748170 | 0.960469 |
