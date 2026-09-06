# Supplementary computation-allocation tables

New supplementary evaluation; the original trajectory grouping is post-hoc descriptive. Ratios and 95% percentile intervals use paired, family-stratified resampling of queries (three nested calls) or entire trajectories (150 dependent frames), with 4,000 resamples. All fixed-length calls are retained. Quantiles below are whole-call empirical quantiles, not sums of stage quantiles. No composite pass/fail gate is used.

## Witness-feasible points: 2,500 queries per robot

| Robot | Method | Verified % | Within 20 ms % | Mean ms | P50 | P95 | P99 | Mean FEV | Reject / defer / fallback calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda | Always-hard | 94.920 | 93.320 | 15.107 | 1.501 | 89.893 | 347.605 | 20.559 | 0 / 0 / 534 |
| Panda | CG-HIK | 94.920 | 93.307 | 15.610 | 1.984 | 90.488 | 348.073 | 20.583 | 0 / 0 / 534 |
| Panda | Geometry rule | 94.920 | 93.307 | 15.596 | 1.984 | 90.321 | 348.077 | 20.572 | 0 / 0 / 534 |
| Panda | P50-selection | 94.920 | 93.307 | 15.634 | 1.989 | 90.765 | 348.019 | 20.702 | 0 / 0 / 534 |
| Panda | Reject-only + hard | 94.920 | 93.293 | 15.613 | 1.982 | 90.633 | 348.045 | 20.559 | 0 / 0 / 534 |
| Panda | Routing-only | 94.920 | 93.307 | 15.612 | 1.986 | 90.548 | 348.007 | 20.583 | 0 / 0 / 534 |
| Panda | TRAC-IK 100ms | 97.880 | 97.880 | 0.263 | 0.229 | 0.471 | 0.656 | not comparable | 0 / 0 / 0 |
| Panda | TRAC-IK 20ms | 97.827 | 97.827 | 0.264 | 0.230 | 0.472 | 0.645 | not comparable | 0 / 0 / 0 |
| Panda | TRAC-IK 400ms | 97.933 | 97.933 | 0.264 | 0.230 | 0.472 | 0.646 | not comparable | 0 / 0 / 0 |
| Panda | TRAC-IK 5ms | 97.907 | 97.907 | 0.263 | 0.230 | 0.475 | 0.646 | not comparable | 0 / 0 / 0 |
| UR5e | Always-hard | 100.000 | 100.000 | 1.527 | 1.342 | 2.021 | 3.668 | 3.462 | 0 / 0 / 66 |
| UR5e | CG-HIK | 100.000 | 100.000 | 1.975 | 1.793 | 2.479 | 4.020 | 3.464 | 0 / 0 / 66 |
| UR5e | Geometry rule | 100.000 | 100.000 | 1.971 | 1.788 | 2.486 | 4.474 | 3.468 | 0 / 0 / 66 |
| UR5e | P50-selection | 100.000 | 100.000 | 1.978 | 1.792 | 2.508 | 4.292 | 3.529 | 0 / 0 / 66 |
| UR5e | Reject-only + hard | 100.000 | 100.000 | 1.972 | 1.789 | 2.453 | 4.016 | 3.462 | 0 / 0 / 66 |
| UR5e | Routing-only | 100.000 | 100.000 | 1.975 | 1.794 | 2.464 | 4.236 | 3.464 | 0 / 0 / 66 |
| UR5e | TRAC-IK 100ms | 99.960 | 99.947 | 0.239 | 0.214 | 0.348 | 0.423 | not comparable | 0 / 0 / 0 |
| UR5e | TRAC-IK 20ms | 99.947 | 99.947 | 0.238 | 0.214 | 0.347 | 0.425 | not comparable | 0 / 0 / 0 |
| UR5e | TRAC-IK 400ms | 99.960 | 99.960 | 0.234 | 0.214 | 0.352 | 0.433 | not comparable | 0 / 0 / 0 |
| UR5e | TRAC-IK 5ms | 99.947 | 99.947 | 0.232 | 0.214 | 0.347 | 0.431 | not comparable | 0 / 0 / 0 |

## Constructed inexecutable points: 500 queries per robot

| Robot | Method | Verified % | Within 20 ms % | Mean ms | P50 | P95 | P99 | Mean FEV | Reject / defer / fallback calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda | Always-hard | 0.000 | 0.000 | 279.612 | 292.403 | 308.818 | 314.545 | 469.368 | 0 / 0 / 1500 |
| Panda | CG-HIK | 0.000 | 0.000 | 155.447 | 217.118 | 308.280 | 314.952 | 265.698 | 660 / 840 / 840 |
| Panda | Geometry rule | 0.000 | 0.000 | 155.414 | 216.889 | 308.103 | 315.194 | 265.698 | 660 / 840 / 840 |
| Panda | P50-selection | 0.000 | 0.000 | 155.384 | 216.771 | 308.146 | 314.482 | 265.698 | 660 / 840 / 840 |
| Panda | Reject-only + hard | 0.000 | 0.000 | 155.468 | 216.609 | 308.546 | 315.552 | 265.698 | 660 / 840 / 840 |
| Panda | Routing-only | 0.000 | 0.000 | 280.754 | 293.229 | 310.451 | 315.716 | 475.618 | 0 / 840 / 1500 |
| Panda | TRAC-IK 100ms | 0.000 | 0.000 | 100.139 | 100.133 | 100.193 | 100.235 | not comparable | 0 / 0 / 0 |
| Panda | TRAC-IK 20ms | 0.000 | 0.000 | 20.142 | 20.136 | 20.223 | 20.248 | not comparable | 0 / 0 / 0 |
| Panda | TRAC-IK 400ms | 0.000 | 0.000 | 400.143 | 400.134 | 400.196 | 400.243 | not comparable | 0 / 0 / 0 |
| Panda | TRAC-IK 5ms | 0.000 | 0.000 | 5.144 | 5.138 | 5.227 | 5.248 | not comparable | 0 / 0 / 0 |
| UR5e | Always-hard | 0.000 | 0.000 | 160.759 | 161.664 | 251.815 | 260.706 | 350.694 | 0 / 0 / 1500 |
| UR5e | CG-HIK | 0.000 | 0.000 | 144.969 | 151.916 | 252.742 | 262.970 | 319.994 | 192 / 1308 / 1308 |
| UR5e | Geometry rule | 0.000 | 0.000 | 145.044 | 152.488 | 252.849 | 263.322 | 319.994 | 192 / 1308 / 1308 |
| UR5e | P50-selection | 0.000 | 0.000 | 144.988 | 152.806 | 253.355 | 262.407 | 319.994 | 192 / 1308 / 1308 |
| UR5e | Reject-only + hard | 0.000 | 0.000 | 145.014 | 153.332 | 252.575 | 262.443 | 319.994 | 192 / 1308 / 1308 |
| UR5e | Routing-only | 0.000 | 0.000 | 162.130 | 163.090 | 252.966 | 262.896 | 359.132 | 0 / 1308 / 1500 |
| UR5e | TRAC-IK 100ms | 0.000 | 0.000 | 100.149 | 100.144 | 100.222 | 100.250 | not comparable | 0 / 0 / 0 |
| UR5e | TRAC-IK 20ms | 0.000 | 0.000 | 20.155 | 20.147 | 20.235 | 20.254 | not comparable | 0 / 0 / 0 |
| UR5e | TRAC-IK 400ms | 0.000 | 0.000 | 400.158 | 400.153 | 400.235 | 400.255 | not comparable | 0 / 0 / 0 |
| UR5e | TRAC-IK 5ms | 0.000 | 0.000 | 5.159 | 5.148 | 5.237 | 5.259 | not comparable | 0 / 0 / 0 |

## Paired feasible-point effects

| Robot | Comparison | Mean time ratio [95% CI] | FEV ratio [95% CI] | Success difference (pp) |
| --- | --- | --- | --- | --- |
| Panda | Routing-only / Always-hard | 1.0334 [1.0296, 1.0382] | 1.0012 [1.0006, 1.0019] | 0.000 [0.000, 0.000] |
| Panda | CG-HIK / Always-hard | 1.0333 [1.0293, 1.0382] | 1.0012 [1.0006, 1.0019] | 0.000 [0.000, 0.000] |
| Panda | CG-HIK / Reject-only + hard | 0.9998 [0.9983, 1.0013] | 1.0012 [1.0006, 1.0019] | 0.000 [0.000, 0.000] |
| Panda | CG-HIK / P50-selection | 0.9985 [0.9970, 0.9999] | 0.9942 [0.9925, 0.9958] | 0.000 [0.000, 0.000] |
| Panda | CG-HIK / Geometry rule | 1.0009 [0.9997, 1.0022] | 1.0005 [1.0001, 1.0011] | 0.000 [0.000, 0.000] |
| Panda | CG-HIK / Routing-only | 0.9999 [0.9987, 1.0013] | 1.0000 [1.0000, 1.0000] | 0.000 [0.000, 0.000] |
| Panda | CG-HIK / TRAC-IK 100ms | 59.3409 [52.2050, 66.4076] | not comparable | -2.960 [-3.747, -2.200] |
| Panda | CG-HIK / TRAC-IK 20ms | 59.1777 [52.0375, 66.2348] | not comparable | -2.907 [-3.680, -2.160] |
| Panda | CG-HIK / TRAC-IK 400ms | 59.1374 [52.0791, 66.2155] | not comparable | -3.013 [-3.773, -2.253] |
| Panda | CG-HIK / TRAC-IK 5ms | 59.3147 [52.2047, 66.3842] | not comparable | -2.987 [-3.747, -2.253] |
| UR5e | Routing-only / Always-hard | 1.2935 [1.2864, 1.3003] | 1.0005 [0.9995, 1.0014] | 0.000 [0.000, 0.000] |
| UR5e | CG-HIK / Always-hard | 1.2932 [1.2861, 1.3000] | 1.0005 [0.9995, 1.0014] | 0.000 [0.000, 0.000] |
| UR5e | CG-HIK / Reject-only + hard | 1.0016 [0.9979, 1.0053] | 1.0005 [0.9995, 1.0014] | 0.000 [0.000, 0.000] |
| UR5e | CG-HIK / P50-selection | 0.9983 [0.9942, 1.0022] | 0.9814 [0.9737, 0.9890] | 0.000 [0.000, 0.000] |
| UR5e | CG-HIK / Geometry rule | 1.0017 [0.9978, 1.0055] | 0.9986 [0.9961, 1.0002] | 0.000 [0.000, 0.000] |
| UR5e | CG-HIK / Routing-only | 0.9998 [0.9960, 1.0033] | 1.0000 [1.0000, 1.0000] | 0.000 [0.000, 0.000] |
| UR5e | CG-HIK / TRAC-IK 100ms | 8.2598 [7.7755, 8.5832] | not comparable | 0.040 [0.000, 0.120] |
| UR5e | CG-HIK / TRAC-IK 20ms | 8.2957 [7.8657, 8.5869] | not comparable | 0.053 [0.000, 0.160] |
| UR5e | CG-HIK / TRAC-IK 400ms | 8.4403 [8.2951, 8.5766] | not comparable | 0.040 [0.000, 0.120] |
| UR5e | CG-HIK / TRAC-IK 5ms | 8.4973 [8.3682, 8.6282] | not comparable | 0.053 [0.000, 0.160] |

## Feasible reference trajectories: 40 per robot

| Robot | Method | Complete / 40 | All frames within 20 ms | Frame verified % | Frame within 20 ms % | Total s | P50 / P95 / P99 ms | Mean FEV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda | Always-hard | 37 | 36 | 96.100 | 95.950 | 55.346 | 1.413 / 2.071 / 253.679 | 16.034 |
| Panda | CG-HIK | 34 | 34 | 94.700 | 94.617 | 39.003 | 1.877 / 2.377 / 177.755 | 11.719 |
| Panda | Routing-only | 34 | 34 | 94.700 | 94.617 | 72.091 | 1.879 / 72.835 / 284.994 | 19.659 |
| Panda | TRAC-IK 5ms | 36 | 36 | 94.667 | 94.667 | 3.040 | 0.234 / 5.067 / 5.168 | not comparable |
| UR5e | Always-hard | 36 | 36 | 95.750 | 95.750 | 15.950 | 1.336 / 2.038 / 36.702 | 6.323 |
| UR5e | CG-HIK | 36 | 36 | 95.750 | 95.750 | 17.021 | 1.777 / 2.277 / 34.678 | 5.730 |
| UR5e | Routing-only | 36 | 36 | 95.750 | 95.750 | 18.669 | 1.779 / 2.456 / 37.316 | 6.402 |
| UR5e | TRAC-IK 5ms | 39 | 39 | 98.950 | 98.950 | 1.699 | 0.219 / 0.345 / 5.067 | not comparable |

## Trajectory families: ten per robot and family

| Robot | Family | Method | Complete / 10 | Total s | Mean FEV |
| --- | --- | --- | --- | --- | --- |
| Panda | smooth | Always-hard | 9 | 31.998 | 32.758 |
| Panda | near_singular | Always-hard | 8 | 18.124 | 24.291 |
| Panda | joint_limit_return | Always-hard | 10 | 3.039 | 4.091 |
| Panda | high_curvature | Always-hard | 10 | 2.186 | 2.996 |
| Panda | smooth | CG-HIK | 9 | 5.195 | 5.445 |
| Panda | near_singular | CG-HIK | 5 | 28.070 | 35.387 |
| Panda | joint_limit_return | CG-HIK | 10 | 2.869 | 3.047 |
| Panda | high_curvature | CG-HIK | 10 | 2.869 | 2.996 |
| Panda | smooth | Routing-only | 9 | 32.072 | 31.119 |
| Panda | near_singular | Routing-only | 5 | 34.261 | 41.475 |
| Panda | joint_limit_return | Routing-only | 10 | 2.880 | 3.047 |
| Panda | high_curvature | Routing-only | 10 | 2.879 | 2.996 |
| Panda | smooth | TRAC-IK 5ms | 9 | 0.938 | not comparable |
| Panda | near_singular | TRAC-IK 5ms | 7 | 1.375 | not comparable |
| Panda | joint_limit_return | TRAC-IK 5ms | 10 | 0.362 | not comparable |
| Panda | high_curvature | TRAC-IK 5ms | 10 | 0.364 | not comparable |
| UR5e | smooth | Always-hard | 8 | 6.124 | 9.162 |
| UR5e | near_singular | Always-hard | 9 | 4.684 | 8.120 |
| UR5e | joint_limit_return | Always-hard | 9 | 3.086 | 5.009 |
| UR5e | high_curvature | Always-hard | 10 | 2.055 | 3.000 |
| UR5e | smooth | CG-HIK | 8 | 5.221 | 6.761 |
| UR5e | near_singular | CG-HIK | 9 | 5.340 | 8.127 |
| UR5e | joint_limit_return | CG-HIK | 9 | 3.747 | 5.031 |
| UR5e | high_curvature | CG-HIK | 10 | 2.713 | 3.000 |
| UR5e | smooth | Routing-only | 8 | 6.845 | 9.450 |
| UR5e | near_singular | Routing-only | 9 | 5.346 | 8.127 |
| UR5e | joint_limit_return | Routing-only | 9 | 3.764 | 5.031 |
| UR5e | high_curvature | Routing-only | 10 | 2.715 | 3.000 |
| UR5e | smooth | TRAC-IK 5ms | 10 | 0.362 | not comparable |
| UR5e | near_singular | TRAC-IK 5ms | 10 | 0.346 | not comparable |
| UR5e | joint_limit_return | TRAC-IK 5ms | 9 | 0.647 | not comparable |
| UR5e | high_curvature | TRAC-IK 5ms | 10 | 0.343 | not comparable |

## Reference-trajectory cost decomposition: CG-HIK vs hard

Outcome-conditioned groups describe where the measured difference resides; they are not randomized causal subgroups.

| Robot | Group | Trajectories | Hard s | CG-HIK s | Hard minus CG-HIK s | Hard FEV | CG-HIK FEV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Panda | both_complete | 34 | 8.227 | 9.727 | -1.500 | 16607 | 15108 |
| Panda | only_cghik_complete | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| Panda | only_hard_complete | 3 | 0.678 | 10.477 | -9.799 | 1310 | 17444 |
| Panda | neither_complete | 3 | 46.440 | 18.799 | 27.641 | 78288 | 37760 |
| UR5e | both_complete | 36 | 7.390 | 9.764 | -2.374 | 16042 | 16052 |
| UR5e | only_cghik_complete | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| UR5e | only_hard_complete | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| UR5e | neither_complete | 4 | 8.560 | 7.257 | 1.303 | 21895 | 18326 |

## Paired trajectory effects

| Robot | Comparison | Total time ratio [95% CI] | Completion difference (pp) [95% CI] | Lost / gained UIDs |
| --- | --- | --- | --- | --- |
| Panda | CG-HIK / Always-hard | 0.7047 [0.3147, 2.4703] | -7.5 [-15.0, 0.0] | 3 / 0 |
| Panda | Routing-only / Always-hard | 1.3026 [1.0161, 3.3297] | -7.5 [-15.0, 0.0] | 3 / 0 |
| Panda | CG-HIK / Routing-only | 0.5410 [0.2842, 0.9992] | 0.0 [0.0, 0.0] | 0 / 0 |
| Panda | CG-HIK / TRAC-IK 5ms | 12.8318 [6.4377, 21.9054] | -5.0 [-15.0, 7.5] | 4 / 2 |
| UR5e | CG-HIK / Always-hard | 1.0671 [0.9262, 1.2859] | 0.0 [0.0, 0.0] | 0 / 0 |
| UR5e | Routing-only / Always-hard | 1.1705 [1.1157, 1.2882] | 0.0 [0.0, 0.0] | 0 / 0 |
| UR5e | CG-HIK / Routing-only | 0.9117 [0.8073, 0.9991] | 0.0 [0.0, 0.0] | 0 / 0 |
| UR5e | CG-HIK / TRAC-IK 5ms | 10.0188 [7.0831, 14.8661] | -7.5 [-15.0, 0.0] | 3 / 0 |

## Supporting records and figures

Full completion UIDs, first-failure frames/reasons and trajectory cumulative mean/median/P95 appear in `trajectory_main_table.csv`, `trajectory_units.csv` and `trajectory_paired_comparisons.csv`. Accepted pose errors and joint steps are retained in the main tables and raw records. All point witnesses had zero learned false rejection. External nonnegative solver returns that fail the public velocity check remain failures; they are itemized in `external_verifier_rejections.csv`.

The first partial adapter run was discarded in full after a code-inspection timing correction; its raw file and seal remain in `diagnostic_adapter_attempt_01`. Neither its outcomes nor the complete supplemental outcomes were used to tune the method. See the measurement notes.

### Original trajectory savings decomposition

![Original trajectory savings decomposition](/home/eric/wjg/btry/outputs/revision_compute_allocation/reports/publication_figures/cost_sources_decomposition.png)

### Six internal strategies, feasible points

![Six internal strategies, feasible points](/home/eric/wjg/btry/outputs/revision_compute_allocation/reports/publication_figures/internal_strategy_mechanisms.png)

### External verified-success versus actual time

![External verified-success versus actual time](/home/eric/wjg/btry/outputs/revision_compute_allocation/reports/publication_figures/trac_ik_success_time.png)

### Reference-path completion and all-frame cost

![Reference-path completion and all-frame cost](/home/eric/wjg/btry/outputs/revision_compute_allocation/reports/publication_figures/feasible_reference_trajectories.png)
