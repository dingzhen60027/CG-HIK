# Main results

## Solver-native contract mapping

See `01_protocol/native_mappings.json` and the fixed protocol mapping table.

## Witness-feasible points

| Robot | Setting | Verified % | P50 / P95 / P99 ms | Mean ms |
|---|---|---:|---|---:|
| panda | Position 5 | 100.000 | 0.211 / 0.379 / 0.681 | 0.241 |
| panda | DLS strict | 99.480 | 1.038 / 8.944 / 9.230 | 3.015 |
| panda | Aligned 5 | 100.000 | 0.199 / 0.332 / 0.478 | 0.221 |
| panda | Strict 5 | 100.000 | 0.211 / 0.367 / 0.487 | 0.236 |
| panda | Aligned 20 | 100.000 | 0.199 / 0.331 / 0.467 | 0.220 |
| panda | Strict 20 | 100.000 | 0.211 / 0.362 / 0.485 | 0.236 |
| panda | Orientation 5 | 100.000 | 0.210 / 0.382 / 0.799 | 0.247 |
| panda | DLS aligned | 99.480 | 0.625 / 1.042 / 4.479 | 0.762 |
| ur5e | Position 5 | 100.000 | 0.199 / 0.317 / 0.414 | 0.216 |
| ur5e | DLS strict | 99.960 | 1.185 / 8.113 / 8.427 | 2.757 |
| ur5e | Aligned 5 | 100.000 | 0.191 / 0.312 / 0.415 | 0.209 |
| ur5e | Strict 5 | 100.000 | 0.201 / 0.318 / 0.409 | 0.218 |
| ur5e | Aligned 20 | 100.000 | 0.192 / 0.313 / 0.410 | 0.210 |
| ur5e | Strict 20 | 100.000 | 0.200 / 0.320 / 0.424 | 0.222 |
| ur5e | Orientation 5 | 99.987 | 0.200 / 0.324 / 0.445 | 0.219 |
| ur5e | DLS aligned | 99.960 | 0.578 / 0.926 / 1.458 | 0.656 |

## Complete trajectories

| Robot | Setting | Complete / 40 | P50 / P95 / P99 ms | Cumulative s |
|---|---|---:|---|---:|
| panda | DLS strict | 37 | 0.943 / 9.119 / 10.624 | 14.255 |
| panda | DLS aligned | 37 | 0.550 / 1.451 / 6.830 | 4.814 |
| panda | Strict 20 | 34/34/34 | 0.192 / 20.352 / 20.559 | 8.654 |
| panda | Strict 5 | 34/34/34 | 0.190 / 5.249 / 5.434 | 3.187 |
| panda | Aligned 20 | 37/37/38 | 0.180 / 0.462 / 20.392 | 3.600 |
| panda | Aligned 5 | 36/37/37 | 0.179 / 0.484 / 5.305 | 2.150 |
| ur5e | DLS strict | 35 | 0.816 / 7.813 / 7.955 | 16.965 |
| ur5e | DLS aligned | 28 | 0.495 / 5.066 / 7.769 | 6.586 |
| ur5e | Strict 20 | 37/37/37 | 0.169 / 0.303 / 20.357 | 4.932 |
| ur5e | Strict 5 | 37/37/37 | 0.169 / 0.300 / 5.256 | 2.032 |
| ur5e | Aligned 20 | 39/39/39 | 0.161 / 0.229 / 0.412 | 1.568 |
| ur5e | Aligned 5 | 39/39/39 | 0.161 / 0.223 / 0.405 | 1.167 |
