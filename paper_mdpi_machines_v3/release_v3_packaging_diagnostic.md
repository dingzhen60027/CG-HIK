# `release_v3_locked` packaging diagnostic

Diagnostic date: 22 August 2026

## Protocol outcome

The first formal packaging attempt stopped at `ur5e/seed43`, as required by the
locked protocol. Five earlier robot/seed combinations had passed. The incomplete
package remains at:

`outputs/.release_v3_locked.incomplete.71735`

No formal `outputs/release_v3_locked/` directory was published, and no
`test_v3` dataset or process was started.

## Validation-only reproduction

The failing combination was re-exported to a temporary directory and checked on
the same validation-only release subsets. The temporary directory was not a
formal output and was removed automatically after the diagnostic.

| Check | Result | Locked requirement | Gate |
|---|---:|---:|---|
| Seed output max absolute error | `0.0` | `<= 1e-6` | Pass |
| Risk probability max absolute error | `1.2598255771933964e-12` | `<= 1e-12` | **Fail** |
| Risk score max absolute error | `1.144973005295924e-12` | `<= 1e-12` | **Fail** |
| Stored-feature route agreement | `1.0` | `1.0` | Pass |
| Risk-metric max absolute delta | `7.28583859910259e-17` | `<= 1e-12` | Pass |
| Accepted joint-command max error | `0.0` | `<= 1e-6` | Pass |

Across 3700 paired runtime records, the following were all exactly `1.0`:

- accepted agreement;
- all, point, and trajectory route-action agreement;
- all, point, and trajectory FEV agreement;
- fallback agreement;
- executed-stage agreement;
- verification-reason agreement;
- query-hash agreement.

Runtime risk probability and score maximum differences were much smaller
(`4.773959005888173e-15` and `4.551914400963142e-15`). All baseline and proposed
success, rejection, mean FEV, trajectory completion, and command-spike deltas
were zero.

## Interpretation and action boundary

This is a strict offline frozen-risk export-equivalence failure, not evidence of
a routing or solver-behaviour difference on the sampled runtime queries. The
locked tolerance nevertheless failed, so Phase A is incomplete. Per protocol,
the tolerance was not relaxed, the backend remained `torchscript_exact`, no
algorithm or solver parameter was changed, and Phase B was not started.
