# Release v4 candidate: independent validation-only audit

Audit date: 2026-08-30  
Audited candidate: `outputs/release_v4_candidate`  
Development data: Panda and UR5e, seed 17, disjoint risk-training,
calibration and policy-validation roles.

## Decision

**Candidate artifact integrity and numerical reproduction: PASS. No blocking
candidate corruption or split leakage was found. Scientific claim status:
CONDITIONAL. This audit does not authorize a formal test.**

The eleven sealed payload artifacts, their provenance chain, all 40,000 source
query identities, both checkpoints, Platt calibration, 99% ID OOD thresholds,
all 252 policy candidates per robot and both selected policies were
independently reconstructed. The selected-policy candidate JSON and every
reported route, hard gate, FEV and latency value reproduce with maximum
absolute error zero under the frozen 8/1 CPU thread setting.

Three limitations must remain explicit:

1. the perfect AUROC is an **UR5e semantic success/fail-all** result, not an OOD
   AUROC; no labeled OOD example was used or evaluated;
2. zero false reject holds only on the represented validation distribution,
   where rejected queries are exclusively `large_step` and `unreachable`; and
3. all 252 candidates pass and collapse to six effective route policies, so
   success and reject thresholds are saturated rather than empirically
   identified.

Machine-readable evidence is in
[`candidate_audit.json`](./candidate_audit.json). The independent audit can be
reproduced with:

```bash
/home/eric/anaconda3/envs/isaaclab_3/bin/python \
  scripts/audit_release_v4_candidate.py \
  --workspace /home/eric/wjg/btry \
  --candidate-root /home/eric/wjg/btry/outputs/release_v4_candidate \
  --json /home/eric/wjg/btry/docs/audits/release_v4_candidate/candidate_audit.json
```

The script does not import the v4 training runner or predictor. Its only write
is the audit JSON under `docs/audits`; it does not modify the candidate or any
source output.

## 1. Artifact and provenance integrity

The manifest declares exactly eleven payload files, and the directory contains
exactly those eleven plus the two control manifests. There are no missing,
extra or symbolic-link payloads.

| Payload | Bytes | SHA-256 | Result |
|---|---:|---|:---:|
| `data_audit.json` | 108,166 | `c7b5382e…51c2f` | PASS |
| `environment.json` | 482 | `9ab89c8c…1d6265` | PASS |
| `frozen_config.yaml` | 1,459 | `7bf74852…007e1` | PASS |
| `models/panda_seed17_predictor.pt` | 30,221 | `b2f7d8c0…48c39` | PASS |
| `models/ur5e_seed17_predictor.pt` | 30,139 | `80bef767…5f04c` | PASS |
| `policies/panda_seed17_policy.json` | 2,553 | `6230684c…87c2` | PASS |
| `policies/ur5e_seed17_policy.json` | 2,556 | `a4ad0503…3698` | PASS |
| `policy_candidates/panda_seed17.json` | 406,660 | `a8c84ed2…70ee` | PASS |
| `policy_candidates/ur5e_seed17.json` | 408,596 | `ff65fc37…75ff` | PASS |
| `policy_selection.json` | 4,167 | `c4107d13…5132` | PASS |
| `training_metrics.json` | 20,388 | `7097264e…2500` | PASS |

The following chain also reproduces:

- artifact-manifest SHA-256:
  `7efa4572c0885eb96da48656f3082d2ec424fb0da9eb3bd4d970612dbf95bf5b`;
- training-config SHA-256:
  `7bf74852da3e25ce7bf6f57f81faadc704b82a8c10c95188fa800043042007e1`;
- completed-bulk manifest SHA-256:
  `a813bf633b8fe984162d675e64649f7256569b16060f9232ca46eeb1e89937e1`;
- release digest:
  `b8b62e7e696ae1f153929b141653d4be21a0e26684cf74ef4bbe3f946e5066c7`.

The recorded source-tree digest
`9234d77c89fe4cb291aeb6b127d23ff7f2df486dfe51cdf7b4eba0b49b6a9b11`
matches the 64 tracked `src/confik/*.py` files at commit
`54ca71245d0e9f26c9c78c2fda90beb3a70fbf5c`. At audit time, three untracked
`src/confik/release_v4_locked/*.py` files from subsequent work made the live
filesystem tree differ from that sealed snapshot; tracked source files were
clean. This is a post-candidate workspace addition, not a candidate hash
failure.

The audit opened no formal-test query, label, metric or result artifact. All
candidate, policy, selection, bulk and chunk `test_data_loaded` declarations
are false, and all accepted input paths resolve to the three named development
roles. Source-tree hashing reads committed Python source blobs; it does not
read formal-test result data.

## 2. Development split and original-source lineage

| Robot | Risk training | Calibration | Policy validation | Pairwise query overlap | Exact feature-row overlap |
|---|---:|---:|---:|---:|---:|
| Panda | 15,000 | 2,500 | 2,500 | 0 | 0 |
| UR5e | 15,000 | 2,500 | 2,500 | 0 | 0 |

For every selected query, the audit independently followed:

```text
paper_v2_seed17 source row
  → bulk selection source_index
  → selection query SHA-256
  → contiguous chunk NPZ row
  → candidate train/calibration/policy role
```

All 40,000 source indices are in range and unique within their roles. Query
SHA-256 was recalculated from `previous_q`, target position, target rotation
and frozen `dt=0.02`; all 40,000 match. Category,
`expected_reachable` and `continuity_feasible` also match the original source
rows. Every source file and every candidate-listed chunk file matches its
recorded size and SHA-256.

Role use is therefore correctly separated:

- `risk_train_queries`: model fitting and OOD location/covariance fitting;
- `calibration_queries`: shared Platt calibration and the 99% ID OOD threshold;
- `policy_validation_queries`: the finite policy-grid selection only.

No policy-validation row enters model or Platt fitting, and no calibration or
policy row overlaps training.

## 3. Semantic success, shared head and Platt calibration

Across all six robot × role combinations, `easy`, `medium` and `hard` semantic
verified-success labels are exactly identical, as required by terminal robust
fallback invariance. The saved model contains one semantic-success logit and
broadcasts it exactly across the three actions. `verified_success_before_deadline`
is diagnostic only and is not a learned or Platt target.

Independent refitting recovers every serialized Platt parameter exactly:

| Robot / head | Slope | Intercept | Parameter max error |
|---|---:|---:|---:|
| Panda / shared success | 1.231156 | 0.106155 | 0 |
| Panda / fail-all | 1.232767 | −0.105985 | 0 |
| UR5e / shared success | 0.999378 | 0.035263 | 0 |
| UR5e / fail-all | 0.999399 | −0.034664 | 0 |

Calibration metrics also reproduce exactly:

| Robot / head | AUROC | AUPRC | ECE | Brier |
|---|---:|---:|---:|---:|
| Panda / shared success | 0.999499 | 0.999855 | 1.39×10⁻⁵ | 3.999×10⁻⁴ |
| Panda / fail-all | 0.999596 | 0.999111 | 7.21×10⁻⁶ | 3.998×10⁻⁴ |
| UR5e / shared success | 1.000000 | 1.000000 | 5.09×10⁻⁶ | 1.03×10⁻¹⁰ |
| UR5e / fail-all | 1.000000 | 1.000000 | 4.65×10⁻⁶ | 1.24×10⁻¹⁰ |

The success and fail-all targets are exact complements, so their discrimination
figures are not two independent pieces of evidence.

## 4. OOD threshold audit

The Mahalanobis mean, shrinkage covariance/precision, calibration scores and
`method="higher"` 99% quantile threshold reproduce exactly.

| Robot | OOD threshold | Calibration ID coverage | Policy ID defer rate | Labeled OOD examples |
|---|---:|---:|---:|---:|
| Panda | 25.105068 | 99.04% | 0.92% | 0 |
| UR5e | 50.111272 | 99.04% | 1.32% | 0 |

The 99.04% rather than exactly 99.00% coverage is the expected finite-sample
result of the higher empirical quantile: 2,476 of 2,500 calibration examples
lie at or below the threshold.

This validates an **ID coverage/defer threshold**, not OOD discrimination.
There is no OOD AUROC, AUPRC or recall evidence in this candidate because no
labeled shifted example was used. Any paper statement that this candidate has
“perfect OOD AUROC” would be false.

## 5. Policy-grid and selected-policy reproduction

The grid size is exactly:

```text
7 success thresholds × 6 reject thresholds × 6 tie margins = 252
```

All 252 candidates pass all three frozen hard gates for both robots. Every
candidate row and the independently selected row reproduce exactly. However,
the 252 configurations contain only six distinct route/outcome signatures:
within each tie margin, all 42 combinations of success and reject thresholds
produce the same routes and metrics.

Thus, policy validation identifies `latency_tie_margin_ms=0.15`. It does not
empirically distinguish `minimum_success_probability=0.95` from the six lower
grid values or `reject_probability=0.70` from the five higher values. The 0.95
value is reached by the deterministic conservative tie breaker; 0.70 is the
first otherwise-equivalent reject threshold in grid order. This is valid
deterministic locking, but not threshold-sensitivity evidence.

### Selected validation metrics

| Metric | Panda | UR5e |
|---|---:|---:|
| Queries | 2,500 | 2,500 |
| Selected verified success | 81.24% | 81.96% |
| Fixed verified success | 81.24% | 81.96% |
| Difference versus fixed | 0.00 pp | 0.00 pp |
| Mean FEV, selected | 7.0972 | 6.2016 |
| Mean FEV, fixed | 78.2204 | 44.6756 |
| All-query FEV reduction | 90.93% | 86.12% |
| FEV reduction on fixed-success rows | 31.14% | 42.10% |
| FEV reduction on operational-feasible rows | 28.89% | 42.10% |
| Observed latency P50 | 1.641 ms | 1.549 ms |
| Observed latency P95 | 2.271 ms | 2.362 ms |
| Observed latency P99 | 90.625 ms | 48.569 ms |
| Successful-command latency P95 | 2.264 ms | 2.332 ms |
| Successful-command deadline miss | 0.20% | 0.00% |
| Fixed-success false reject | 0 / 2,031 | 0 / 2,049 |
| Operational-feasible false reject | 0 / 2,035 | 0 / 2,049 |

Routes:

| Robot | Easy | Medium | Hard | Reject | Defer |
|---|---:|---:|---:|---:|---:|
| Panda | 572 | 9 | 1,454 | 442 | 23 |
| UR5e | 118 | 0 | 1,931 | 418 | 33 |

The reported latency scope is the five-repeat solver-plus-verifier label; it
excludes learned-gate overhead and is not a deployment end-to-end latency
claim. Rejected rows have zero solver FEV/latency in the policy accounting, so
the mixed all-query FEV reduction is partly driven by command reject. The
separately recomputed feasible/fixed-success reductions above remain positive
and are the more defensible efficiency evidence.

Both P99 values remain far above the nominal 20 ms deadline despite low P95;
the paper must report the tail rather than summarize this candidate as simply
“below 20 ms.”

## 6. Why perfect AUROC and zero false reject occur

The audit found no query-hash overlap, exact feature-row overlap or
post-solver label field in the feature function. `cached_risk_features` accepts
only the query and prepared seed diagnostics; it uses seed residual,
uncertainty, singular value, joint margin/step and current-to-target pose step,
all available before numerical refinement and verification.

The more likely explanation is strong dataset construction separability:

- In UR5e training, calibration and policy validation, every semantic fail-all
  query is either `large_step` or `unreachable`; every other category succeeds.
- `learned_seed_position_error` alone has AUROC 1.0 in all three roles. Its
  fail/success ranges do not overlap:

| UR5e role | Minimum fail-all seed error | Maximum success seed error |
|---|---:|---:|
| Risk training | 0.10120 | 0.08769 |
| Calibration | 0.13534 | 0.07952 |
| Policy validation | 0.12679 | 0.07631 |

- `current_pose_position_step` also separates perfectly in all three roles.
- Panda is almost, but not perfectly, separated: seed-position-error AUROC is
  0.9957/0.9998/0.9975 for train/calibration/policy validation because a small
  number of ID, near-singular or workspace-boundary queries fail.

This pattern is reproducible and consistent with legitimate pre-solver
features, so there is no evidence of direct label leakage or query
memorization. Scientifically, however, it is a **category/support shortcut**:
the model has learned an unusually clean boundary created by the development
query design. It does not establish generalization to novel feasible failures,
unseen OOD mechanisms or real closed-loop disturbances.

The zero-false-reject result follows from the same boundary. All rejected
policy-validation queries are operationally invalid under the dataset
contract:

| Robot | Rejected `large_step` | Rejected `unreachable` | Rejected other categories | Rejected fixed successes |
|---|---:|---:|---:|---:|
| Panda | 215 | 227 | 0 | 0 |
| UR5e | 218 | 200 | 0 | 0 |

It is valid evidence for the represented invalid-query guard. It is not valid
evidence for a broad high-confidence reject mechanism on difficult but
feasible queries. The candidate itself correctly keeps the broad reject claim
disabled.

## 7. Exit conditions

The candidate is internally reproducible and may proceed to a separate code
lock/readiness decision under these conditions:

1. preserve the exact candidate hashes and the commit-matched source snapshot;
2. describe the 0.95/0.70 thresholds as deterministic saturated selections,
   not uniquely optimized operating points;
3. keep OOD → defer and state that OOD AUROC/AUPRC remain unevaluated;
4. restrict reject claims to represented operational-invalid classes;
5. report feasible-stratified FEV and P99 latency alongside all-query means;
6. add learned-gate overhead before making any end-to-end latency claim; and
7. do not use a later formal result to revise this candidate, grid or gates.

This audit makes no formal-test authorization. It confirms only that the
validation-only candidate is sealed, reproducible and suitable to be handed to
the separately controlled release-lock process with the limitations above.
