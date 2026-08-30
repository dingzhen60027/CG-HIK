# Counterfactual v4 bulk: independent completion audit

Audit date: 2026-08-30  
Audited output: `outputs/counterfactual_v4_bulk`  
Scope: Panda and UR5e, seed 17, disjoint training/calibration/policy-validation
roles, three executed decision actions and one semantic-alias audit action.

## Decision

**Artifact and split integrity: PASS. Development-training use: GO WITH
CONDITIONS. Broad command-reject claim: NO-GO. Formal v4 test: NOT
AUTHORIZED.**

No blocking corruption, incompleteness, hash failure, selection mismatch or
split leakage was found. The completed bulk is structurally usable for
development training and calibration. Two evidence limitations must remain
visible in every downstream model and paper claim:

1. the frozen minimum-support gate for a broad command-reject claim fails on
   policy validation for both robots; and
2. `easy`, `medium` and `hard` have zero semantic verified-success
   disagreement, so these labels contain counterfactual cost signal but no
   action-specific semantic-success signal.

The machine-readable evidence is in
[`bulk_audit.json`](./bulk_audit.json). The independent audit can be reproduced
with:

```bash
/home/eric/anaconda3/envs/isaaclab_3/bin/python \
  scripts/audit_counterfactual_v4_bulk.py \
  --workspace /home/eric/wjg/btry \
  --bulk-root /home/eric/wjg/btry/outputs/counterfactual_v4_bulk \
  --require-complete \
  --json /home/eric/wjg/btry/docs/audits/counterfactual_v4_bulk/bulk_audit.json
```

The audit does not import the bulk runner and does not write below `outputs/`.

## 1. Dataset and grain

The intended grain is one row per `(robot, source role, query, collected
action)`, with five raw timing repeats attached to every row. `fixed_robust` is
an exact semantic alias of `easy`; it is stored as a fourth query/action row
but was not executed again.

| Contract | Expected | Observed | Result |
|---|---:|---:|:---:|
| Unique development queries | 40,000 | 40,000 | PASS |
| Query/action rows | 160,000 | 160,000 | PASS |
| Completed chunks | 160 | 160 | PASS |
| Real timed executions, three actions × five repeats | 600,000 | 600,000 | PASS |
| `fixed_robust` alias rows | 40,000 | 40,000 | PASS |
| Aliased repeat samples | 200,000 | 200,000 | PASS |
| Actual `fixed_robust` solver executions | 0 | 0 | PASS |

Per robot, the bulk contains 15,000 risk-training, 2,500 calibration and 2,500
policy-validation queries. The risk-training category quota was reproduced
exactly for each robot: 3,500 `id`; 2,500 each of `hard_valid`,
`near_singular`, `near_limit` and `workspace_boundary`; and 750 each of
`large_step` and `unreachable`.

## 2. Checks performed

The audit independently verified:

- exact top-level, chunk-manifest, raw JSONL and consolidated NPZ schemas;
- every artifact size and SHA-256 digest, plus each canonical chunk-payload
  digest;
- exact replay of the frozen seed-derived selection, query SHA-256 identity,
  source index, category and reachability/continuity metadata;
- raw-record ↔ chunk NPZ ↔ consolidated NPZ ↔ source-dataset equality;
- unique query/action keys, four actions per query and contiguous chunk ranges;
- all five timing samples, timing decomposition, stored P50/P95, function
  evaluations, fallback state, commands and failure reasons;
- exact `fixed_robust = easy` semantic aliasing, including the copied timing
  samples and the absence of a fourth solver execution;
- summary and run-manifest reproduction; no failed or incomplete chunk
  directories;
- frozen code, release, source-dataset and pilot-alias evidence hashes; and
- the protected-tree before/after snapshot.

All six robot × source-role combinations pass with zero chunk issue counts.
The frozen provenance covers 64 code files, 54 release files, six source
datasets and three alias-evidence files per robot; all match their recorded
digests.

## 3. Split and test discipline

Within each robot, all pairwise intersections among risk training,
calibration and policy validation are zero. Every bulk role also has zero query
overlap with the earlier validation pilot.

The audit used no `test_v3` performance value. For formal-test paths it called
filesystem metadata operations only: 56 file entries were stat'ed and zero
test-content files were opened. The protected baseline digest matches, and the
current protected snapshot matches the recorded post-collection snapshot over
312 protected output files plus the recorded `czy` tree.

This establishes the auditable development/test boundary; it is not a claim
about reads that the operating system did not record. No evidence of test
selection or protected-result mutation was found.

## 4. Action semantics and training consequence

`fixed_robust` is an exact audit alias of `easy` across the completed bulk. The
three actual decision actions, however, have **zero semantic-success
disagreement** in every robot × role combination:

| Robot | Risk training | Calibration | Policy validation |
|---|---:|---:|---:|
| Panda | 0 | 0 | 0 |
| UR5e | 0 | 0 | 0 |

This is expected under the frozen cascade: every entry action retains access
to the same terminal robust stage. The data therefore support learning
action-specific latency/FEV/fallback costs. They do not support a claim that
three action-specific semantic-success heads discriminate different terminal
success outcomes. Deadline disagreements exist but are sparse—Panda 8/0/1 and
UR5e 4/0/1 for training/calibration/policy validation—and must not be relabeled
as semantic success differences.

The scientifically faithful implementation is a per-robot compact model with
a shared semantic feasibility/fail-all target and action-specific latency
heads. Three success outputs may be retained for interface compatibility, but
their structural identity must be disclosed and they must use
`verified_success`, not `verified_success_before_deadline`.

## 5. Semantic versus deadline fail-all support

The two labels remain distinct:

```text
semantic_fail_all = not any(verified_success[easy, medium, hard])
deadline_fail_all = not any(verified_success_before_deadline[easy, medium, hard])
```

| Robot / role | Semantic fail-all | Deadline fail-all | Deadline-only | Contract-feasible semantic fail-all |
|---|---:|---:|---:|---:|
| Panda / risk training | 1,522 | 1,569 | 47 | 22 |
| Panda / calibration | 502 | 510 | 8 | 1 |
| Panda / policy validation | 469 | 474 | 5 | 4 |
| UR5e / risk training | 1,500 | 1,500 | 0 | 0 |
| UR5e / calibration | 465 | 465 | 0 | 0 |
| UR5e / policy validation | 451 | 451 | 0 | 0 |

The preregistered minimum count of 30 contract-feasible semantic fail-all
queries is not met in either robot's policy-validation role: Panda has 4 and
UR5e has 0. Even pooling all development roles would yield only 27 for Panda
and 0 for UR5e, and pooling roles is not permitted to rescue a failed
policy-validation gate.

Consequently, this bulk cannot support a broad “high-confidence feasible
command reject” claim. Reject evidence must be restricted to the represented
operationally invalid classes such as `unreachable` and continuity-infeasible
`large_step`, or new preregistered training/calibration/policy-validation data
must be collected before that claim is reconsidered. OOD or uncertainty must
still route to defer, never to reject.

## 6. Environment contamination and recollection

All retained chunks are marked clean, all recorded contaminated attempts were
discarded and recollected, `post_chunk_busy_processes` is empty, and no
contaminated retained chunk was found.

| Robot / role | Recollected queries | Quiet-wait events | Quiet-wait time |
|---|---:|---:|---:|
| Panda / risk training | 7 | 8 | 20.330 s |
| Panda / calibration | 1 | 1 | 3.052 s |
| Panda / policy validation | 3 | 3 | 8.126 s |
| UR5e / risk training | 64 | 67 | 1,722.762 s |
| UR5e / calibration | 1 | 1 | 7.098 s |
| UR5e / policy validation | 0 | 0 | 0.000 s |

The retained dataset is not corrupted by these attempts, but the UR5e
risk-training host contention is operationally notable: 64 queries were
recollected and quiet waiting consumed 28.71 minutes. This should be reported
as a measurement-control caveat, and the retry audit trail should remain with
the release. It is not evidence for deleting or reweighting retained rows.

## 7. Direct downstream label contract

Train **separate Panda and UR5e models** with the same architecture and label
definitions. Do not pool them under an ambiguous “shared gate.” For query `i`,
decision action `a ∈ {easy, medium, hard}` and repeat `r ∈ {1,…,5}`:

```text
x_i                  = the nine stored risk features
y_verified[i,a]      = verified_success[i,a]
y_deadline[i,a]      = verified_success_before_deadline[i,a]
y_latency[i,a,r]     = latency_samples_ns[i,a,r] / 1e6
y_semantic_fail[i]   = not any_a y_verified[i,a]
y_deadline_fail[i]   = not any_a y_deadline[i,a]
```

Rules:

1. Use only `risk_train_queries` for fitting, `calibration_queries` for
   calibration and `policy_validation_queries` for development selection and
   gates.
2. Fit action P50/P95 with quantile loss on all five raw repeats grouped by
   query; do not convert the noisy empirical-P95 winner into a hard class.
3. Keep semantic fail-all, deadline fail-all and OOD/defer conceptually and
   operationally separate.
4. `fixed_robust` is an audit/reference alias, not a fourth learned action.
5. `category`, `expected_reachable` and `continuity_feasible` are audit
   metadata and must not leak into deployment features.
6. Deterministic numerical solving and verification remain the only command
   acceptance authority; the learned gate allocates resources only.

## 8. Exit status and next allowed step

The data-quality exit decision is **GO WITH CONDITIONS for development
training**. Training may begin under the contract above, and policy validation
may be used for architecture/risk calibration and the frozen development
gates. The next review must explicitly check that the implementation did not
replace semantic labels with deadline labels and did not overstate the zero
action-success disagreement.

This audit does **not** freeze a v4 release, authorize a new formal test, permit
threshold changes based on any formal result, or make the broad reject claim
eligible. Those require a separately frozen model/configuration and a fresh
authorization after development gates are evaluated without test access.
