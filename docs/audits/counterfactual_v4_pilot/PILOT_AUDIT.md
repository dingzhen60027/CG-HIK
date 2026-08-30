# Counterfactual v4 validation pilot: independent exit audit

Audit date: 2026-08-30  
Audited output: `outputs/counterfactual_v4_pilot`  
Scope: Panda and UR5e, frozen training seed 17, 2,000
`risk_validation_queries` per robot, four collected actions, five timing repeats.

## Decision

**Artifact integrity: PASS. Bulk exit: GO WITH CONDITIONS. Formal-claim status:
NO-GO.**

The pilot is internally consistent and is adequate to justify a reduced,
validation-stage seed-17 two-robot collection with one model per robot. It does
**not** justify treating
the five-repeat empirical-P95 winner as a classification label, claiming that
the action-success heads discriminate among actions, or starting a fresh formal
v4 test.

The machine-readable evidence is in
[`pilot_audit.json`](./pilot_audit.json). It can be regenerated without writing
under `outputs`:

```bash
/home/eric/anaconda3/envs/isaaclab_3/bin/python \
  scripts/audit_counterfactual_v4_pilot.py
```

## 1. Evidence and split discipline

- Both selections come only from `risk_validation_queries.npz`; each source has
  15,000 queries and each pilot chooses 2,000 unique indices without
  replacement.
- The run and selection manifests record `test_data_loaded=false` and
  `test_v3_used_for_parameter_selection=false`.
- The protected v2/v3 output snapshot is identical before and after collection.
- This audit reads no formal `test_v3` metric and uses no test result for its
  exit decision or scale recommendation.
- The output is explicitly `pilot_only=true` and
  `eligible_for_formal_claims=false`.

This is a provenance audit of the recorded source paths and code contract; it
cannot prove every operating-system read. Within the auditable evidence, no
test-query use was found.

## 2. Integrity results

| Check | Panda | UR5e |
|---|---:|---:|
| Query/action records | 8,000 | 8,000 |
| Unique queries | 2,000 | 2,000 |
| Duplicate `(query_index, action)` keys | 0 | 0 |
| Missing or extra actions per query | 0 | 0 |
| Record-schema mismatches | 0 | 0 |
| Raw/NPZ/source mismatches | 0 | 0 |
| Query-hash duplicates | 0 | 0 |
| Generated-file hash/size mismatches | 0 | 0 |
| Frozen-release artifact mismatches | 0 | 0 |
| Timing decomposition/percentile mismatches | 0 | 0 |
| Pilot-summary reproduction | PASS | PASS |

The selected category proportions remain close to their complete source split:
the maximum absolute share difference is 0.85 percentage points for Panda and
0.60 percentage points for UR5e.

One provenance improvement is required for bulk collection: the run manifest
hashes the runner but not the imported collector explicitly. The collector is
recoverable from the clean recorded collection commit, but the bulk manifest
should hash the collector and all transitive runtime/config dependencies
directly.

## 3. Action semantics

`fixed_robust` is a valid audit duplicate of `easy`: across all 4,000 queries,
acceptance, executed stages, command, FEV, iterations, fallback state, failure
reason and dynamic diagnostics match exactly. UR5e has one different
before-deadline label because a single fixed-arm timing repeat crossed 20 ms;
this is timing noise, not a semantic mismatch.

The three decision actions have **zero semantic-success disagreement** on both
robots. This follows from the current portfolio:

- `easy` enters easy, then escalates through medium and hard on failure;
- `medium` enters medium, then escalates to hard;
- `hard` directly enters the frozen robust hard stage.

All entries retain access to the same terminal robust solver. The pilot
therefore supplies useful counterfactual **cost** labels, but no evidence that
three separate semantic-success heads learn different action outcomes. If the
paper requires action-specific success discrimination, the action contract must
be revised and re-piloted. If the contract stays fixed, use a shared semantic
feasibility/fail-all head and action-specific latency heads, and report the
success-disagreement null honestly.

## 4. Five-repeat timing quality and winner stability

| Diagnostic | Panda | UR5e |
|---|---:|---:|
| Median per-query latency CV, action range | 5.8–6.2% | 6.4–7.0% |
| P95 per-query latency CV, action range | 18.3–19.1% | 19.9–20.7% |
| Mean repeat winner agreement with full-P95 winner | 55.6% | 60.5% |
| All five repeat winners identical | 20.4% | 27.4% |
| P50/P95 winner agreement | 58.9% | 63.9% |
| Full winner supported by at least 4/5 leave-one-out fits | 86.9% | 91.3% |
| Median winning margin | 0.180 ms | 0.238 ms |

With NumPy's linear quantile convention, five samples make empirical P95 equal
to `0.8 × maximum + 0.2 × second-largest`. It is not a well-resolved tail
estimate. The observed winner has some real category signal—`hard` wins 296/313
Panda and 284/310 UR5e `hard_valid` queries—but many easy, near-limit,
near-singular and workspace-boundary winners fluctuate at sub-millisecond
margins.

**Training consequence:** retain all five raw latency observations and apply
P50/P95 pinball loss to the observations, grouped by query. Do not train a hard
`argmin(empirical P95)` winner classifier and do not discard the non-winning
action samples.

The randomized Latin rotation is well balanced. The first four repeats place
every action once at every within-query order position; the fifth repeats the
random base order. Across the complete pilot, the largest position-count
deviation from perfect balance is 1.36% for Panda and 1.40% for UR5e.

## 5. Deadline and fail-all labels

Deadline labels are much more stable than fine-grained winner labels:

- No Panda decision action has timing repeats on both sides of 20 ms.
- No UR5e decision action straddles 20 ms.
- One UR5e `fixed_robust` audit record has one of five samples above 20 ms.

The stored summary's `fail_all` is actually “no decision action achieves
verified success before P95 20 ms.” It must not be used directly as the
high-confidence **command reject** target:

| Decomposition | Panda | UR5e |
|---|---:|---:|
| Stored no-action-before-deadline | 385 (19.25%) | 357 (17.85%) |
| All actions semantically fail verification | 377 | 357 |
| Semantically feasible but all miss the deadline | 8 | 0 |
| Contract-feasible stored fail-all | 10 | 0 |
| Fraction from `large_step + unreachable` | 97.40% | 100.00% |

Two distinct targets are required:

```text
semantic_fail_all = not any(verified_success[action])
deadline_fail_all = not any(verified_success_before_deadline[action])
```

Only high-confidence, in-distribution `semantic_fail_all` may support command
reject. A deadline-only failure is still potentially executable and must not be
renamed “unreachable”; it should defer/escalate or be reported as a deadline
miss according to the frozen runtime policy.

The random pilot contains too few difficult, contract-feasible fail-all cases
for a strong command-reject story. Bulk selection must prespecify enrichment of
ambiguous boundary/feasible-failure queries, or the final paper must limit the
reject claim to the represented unreachable and continuity-infeasible cases.

## 6. Environment resampling

- Panda discarded and recollected five contaminated queries; quiet-host waiting
  consumed 86.13 s, or 4.53% of recorded loop time.
- UR5e performed five contaminated recollections across four unique queries;
  query 1400 was contaminated twice. Waiting consumed 97.25 s, or 13.04% of
  loop time.
- Retained records contain only the final clean measurements. The manifest
  reports retries and wait events, but does not retain hashes or timings of the
  discarded attempts. Bulk collection should record discarded query index,
  detection time, process snapshot and retry count in a separate audit log.

## 7. Direct training-label contract

For query `i`, decision action `a ∈ {easy, medium, hard}` and repeat
`r ∈ {1,…,5}`:

```text
x_i                 = nine risk_features for the corresponding robot model
y_verified[i,a]     = verified_success
y_deadline[i,a]     = verified_success_before_deadline
y_latency[i,a,r]    = latency_samples_ns / 1e6
y_semantic_fail[i]  = not any_a y_verified[i,a]
y_deadline_fail[i]  = not any_a y_deadline[i,a]
```

Contract rules:

1. `fixed_robust` remains an audit arm; it is not a fourth learned decision.
2. Fit latency P50/P95 with pinball loss on `y_latency` raw repeats, using query
   groups for sampling and validation.
3. Keep `y_semantic_fail` and `y_deadline_fail` separate. Command reject may use
   only calibrated, ID `y_semantic_fail`; deadline failure is a compute/defer
   outcome.
4. The current `y_verified[:, easy/medium/hard]` labels are identical. A compact
   implementation may retain three outputs for interface compatibility, but
   scientific claims must treat them as a structural null unless a revised
   action contract produces real disagreement on a new training/validation
   pilot.
5. `category`, `expected_reachable` and `continuity_feasible` are audit metadata,
   not runtime features. They must not leak into the gate unless explicitly
   justified as available deployment inputs.
6. The accepted command and verifier remain downstream deterministic outputs;
   the network never certifies or directly accepts a command.

At policy validation, select the lowest predicted P95 action among actions that
meet the frozen calibrated risk constraint. If none qualifies, use semantic
fail-all confidence only for ID command reject; otherwise defer to the full
fixed robust cascade. The later OOD detector must always map uncertainty/OOD to
defer, never directly to reject.

## 8. Recommended seed-17 two-model scale

Use the same compact architecture and label contract, but fit **independent
Panda and UR5e models**. Interpret `30k/5k/5k` as the combined two-robot
evidence budget:

| Role | Total | Panda | UR5e | Source split |
|---|---:|---:|---:|---|
| Training | 30,000 | 15,000 | 15,000 | `risk_train_queries` |
| Calibration | 5,000 | 2,500 | 2,500 | `calibration_queries` |
| Policy validation | 5,000 | 2,500 | 2,500 | `policy_validation_queries` |
| Total | 40,000 | 20,000 | 20,000 | test excluded |

The formal bulk runner executes only the three decision actions. Because the
pilot proves `fixed_robust` is a semantic alias of `easy`, it records the audit
arm as an alias instead of running the solver again. The 40,000-query plan
therefore contains:

- 600,000 timed solver executions: 40,000 queries × 3 actions × 5 repeats;
- 40,000 fixed-robust query/action alias rows containing 200,000 aliased repeat
  samples;
- no additional fixed-robust solver executions.

Subtracting the observed fixed-arm runtime from the pilot wall time gives a
planned sequential projection of approximately 5.72 h. Scaling the original
four-executed-action pilot unchanged would be a conservative 7.35 h upper
bound. Expected compressed raw-record plus NPZ storage is about 56 MiB.
Interpreting 30k/5k/5k **per robot** doubles the work to 80,000 queries,
approximately 11.45 h under aliasing (14.71 h under four actual actions) and
112 MiB; the pilot does not justify that larger first-stage budget.

Because the weights are not pooled, no robot indicator is required. The two
models must use the same frozen architecture and label definitions, while
calibration and quantile coverage are fitted and reported separately. Training
seed 17 is a dataset/model construction seed, not an independent experimental
replicate.

## 9. Conditions attached to GO

Before treating the 40k collection as a frozen v4 development dataset:

1. consume raw repeats with quantile loss rather than hard P95 winners;
2. separate semantic fail-all from deadline fail-all;
3. either simplify to shared feasibility plus action-latency heads or re-pilot a
   genuinely success-disagreeing action contract;
4. preregister ambiguous feasible-failure enrichment and its minimum support;
5. keep the three source roles disjoint and exclude every formal test artifact;
6. hash the collector and transitive dependencies and retain a retry audit log;
7. freeze the common architecture and the two independent robot-specific model
   pipelines before model comparison.

If any of these conditions is rejected, the correct decision is **NO-GO for the
full v4 story**. The existing pilot remains valid only as a cost-label and
instrumentation diagnostic.
