# Reviewer 1 technical assessment

## Review setup

- **Input scope**  The assessment is limited to `main.tex`, `generated/evidence_snapshot.json`, `claim_evidence_map.md`, and `release_v3_packaging_diagnostic.md`.
- **Assessment boundary**  The six `paper_v2` runs are the only formal tests. The `latency_pilot_v3` result is validation-only. The optimized `test_v3` is absent and no unreported experiment is assumed.
- **Emphasis**  Technical soundness and technical failings. The five Nature-style axes are used only as rigor checks for an engineering-journal submission.
- **Central claim as assessed**  A history-conditioned learned gate can choose an entry action for a shared, verified IK cascade, thereby preserving feasible-query success while reducing numerical solver evaluations and avoiding work on constructed rejectable queries. The current evidence does not establish a positive optimized end-to-end feasible-latency result.

## Overall assessment

This is an unusually transparent pre-submission draft. The controlled proposed-versus-fixed comparison is well conceived because both arms share candidate generation, solver stages, fallback, and the verifier, with only the entry action changing. The paper also reports the failed formal feasible-latency gate and does not silently substitute the selected validation result for a fresh test. Within the exact-URDF benchmark, the formal evidence supports preserved verified success and lower numerical solver work.

The manuscript nevertheless requires major revision before submission. Most importantly, the positive export-latency narrative is not release-qualified. The locked packaging procedure stopped on an equivalence tolerance failure, no formal release directory was produced, and `test_v3` was not started. This conflicts with the statement that the implementation is ready for release testing. The paper can close this issue either by completing the locked release and independent test or by making the formal negative latency result the only submission-level runtime conclusion and treating the export study strictly as non-confirmatory development evidence. Additional major issues concern the absence of reported results for four declared comparators, incomplete evidence for the four-action calibration mechanism, possible generator-specific rejection performance, incomplete definition of the primary FEV outcome, and limited support for trajectory retention.

**Recommendation posture**  Major revision. The bounded algorithmic allocation result is promising, but the present package is not ready to support a positive optimized-runtime claim.

## Who would be interested in the results, and why

The work should interest robotics researchers who build online IK and motion-generation stacks, real-time systems researchers studying learned computation control, and safe-learning researchers interested in keeping learned proposals subordinate to an explicit verifier. The shared-cascade experiment is particularly useful because it separates learned routing from changes in the underlying numerical portfolio. Interest beyond these areas will depend on demonstrating that the resource-allocation effect survives more realistic operating conditions or articulating the result as a general design principle rather than a two-robot performance claim.

## Major strengths

- The primary comparison controls the solver portfolio and verifier, which makes the formal FEV difference attributable to entry routing within that fixed portfolio (`main.tex`, lines 302--342 and 666--671).
- Training, model selection, calibration, policy selection, risk testing, and runtime testing are assigned distinct roles (`main.tex`, lines 280--300 and 390--414).
- The point-query and complete-path units are distinguished, and the three training seeds are correctly described as sensitivity runs on a shared test set rather than independent test datasets (`main.tex`, lines 451--461 and 731--737).
- The formal negative latency result is stated directly. The authors correctly distinguish FEV from end-to-end latency (`main.tex`, lines 601--615 and 681--695).
- Null ablations and the limits of kinematic verification are retained rather than converted into unsupported mechanism or safety claims (`main.tex`, lines 568--587 and 713--729).

## Major Concerns

### R1-M1 [technical soundness, reproducibility]

**Severity**  Major  
**Blocking**  Yes  
**Claim pointer**  The validation-only exact export is presented as a numerically equivalent low-overhead implementation that met pilot latency gates and is ready for release testing.  
**Evidence pointer**  `main.tex`, lines 50--57, 353--376, 617--638, 731--737, and 754--758; `generated/evidence_snapshot.json`, lines 58--96; `release_v3_packaging_diagnostic.md`, lines 5--14 and 22--52.  
**Concern**  The selected backend passed the seed-17 validation pilot for both robots, but the subsequent locked packaging protocol stopped at UR5e seed 43. Frozen risk-probability and risk-score errors exceeded the preregistered `1e-12` limits, no formal release directory was published, and `test_v3` was not started. Although the diagnostic reports no observed routing or solver-behaviour difference in 3700 sampled records, the locked equivalence requirement still failed. The sentence that the implementation is "ready for release testing" is therefore not supported by the release state. More broadly, selecting and reporting a backend on validation cannot establish a positive feasible-latency result.  
**Why it matters**  End-to-end latency is the engineering quantity that contradicted the FEV proxy in every formal run. A positive optimized-runtime implication cannot rest on the same validation evidence used to select the implementation, especially when the preregistered release gate is incomplete. This blocks a submission-level claim that the optimized method improves feasible-query latency.  
**Resolution test**  Satisfy the unchanged locked packaging criteria for all six robot and seed artifacts, publish the sealed release manifest, and evaluate a fresh one-shot `test_v3` without threshold, backend, solver, or claim-gate changes. Alternatively, submit a `paper_v2`-bounded paper whose runtime conclusion is the formal negative result, remove readiness language and any implication that the pilot confirms deployable speed, and label all pilot values consistently as development evidence.

### R1-M2 [experimental design, significance]

**Severity**  Major  
**Blocking**  No  
**Claim pointer**  The benchmark contains six main methods and is intended to establish the value of learned action routing relative to conventional IK strategies and a matched threshold guard.  
**Evidence pointer**  `main.tex`, lines 125--160, 427--441, 487--599, and 707--711.  
**Concern**  Four additional primary comparators are declared, but the visible Results provide no complete quantitative results for previous-state DLS, learned-seed DLS, previous-state TRF, or the threshold guard. The threshold guard is discussed only through an approximately 0.9 percentage-point success loss. The reported controlled fixed-cascade comparison isolates the within-portfolio routing effect, but it does not show whether the proposed system is competitive against the declared conventional alternatives under matched accuracy, rejection, trajectory, and latency constraints.  
**Why it matters**  Internal attribution and practical comparative value are different questions. The current evidence supports the first but leaves the second largely unevaluable, which weakens both engineering significance and the field-relative case for the method.  
**Resolution test**  Report all declared main comparators with the same per-robot outcomes, acceptance contract, feasible and rejectable strata, trajectory unit, and latency protocol used for the primary pair. Include uncertainty or explicitly descriptive per-run results as appropriate. If those comparisons were not prespecified for inferential use, state that boundary and narrow comparative claims accordingly.

### R1-M3 [mechanism evidence, claim moderation]

**Severity**  Major  
**Blocking**  No  
**Claim pointer**  A calibrated four-action risk model allocates easy, medium, hard, or reject effort, and the calibrated history-conditioned gate is the learned mechanism behind the allocation effect.  
**Evidence pointer**  `main.tex`, lines 261--300, 471--485, 568--587, and 697--711; `claim_evidence_map.md`, lines 5 and 11.  
**Concern**  The held-out evidence emphasizes reject AUROC, reject ECE, reject recall, and false rejection. The manuscript reports non-reject macro-F1 of only about 0.59 to 0.64, but it does not show per-action prevalence, confusion, calibration, selected thresholds, routing frequencies, oracle-stage regret, or FEV conditional on the chosen action. Moreover, removing history, ensemble membership, uncertainty features, or calibration produced mostly null operational ablations. Aggregate FEV reduction establishes that the frozen routing policy changes work, but the reported analysis does not establish which parts of the proposed four-action calibrated architecture are responsible for feasible-query savings.  
**Why it matters**  The principal technical novelty is not merely a reject detector. It is a calibrated portfolio-action policy. Without action-resolved evidence, readers cannot judge whether easy, medium, and hard routing behaves as intended, whether probability calibration contributes operationally, or whether a materially simpler policy would yield the same result.  
**Resolution test**  Provide held-out, per-action support and reliability, threshold values, predicted versus oracle action tables, routing proportions, false-skip or escalation rates, and outcome or FEV breakdowns by predicted and oracle action. Compare these with the threshold guard and relevant ablations. If the action-resolved results remain nondiscriminating, moderate the mechanism claim to a frozen policy-level effect and avoid attributing the outcome specifically to calibration, uncertainty, or history.

### R1-M4 [experimental design, generalizability]

**Severity**  Major  
**Blocking**  No  
**Claim pointer**  Explicit rejection identifies futile requests and removes unnecessary numerical work without rejecting feasible commands.  
**Evidence pointer**  `main.tex`, lines 263--268, 416--425, 538--553, 673--679, and 722--729; `claim_evidence_map.md`, lines 6 and 9.  
**Concern**  Unreachable and deliberately discontinuous samples are assigned reject labels by construction, without solver work during label creation, and the runtime test uses the same broad constructed categories. The manuscript does not provide enough detail to determine how "provably unreachable" targets were generated, how close rejectable samples lie to the reachability or one-frame continuity boundaries, whether generator-specific cues persist across splits, or how performance varies between unreachable and discontinuous cases. The extremely high reject AUROC and nearly zero reject FEV may therefore reflect easy separation of sampling regimes as well as useful portfolio-risk prediction.  
**Why it matters**  Safe zero-solve refusal is the strongest reported mechanism and produces nearly all rejectable-query savings. Its technical credibility depends on showing that the gate recognizes contract failure rather than artifacts of the synthetic generator.  
**Resolution test**  Specify the rejectable-query generators and proof criteria, quantify distance to the relevant kinematic and step boundaries, and report per-subtype false-reject, recall, FEV, and latency. Add an independently generated or boundary-focused rejection set if such evidence exists. Otherwise, restrict the claim to the stated synthetic strata and avoid general refusal or reachability language.

### R1-M5 [reproducibility, outcome validity]

**Severity**  Major  
**Blocking**  No  
**Claim pointer**  The gate reduces mean function evaluations by about 28 percent for Panda and 34 percent for UR5e while preserving verified success.  
**Evidence pointer**  `main.tex`, lines 302--342, 443--461, 487--522, and 683--689; `claim_evidence_map.md`, lines 5 and 7.  
**Concern**  FEV is the central positive efficiency endpoint, yet its accounting is not defined with enough precision for independent reproduction. DLS includes Jacobian evaluation and multiple backtracking scales, while TRF has its own residual-call convention. The text does not state exactly which FK, residual, Jacobian, backtracking, candidate-scoring, fallback, or verification operations increment FEV, whether cached calls are counted, or how counts are normalized across DLS and TRF. The manuscript appropriately acknowledges that FEV excludes the learned inference overhead, but the undefined counter still limits interpretation of the claimed numerical-work reduction.  
**Why it matters**  A shared implementation makes within-code comparisons plausible, but readers still need to know what physical computation the headline metric represents and whether one FEV is comparable across stages.  
**Resolution test**  Define the FEV counter operationally for every solver stage, describe caching and repeated evaluations, distinguish residual and Jacobian work where appropriate, and release the counting code and query-level values referenced in the Data Availability statement. Retain end-to-end latency as a separate endpoint and do not translate FEV reduction into runtime improvement.

### R1-M6 [statistical rigor, experimental design]

**Severity**  Major  
**Blocking**  No  
**Claim pointer**  Trajectory outcomes were retained under the tested kinematic contract across forty 150-frame paths per run.  
**Evidence pointer**  `main.tex`, lines 421--425, 443--461, 555--566, and 731--733; `claim_evidence_map.md`, line 10.  
**Concern**  The whole path is correctly treated as the independent unit, but the Results give only completion percentages across forty paths and no paired path-level uncertainty or prespecified non-inferiority analysis. The same locked trajectories are reused across training seeds, so the apparent stability across three runs does not expand the independent path sample. One UR5e run improves by one path and the other two tie. These data support a descriptive statement but do not establish equivalence or broad trajectory retention.  
**Why it matters**  Continuity is an important part of the online-IK case, and frame-level acceptance cannot substitute for path-level evidence. The current wording risks reading as a general preservation result from a small fixed path set.  
**Resolution test**  Report paired path-level outcomes by trajectory family, failure locations and reasons, and an uncertainty interval or explicitly prespecified retention margin at the path level. If power is insufficient, label the trajectory result descriptive and restrict it to the forty locked paths per robot.

## Minor Comments

### R1-m1 [writing clarity, reproducibility]

**Severity**  Minor  
**Affected element**  Accepted command-spike outcome.  
**Evidence pointer**  `main.tex`, lines 443--449 and 555--566.  
**Issue**  The manuscript reports zero accepted command spikes but does not define the spike metric or threshold separately from the verifier's one-frame velocity constraint.  
**Required correction**  Give the exact formula, threshold, units, and denominator, and state whether this outcome is logically guaranteed by the acceptance contract or measures an additional property.

### R1-m2 [statistical rigor]

**Severity**  Minor  
**Affected element**  Cluster bootstrap, sign-flip tests, and adjusted p values.  
**Evidence pointer**  `main.tex`, lines 451--460 and 495--501.  
**Issue**  The generating-identifier cluster is not defined, and the phrase "exact adjusted p values" is ambiguous beside 10,000 resamples. It is unclear whether the reported `p=0.000300` values are exact enumerations or Monte Carlo estimates followed by Holm adjustment.  
**Required correction**  Define the cluster membership, sign-flip implementation, number of permutations, random seed, finite-sampling correction, and the exact order of multiplicity adjustment. Use "exact" only if the permutation space was exhaustively enumerated.

### R1-m3 [figures and tables]

**Severity**  Minor  
**Affected element**  Prespecified secondary hard-valid outcome.  
**Evidence pointer**  `main.tex`, lines 416--425 and 443--449.  
**Issue**  Hard-valid success is named as a secondary outcome and is important for skipped-stage risk, but no numerical hard-valid result is visible in the Results tables or prose.  
**Required correction**  Report hard-valid success, escalation, and FEV for the primary pair, or explain why the prespecified secondary endpoint is omitted.

### R1-m4 [reproducibility]

**Severity**  Minor  
**Affected element**  End-to-end latency protocol.  
**Evidence pointer**  `main.tex`, lines 353--361, 380--388, and 459--467.  
**Issue**  The software and GPU are reported, but CPU model, thread counts, affinity, power or clock policy, background-load control, and repeated-timing structure are not. These details can materially affect millisecond-scale batch-one P95 comparisons.  
**Required correction**  Provide the missing host and runtime controls, state whether each query was timed once or repeatedly, and report an uncertainty or repeatability check for the tail summaries.

### R1-m5 [writing clarity]

**Severity**  Minor  
**Affected element**  Formal versus validation-only visual hierarchy.  
**Evidence pointer**  `main.tex`, lines 60--88, 601--650, and 745--761.  
**Issue**  The prose is careful, but the abstract gives precise favorable pilot ratios immediately after the failed formal endpoint. A reader scanning numbers can still mistake selected validation performance for confirmation.  
**Required correction**  Prefix every pilot value in the abstract, figure caption, table caption, and conclusion with an explicit validation-only marker and state in the same sentence that the backend was selected before any independent optimized test.

## Technical failings that need to be addressed before the case is established

The blocking issue is R1-M1. No positive optimized feasible-latency conclusion is established until the locked packaging and independent test conditions pass, or until the submission is explicitly recast around the formal negative runtime result. R1-M2 through R1-M6 materially limit comparative value, mechanism attribution, refusal validity, reproducibility of the primary efficiency metric, and the trajectory claim.

## Assessment against the five rigor axes

- **Originality**  The portfolio-specific action label and the matched shared-cascade comparison are a clear and potentially original systems contribution. Originality relative to existing learned warm starts and solver portfolios is plausible from the manuscript's own discussion but is not fully established because results for the declared conventional comparators are absent.
- **Scientific importance**  The separation of learned allocation, numerical refinement, and explicit verification is useful for robotics engineering. The present importance is field-local because the work is an exact-kinematics benchmark and the only formal end-to-end feasible-latency result is negative.
- **Interdisciplinary readership interest**  The design principle could interest adaptive-computation, real-time systems, and safe-learning audiences. Broader interest currently rests on the architecture and the transparent negative result rather than on demonstrated deployment performance.
- **Technical soundness**  The formal within-benchmark success and FEV comparison is credible and unusually well bounded. The optimized latency claim is not independently tested, the release package failed a locked equivalence tolerance, and several core mechanisms and outcomes need fuller reporting.
- **Readability for nonspecialists**  The manuscript explains the proposal, cascade, and verifier roles clearly and openly distinguishes formal from validation evidence. Readability would improve further if FEV were operationally defined and if favorable pilot numbers were visually separated more sharply from the failed formal latency conclusion.

## Risk and unsupported-claim register

- A positive optimized feasible-latency claim is unsupported by formal evidence. `test_v3` is absent.
- Release-ready numerical equivalence across all six artifacts is unsupported because the locked packaging protocol stopped on UR5e seed 43.
- Competitive advantage over the four declared conventional comparators is not assessable from the reported Results.
- Individual necessity of history, ensemble diversity, uncertainty, or calibration is not supported by the mostly null operational ablations.
- General rejection safety, physical reachability classification, collision safety, controller tracking, acceleration or jerk compliance, torque compliance, and hardware reliability are not established. The manuscript mostly acknowledges these boundaries.
- Cross-robot generalization and deployment-prevalence performance are not established. The two robot-specific models and stratified query mixture do not test those claims.
- Broad trajectory equivalence is not established from forty fixed paths per robot without path-level uncertainty. The supported statement is descriptive retention on the locked kinematic path set.
