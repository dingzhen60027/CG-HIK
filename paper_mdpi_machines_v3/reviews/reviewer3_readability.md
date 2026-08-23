# Reviewer 3 Report

## Review setup

**Input scope**

The assessment covers `main.tex`, `generated/evidence_snapshot.json`, `claim_evidence_map.md`, and `release_v3_packaging_diagnostic.md`. No other reviewer report was consulted.

**Assessment boundary**

The evidentiary record consists of six frozen `paper_v2` formal runs, with two robot-specific locked runtime test sets and three training-seed sensitivity runs per robot. The `latency_pilot_v3` results are validation-only and use training seed 17. The all-artifact release package is incomplete, and `test_v3` has not started. I do not infer any experiment or result beyond that boundary.

**Emphasis**

Interdisciplinary readership and readability for nonspecialists. The five Nature-style axes are used only as rigor checks for an engineering-journal submission, not as a venue-fit decision.

## Overall assessment

The manuscript has a clear and potentially useful system-level message. Learning chooses where a shared numerical IK cascade should begin and whether it should refuse a request, while a kinematic verifier retains authority over acceptance. The controlled fixed-entry comparison, retention of negative latency evidence, and explicit claim boundaries are strong features. The formal results support reduced numerical work without loss of feasible-query success on the tested kinematic benchmark. They do not establish improved feasible-query latency, physical safety, hardware reliability, or a hard real-time guarantee.

The paper is technically literate and unusually transparent about its negative formal result. Its main readability weakness is not sentence-level prose. It is evidence-state management. A reader must combine the boxed notice, the abstract, two latency subsections, the limitations, an appendix table, and the external packaging diagnostic to understand which latency result is formal, which is validation-only, which artifact set failed a locked tolerance, and which new test is absent. A second interpretive risk is that repeated references to a two-robot, three-seed or six-run evaluation can be read as six independent test datasets, although the paper later explains that each robot reused one locked runtime test across seeds.

I would support publication after major revision focused on evidence provenance, scope language, and the operational definitions needed to make the work legible outside specialist IK and machine-learning audiences. None of the concerns below is blocking for the bounded formal claim that the gate reduced numerical evaluations while preserving tested outcomes. Several are material to the broader online-runtime interpretation.

## Who would be interested in the results, and why

The immediate audience includes researchers in robot kinematics, motion generation, numerical optimization, and learned warm starts. The system pattern should also interest readers working on runtime assurance, hybrid learning and optimization, adaptive computation, and safety-related software architectures. For these broader readers, the valuable principle is that a learned component can allocate effort and recommend refusal without becoming the acceptance authority. The manuscript should state more explicitly which parts of that principle transfer beyond this exact IK portfolio and which parts remain portfolio-specific.

## Major strengths

- The primary pair shares the proposal ensemble, numerical cascade, fallback, and verifier, so the comparison isolates entry-action routing more cleanly than a comparison against an unrelated solver would.
- The paper reports the failed formal feasible-latency gate instead of replacing it with a favorable validation result.
- The acceptance contract is explicit and separates solver convergence from command admissibility.
- The statistical unit is defined for point and trajectory outcomes, and the text states that training seeds are sensitivity replicates rather than independent test datasets.
- Null ablations and limitations are retained. The manuscript avoids claiming individual necessity for ensemble, calibration, uncertainty, and history components when those ablations were operationally close.
- The conclusions appropriately limit current evidence to exact URDF kinematics and exclude collision, torque, contact, controller tracking, and hardware reliability claims.

## Major Concerns

### R3-M1 [writing-clarity]

**Severity** Major

**Blocking** No

**Axis** Readability for nonspecialists and technical soundness

**Claim pointer** The formal eager implementation failed the feasible-latency gate, an exact exported implementation then passed validation-only latency and numerical-equivalence checks at seed 17, release packaging later remained incomplete, and no independent optimized test has been run.

**Evidence pointer** `main.tex`, Evidence-status notice, Abstract, Exact-export latency remediation, The original formal feasible-latency claim failed, Validation-only exact export met the runtime pilot gates, and Limitations and next validation steps. `generated/evidence_snapshot.json`, `formal_test.paper_gate_pass`, `latency_pilot_v3.rows`, `latency_pilot_v3.training_seed`, and `latency_pilot_v3.test_v3_started`. `release_v3_packaging_diagnostic.md`, Protocol outcome and Interpretation and action boundary.

**Concern** The paper does not provide one consolidated provenance view of these four states. It also states in the validation-pilot subsection that the exact backend met numerical equivalence, while the later six-artifact packaging attempt stopped at UR5e seed 43 because frozen offline risk-probability and risk-score errors exceeded the locked tolerance by small amounts. The boxed notice says packaging is incomplete, but it does not explain this distinction. A nonspecialist can reasonably leave with the impression that the optimized implementation has already passed the complete equivalence and release process for all robot and seed combinations.

**Why it matters** The distinction determines what latency claim is established. Ambiguity here can turn a carefully bounded validation result into an apparent independent test claim and can obscure why the formal paper-level gate remains false.

**Resolution test** Add a compact evidence-provenance table or flow in the main paper. For each phase, report the data split, robots and seeds, artifact status, gate outcome, and allowed inference. State explicitly that the seed-17 validation pilot passed its equivalence gates, that the all-artifact packaging attempt stopped at UR5e seed 43 after a strict offline frozen-risk tolerance failure, that sampled runtime routing and solver behavior still agreed in the diagnostic, and that `test_v3` is absent. Repeat a one-sentence version at the first optimized-latency claim in the abstract and conclusion.

### R3-M2 [claim-moderation]

**Severity** Major

**Blocking** No

**Axis** Technical soundness, scientific importance, and readability for nonspecialists

**Claim pointer** The method is presented for online robot control with a 20 ms frame interval, bounded numerical budgets, and operationally meaningful tail latency.

**Evidence pointer** `main.tex`, Abstract, Online IK query and acceptance contract, Outcomes and statistical analysis, Why the negative latency result matters, Validation-only exact export met the runtime pilot gates, and Conclusions. The validation subsection reports a 306.46 ms Panda outlier after 373 function evaluations.

**Concern** The manuscript does not define whether a returned command must meet the 20 ms control deadline, what the runtime does on a deadline miss, or how often the formal and pilot distributions exceed that deadline. P95 is informative but does not establish bounded wall-clock execution or hard real-time suitability. The reported 306.46 ms validation outlier makes this distinction concrete.

**Why it matters** Readers outside numerical IK may equate a bounded solver budget, a 50 Hz query interval, and low P95 latency with a real-time execution guarantee. The evidence instead supports empirical latency under the measured software configuration. This affects the interdisciplinary significance of the claimed online-runtime contribution.

**Resolution test** Define the timing contract and failure behavior. Report the observed fraction of queries exceeding 20 ms for each relevant formal condition and for the validation pilot if those values are available from the frozen evidence. Otherwise state that deadline-miss frequency is not reported and narrow all real-time language to empirical runtime behavior. Explicitly say that the iteration or FEV budget is not a wall-clock bound.

### R3-M3 [reproducibility]

**Severity** Major

**Blocking** No

**Axis** Technical soundness and readability for nonspecialists

**Claim pointer** The gate rejects futile unreachable or deliberately discontinuous requests before numerical work, while preserving the fixed cascade's final rejection outcome.

**Evidence pointer** `main.tex`, Portfolio-specific action labels and calibrated risk model, Robots, software, and data separation, Explicit rejection removed futile work and reduced verifier load, and IK as verified resource allocation. `claim_evidence_map.md`, rows for explicit rejection and verifier necessity.

**Concern** The paper names 1,000 provably unreachable and 1,000 discontinuous point queries per run but does not give a sufficiently operational definition of how unreachability was proved, how discontinuity was generated, or how a nominally feasible query that exhausts the portfolio differs from the two constructed reject categories. The action-label description combines these cases under `reject`. This makes it difficult for a reader to judge whether rejection performance concerns geometric impossibility, violation of the one-frame command contract, or failure of the locked portfolio.

**Why it matters** These categories have different engineering meanings and different transfer risks. The strongest computational saving comes from explicit rejection, so the construction and scope of the rejectable set are central to reproducing and interpreting that result.

**Resolution test** Add exact generation and adjudication rules for each reject subtype, including the proof criterion for unreachable targets and the threshold or procedure for discontinuity. Report formal rejection and work outcomes separately by subtype when already present in the frozen outputs. If subtype results are unavailable, state that limitation and restrict the claim to the pooled constructed rejectable set. Use distinct terms throughout for geometric unreachability, command-contract discontinuity, and portfolio exhaustion.

### R3-M4 [writing-clarity]

**Severity** Major

**Blocking** No

**Axis** Scientific importance, technical soundness, and readability for nonspecialists

**Claim pointer** The work is described in the title, abstract, contribution list, results, and conclusion as a two-robot, three-training-seed or six-comparison evaluation.

**Evidence pointer** `main.tex`, title, Abstract, Introduction contribution 3, Robots, software, and data separation, Figure 2 caption, the primary-outcomes table labeled `tab:primary`, and Limitations and next validation steps. `generated/evidence_snapshot.json`, `formal_test.runtime_test_reused_across_seeds_within_robot`. `claim_evidence_map.md`, boundary for the feasible-success and FEV claim.

**Concern** The manuscript eventually states that each robot's same locked runtime test was reused across all three training seeds, but the headline language can still suggest six independent test datasets or six independent robot-level replications. The phrase averaged over three seed runs is especially easy to overread as replication of the test sample rather than sensitivity to model training.

**Why it matters** This affects how much empirical breadth readers assign to the findings. The evidence establishes consistency across training seeds on two fixed robot-specific test sets, not generalization across six independently sampled runtime tests.

**Resolution test** State the dependence structure in the abstract immediately after the seed count. Prefer formulations such as two robot-specific locked test sets, each evaluated with three training-seed sensitivity runs. Use the same wording in the contribution list, figure captions, conclusion, and any six-run summary. Avoid wording that treats seeds as independent experimental replicates.

### R3-M5 [novelty-significance]

**Severity** Major

**Blocking** No

**Axis** Originality, scientific importance, and interdisciplinary readership

**Claim pointer** The claimed advance is a calibrated, portfolio-specific entry action over a shared verified cascade, including explicit zero-solve rejection, rather than learned replacement of the numerical solver.

**Evidence pointer** `main.tex`, Introduction contribution list, Numerical IK, singularities, and solver portfolios, Hybrid learning and numerical refinement, Shared numerical cascade, and IK as verified resource allocation.

**Concern** The ingredients and closest prior approaches are described, but the novelty is distributed across prose. The paper does not give a compact comparison showing which prior systems learn proposals, allocate solver depth, support explicit refusal, use the same acceptance verifier across methods, and test against a matched fixed-entry cascade. Nor does it clearly separate what is reusable beyond IK from what is defined only relative to this locked portfolio.

**Why it matters** Specialist readers can reconstruct the distinction, but interdisciplinary readers may see another learned warm-start system with added classification. The broader scientific value depends on understanding the decision-level contribution and its limits.

**Resolution test** Add a concise, citation-grounded comparison table or structured paragraph against the closest portfolio, warm-start, and path-generation approaches already cited. Compare only dimensions supported by those sources. Then identify the transferable design principle and the portfolio-specific components. Do not broaden the empirical claim beyond the two exact-kinematics benchmarks.

## Minor Comments

### R3-m1 [claim-moderation]

**Severity** Minor

**Axis** Readability for nonspecialists

**Affected element** The word `verified` in the title and repeated description of a common verifier.

**Evidence pointer** `main.tex`, title, Online IK query and acceptance contract, and Limitations and next validation steps.

**Issue** The verifier checks finite values, pose error, joint limits, and one-frame velocity. The unqualified term can be mistaken for collision, dynamic, controller, or formal software verification despite later caveats.

**Required correction** Define the term at first use as kinematic contract verification and consider using `kinematically verified` in the title or subtitle. Preserve the explicit exclusions wherever broader safety interpretation is plausible.

### R3-m2 [reproducibility]

**Severity** Minor

**Axis** Technical soundness and readability for nonspecialists

**Affected element** Command-spike rate and the statement that no accepted trajectory command was classified as a spike.

**Evidence pointer** `main.tex`, Outcomes and statistical analysis, Trajectory outcomes were retained, Figure 4 caption, and Conclusions.

**Issue** The manuscript does not define the spike metric or threshold in the visible methods.

**Required correction** Give the exact formula, units, threshold, and handling of rejected or post-failure frames. If the metric is identical to a component of the acceptance contract, explain the distinction.

### R3-m3 [writing-clarity]

**Severity** Minor

**Axis** Readability for nonspecialists

**Affected element** Action terminology and the sequence from proposal to routing, refinement, verification, and return.

**Evidence pointer** `main.tex`, Abstract, Portfolio-specific action labels and calibrated risk model, Table 1, and Figure 1 caption.

**Issue** `Easy`, `medium`, and `hard` sound like intrinsic query difficulty, although they are portfolio-specific entry actions. The prose states this, but the terminology can still be misread.

**Required correction** Add a short worked example or callout that follows one feasible query and one rejected query through the system. Label the actions as entry levels whenever space permits, and state that they are not universal difficulty classes.

### R3-m4 [figures-and-tables]

**Severity** Minor

**Axis** Readability for nonspecialists

**Affected element** The primary-outcomes table labeled `tab:primary` and Figure 2 presentation of formal latency and work outcomes.

**Evidence pointer** `main.tex`, table labeled `tab:primary` and Figure 2 caption.

**Issue** Percentage reduction changes sign convention across favorable and unfavorable outcomes, and the main table omits absolute feasible-latency values while emphasizing reject P95 reduction. A reader must consult the appendix to see the formal feasible P95 increase in milliseconds.

**Required correction** Put the formal fixed and proposed feasible P95 values, or their ratio, in the main results table. Use an explicit direction label such as lower is better and visually separate the failed feasible-latency gate from favorable rejectable-query latency.

### R3-m5 [writing-clarity]

**Severity** Minor

**Axis** Readability for nonspecialists

**Affected element** Abbreviations and statistical terms in the abstract and early Results.

**Evidence pointer** `main.tex`, Abstract, Held-out action risk was well separated and calibrated, and Outcomes and statistical analysis.

**Issue** AUROC, ECE, macro-F1, cluster-bootstrap intervals, sign-flip tests, and Holm correction appear in a compact sequence without a plain-language statement of what each analysis establishes.

**Required correction** Add one short interpretive sentence after each metric group. Explain that calibration concerns probability reliability, discrimination concerns ranking rejectable cases, and the paired inference concerns query-level work or success differences. Keep the technical values for specialists.

## Technical failings that need to be addressed before the case is established

No `Blocking Yes` technical failing was identified for the bounded formal claim that routing preserved tested feasible success and reduced FEV on the supplied benchmark. Before a broader online-runtime or cross-disciplinary case is established, the manuscript should resolve R3-M1 through R3-M5. In particular, it must make the latency evidence states unambiguous, define the timing contract and rejectable categories, represent the seed dependence accurately, and sharpen the novelty bridge.

## Assessment against the five Nature-style quality axes

**Originality**

The portfolio-specific action label, matched fixed-entry comparison, explicit rejection, and retained non-learned acceptance authority form a credible original system contribution from the supplied text. The distinction from adjacent warm-start and solution-generation work needs a more scannable comparison. This is R3-M5.

**Scientific importance**

The work is significant for engineering practice because it shows that lower numerical work can coexist with worse end-to-end latency and because it preserves a clean boundary between learned allocation and geometric acceptance. Broader importance is plausible but not yet demonstrated beyond exact kinematics, two robot models, and constructed query mixtures. The 20 ms timing interpretation must be clarified under R3-M2.

**Interdisciplinary readership**

Runtime-assurance, adaptive-computation, and hybrid-model readers could benefit from the design pattern. The manuscript currently assumes enough IK knowledge that the transferable principle is easy to miss. R3-M5 and R3-m3 would make the result legible beyond the immediate specialty.

**Technical soundness**

The frozen comparison, common verifier, data-role separation, paired query analysis, and explicit negative latency result are strengths. Confidence is bounded by reused runtime tests across seeds, an incompletely defined rejectable cohort, validation-only optimized latency, incomplete all-artifact packaging, and absent `test_v3`. These boundaries do not invalidate the formal FEV result, but they prevent a positive optimized-latency or hard real-time conclusion.

**Readability for nonspecialists**

The local prose is generally clear, the method sequence is coherent, and the limitations are candid. The principal difficulty is reconstructing evidence provenance and dependence across phases. A single provenance table, consistent seed wording, a worked routing example, and explicit definitions for verification, rejection, spike, and timing would materially improve accessibility.

## Recommendation posture

Major revision. The bounded formal resource-allocation result is promising and transparently reported. The manuscript should be publishable as an engineering study if it makes the evidence hierarchy and scope impossible to misread, closes the operational-definition gaps, and presents the originality in a form accessible to readers outside specialist IK research. The validation-only optimized latency result should remain a readiness result until an independent `test_v3` exists.

## Risk and unsupported-claim flags

- A positive optimized feasible-latency claim is not established by an independent test. The current support is validation-only at seed 17, all-artifact packaging is incomplete, and `test_v3` is absent.
- A hard real-time or 50 Hz deadline guarantee is not established. The manuscript reports empirical percentiles and one 306.46 ms validation outlier.
- Six independent runtime test datasets are not present. There are two robot-specific locked runtime test sets reused across three training seeds.
- Physical safety, collision avoidance, torque or acceleration compliance, controller tracking, and hardware reliability are outside the supplied evidence.
- Natural deployment prevalence and out-of-distribution rejection are not established by the stratified constructed point mixture.
- Individual causal necessity of the ensemble, uncertainty, calibration, and history components is not supported by the mostly null operational ablations.
- Cross-robot model generalization is not established because the models were trained separately for Panda and UR5e.
