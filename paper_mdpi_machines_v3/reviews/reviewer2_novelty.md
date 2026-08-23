# Reviewer 2

## Review setup

- **Preassigned emphasis**  Originality and scientific significance.
- **Input scope**  The assessment is based only on `main.tex`, `generated/evidence_snapshot.json`, `claim_evidence_map.md`, and `release_v3_packaging_diagnostic.md`.
- **Assessment boundary**  The evidential base comprises the six frozen `paper_v2` formal runs and the validation-only latency pilot. `test_v3` is absent and is not inferred. The three training seeds reuse one locked runtime test set within each robot and are sensitivity runs rather than independent test datasets.
- **Journal framing**  I assess the work as an engineering-journal submission. The five Nature-style axes are used only as rigor checks, not as a judgment of venue fit.
- **Shared manuscript claim summary**  A calibrated, history-conditioned four-action gate chooses where to enter a common verified IK cascade or rejects a request without solving. The formal evidence supports preserved feasible-query success, lower mean numerical work, and faster handling of constructed rejectable queries. The formal feasible-query latency gate failed. An exact export improved latency on validation only, but release packaging remains incomplete and no independent optimized test has begun.
- **Missing materials affecting confidence**  There is no fresh optimized test, no complete six-artifact release package, no hardware or physics validation, no natural deployment workload, and no independent test set for each training seed.

## Overall assessment

The manuscript contains a credible and potentially useful engineering idea. Its most original feature is not learned IK in isolation, but the treatment of learning as an auditable resource-allocation layer over a shared numerical portfolio whose verifier retains command authority. The fixed-entry comparison is unusually clean and the paper is commendably transparent about its negative formal latency result and mostly null component ablations.

The present evidence nevertheless supports a narrower contribution than the title and online-control framing may initially suggest. It establishes lower function-evaluation demand and near-zero solver work on the deliberately constructed rejectable set. It does not yet establish a faster feasible-query runtime for the optimized implementation, and it does not show decisively that the learned four-action policy provides a scientifically important advantage over simpler allocation policies or current adaptive solver portfolios. In my view, the manuscript needs major revision before its originality and engineering importance are convincingly established.

## Who would be interested in the results, and why

The immediate audience is researchers and engineers working on numerical IK, manipulator motion generation, real-time robotics software, solver portfolios, and learned warm starts. A secondary audience in dependable machine-learning systems may value the separation between learned allocation and non-learned verification. Interest beyond these areas is likely to remain limited until the approach is shown under a realistic operating distribution or in a physical or physics-based control loop.

## Major strengths

- The portfolio-specific action label and the explicit zero-solve reject action form a coherent system contribution.
- The proposed and fixed methods share seeds, numerical stages, fallback, and verification, so the principal comparison isolates entry routing rather than conflating unrelated solvers.
- The manuscript retains negative latency evidence, null ablations, sensitivity limitations, and the absence of `test_v3` instead of converting validation into a test claim.
- The common verifier gives the work a clear engineering contract and prevents the learned model from being presented as the source of geometric correctness.
- The distinction between training-seed sensitivity and independent test replication is stated accurately.

## Major Concerns

### R2-M1

- **Severity**  Major.
- **Blocking**  Yes.
- **Axis**  Scientific importance and technical soundness.
- **Claim pointer**  The abstract and conclusion frame the method as an online runtime resource-allocation layer, and the Discussion argues that failure after hundreds of evaluations may be operationally unsuitable at 50 Hz.
- **Evidence pointer**  `main.tex`, Abstract; Results subsection `The original formal feasible-latency claim failed`; Results subsection `Validation-only exact export met the runtime pilot gates`; Appendix Table `Validation-Only Latency Table`; `generated/evidence_snapshot.json`, keys `formal_test.paper_gate_pass` and `latency_pilot_v3`; `release_v3_packaging_diagnostic.md`, sections `Protocol outcome` and `Interpretation and action boundary`.
- **Concern**  The only independent formal latency result is negative. Feasible-query P95 latency increased by 31.8 to 38.1 percent for Panda and 50.4 to 65.7 percent for UR5e. The optimized backend was assessed on validation only, at seed 17, and the locked packaging phase stopped at UR5e seed 43 because two strict equivalence tolerances failed. No `test_v3` process was started. Lower FEV is a valid algorithmic result, but it is not yet evidence of improved feasible-query online performance. Because online execution motivates the engineering significance of the work, the manuscript presently combines a demonstrated work-saving result with an unconfirmed practical-runtime remedy.
- **Why it matters**  Without independent optimized testing, a reader cannot decide whether the proposed gate is an online engineering improvement or primarily a study of numerical-work allocation whose original implementation is slower on feasible queries.
- **Resolution test**  Either complete the locked six-artifact packaging and a fresh preregistered `test_v3` without threshold or claim-gate changes, then report the result regardless of direction, or revise the title, abstract, contribution list, discussion, and conclusion so that practical speedup is not part of the established case. Under the latter route, the validation pilot should remain explicitly diagnostic and the significance claim should be limited to reduced FEV and fast rejection within the tested kinematic benchmark.

### R2-M2

- **Severity**  Major.
- **Blocking**  Yes.
- **Axis**  Originality and scientific importance.
- **Claim pointer**  The Introduction presents calibrated query-specific action selection over a fixed portfolio as the distinguishing advance, and the Related Work section states that prior systems do not learn such an action. The Methods list a validation-constrained Cartesian-step guard and several conventional solvers as primary comparators.
- **Evidence pointer**  `main.tex`, Introduction contribution items 1 and 2; Related Work subsections `Numerical IK, singularities, and solver portfolios` and `Hybrid learning and numerical refinement`; Methods subsection `Comparators and ablations`; Discussion paragraphs on the threshold guard. Table `Primary formal-test outcomes` reports only the fixed cascade and proposed method.
- **Concern**  The manuscript does not yet make the closest-alternative test visible enough to establish the claimed originality. The fixed-entry cascade demonstrates that routing can skip work, but it does not show that a calibrated learned policy is needed. The threshold guard is the most important simple alternative, yet its evidence is reduced to a brief statement that it loses about 0.9 percentage points of feasible success. Per-robot and per-seed results, FEV, rejection, trajectory, and latency outcomes for this guard are not presented in the main tables. Likewise, the distinction from adaptive restart, seed-retrieval, or solver-portfolio methods remains largely verbal rather than a precise comparison of decisions, information used, and matched operating constraints.
- **Why it matters**  The central originality claim is the learned allocation rule, not the existence of DLS, fallback, warm starts, or verification. If a simple guard or an existing adaptive policy obtains most of the same allocation benefit, the contribution is narrower and should be positioned accordingly.
- **Resolution test**  Present the already designated primary comparator results under the same estimands and locked test conditions as the proposed method, especially the Cartesian-step guard. Include feasible success, feasible FEV, rejectable acceptance and work, trajectory completion, and latency with per-robot and per-seed visibility. Add a compact feature-by-feature comparison with the closest cited adaptive portfolio methods. If no decisive advantage is supported, narrow the originality claim to a controlled demonstration of portfolio-specific learned routing and explicit verified refusal.

### R2-M3

- **Severity**  Major.
- **Blocking**  No.
- **Axis**  Originality and claim moderation.
- **Claim pointer**  The architecture foregrounds a five-member history-conditioned ensemble, disagreement features, isotonic calibration, and a four-action classifier. The Discussion identifies the calibrated action gate as a whole, especially explicit rejection, as the strongest supported learned mechanism.
- **Evidence pointer**  `main.tex`, Abstract; Methods subsections `History-conditioned candidate proposal` and `Portfolio-specific action labels and calibrated risk model`; Results subsection `Held-out action risk was well separated and calibrated`; Results and Discussion subsections on ablations; `claim_evidence_map.md`, row `Ensemble/calibration/uncertainty components are individually necessary`.
- **Concern**  The scientific advance is not yet localized within the comparatively elaborate learned design. Non-reject macro-F1 is only about 0.59 to 0.64, while removing history, retaining one member, removing disagreement features, or removing calibration changes formal operational FEV only slightly. The manuscript acknowledges these null results, which is a strength, but the abstract and architectural emphasis can still leave the impression that confidence estimation through the full ensemble and calibration stack is central to the demonstrated benefit. The evidence more clearly supports the integrated routing policy and explicit rejection than any necessity of the individual confidence components.
- **Why it matters**  A sharper attribution would make the contribution more original and more credible. At present, sophistication of implementation risks being mistaken for scientific necessity.
- **Resolution test**  Quantify the contribution of non-reject routing and explicit rejection separately, show action frequencies and per-action work savings, and state consistently that the benchmark does not establish the necessity of history, ensemble disagreement, or calibration. If such decomposition cannot distinguish the components, simplify the headline contribution to verified portfolio routing as an integrated system and move component-specific motivation out of the main novelty claim.

### R2-M4

- **Severity**  Major.
- **Blocking**  No.
- **Axis**  Scientific importance and interdisciplinary readership interest.
- **Claim pointer**  The manuscript motivates the method through closed-loop online control and presents the result as a system-level interpretation of modern IK.
- **Evidence pointer**  `main.tex`, Introduction opening paragraphs; Experimental Design subsection `Robots, software, and data separation`; Discussion subsection `Limitations and next validation steps`; Conclusions; `claim_evidence_map.md`, boundaries for rejectable-query prevalence and trajectory continuity.
- **Concern**  The demonstrated scope is two robot-specific models under exact URDF kinematics, synthetic and deliberately stratified point categories, and kinematically generated trajectories. The verifier excludes collision, torque, acceleration, jerk, contact, sensing error, and controller tracking. The reject set consists of provably unreachable or deliberately discontinuous requests, so its very high separability does not establish failure prediction near realistic deployment boundaries or under distribution shift. These limits are disclosed, but they substantially constrain the broader importance of the result.
- **Why it matters**  The current study establishes a careful kinematic benchmark result. It does not yet demonstrate deployment relevance, cross-robot transfer, or a broadly general principle under realistic uncertainty. This distinction should determine the strength of the significance language.
- **Resolution test**  In the current manuscript, consistently describe the evidence as a robot-specific exact-kinematics proof of concept and avoid implications of deployment readiness. A stronger general significance claim would require an independently locked evaluation under a realistic workload and at least one additional operating condition such as physics-based tracking, model perturbation, distribution shift, or hardware execution, with the same verifier and comparator constraints.

## Minor Comments

### R2-m1

- **Severity**  Minor.
- **Blocking**  No.
- **Axis**  Readability for nonspecialists.
- **Affected element**  Title and Abstract.
- **Claim pointer**  The title calls the study a `Two-Robot, Three-Seed Evaluation`, while the abstract describes three training seeds without immediately explaining the shared test set.
- **Evidence pointer**  `main.tex`, title, Abstract, and Experimental Design subsection `Robots, software, and data separation`.
- **Issue**  A nonspecialist may read the three seeds as three independent replications rather than model-sensitivity runs on the same robot-specific test queries.
- **Required correction**  Use `three-training-seed sensitivity evaluation` in the title or explain the shared-test design in the abstract at first mention.

### R2-m2

- **Severity**  Minor.
- **Blocking**  No.
- **Axis**  Readability for nonspecialists and scientific importance.
- **Affected element**  Abstract and Discussion.
- **Claim pointer**  The paper reports large FEV reductions alongside increased formal feasible-query latency.
- **Evidence pointer**  `main.tex`, Abstract; Results subsection `The original formal feasible-latency claim failed`; Discussion subsection `Why the negative latency result matters`.
- **Issue**  The distinction between fewer numerical evaluations and faster wall-clock execution is scientifically central but may not be obvious to readers outside numerical robotics.
- **Required correction**  Add one plain-language sentence near the first FEV result explaining that FEV measures solver work rather than end-to-end response time, and state what engineering benefit remains established when feasible-query latency does not improve.

### R2-m3

- **Severity**  Minor.
- **Blocking**  No.
- **Axis**  Originality and writing clarity.
- **Affected element**  Terminology surrounding `confidence-gated`.
- **Claim pointer**  The title and architecture use confidence as the organizing concept, while the classifier produces calibrated action probabilities for one frozen data-generating process.
- **Evidence pointer**  `main.tex`, title; Methods subsection `Portfolio-specific action labels and calibrated risk model`; Results subsection `Held-out action risk was well separated and calibrated`.
- **Issue**  The paper does not explicitly distinguish in-distribution probability calibration from confidence under distribution shift or model misspecification.
- **Required correction**  Define `confidence` operationally as calibrated action probability on the specified split and state that no out-of-distribution confidence claim is made.

### R2-m4

- **Severity**  Minor.
- **Blocking**  No.
- **Axis**  Technical soundness and reproducibility.
- **Affected element**  Data Availability Statement and submission metadata.
- **Claim pointer**  The source-data package is said to be included and full query-level outputs and model artifacts are promised before publication.
- **Evidence pointer**  `main.tex`, Author Contributions, Funding, Data Availability Statement, Acknowledgments, and Conflicts of Interest; `release_v3_packaging_diagnostic.md`, `Protocol outcome`.
- **Issue**  The manuscript retains author-input placeholders and a repository or DOI placeholder, while the optimized release package is incomplete. These are understandable in a working draft but must not remain in a submission package.
- **Required correction**  Resolve all author-input fields, identify the archival repository and access conditions, and make the data statement match the actual frozen package available at submission.

## Technical failings that need to be addressed before the case is established

The blocking issues are R2-M1 and R2-M2. The first concerns the unresolved distinction between solver-work reduction and practical online latency. The second concerns whether the learned allocation policy is meaningfully differentiated from the most relevant simple and adaptive alternatives. R2-M3 and R2-M4 do not invalidate the bounded kinematic findings, but they limit the strength and breadth of the originality and significance claims.

## Assessment against the five rigor axes

- **Originality**  Moderately strong at the integrated-system level. Portfolio-specific four-action routing with explicit verified rejection is a credible contribution, but its advantage over simpler allocation rules and closest adaptive portfolios is not yet demonstrated clearly enough.
- **Scientific importance**  Meaningful for numerical robotics engineering if framed as work allocation under an exact-kinematics contract. Importance as an online performance advance remains unconfirmed because the independent feasible-latency result is negative and optimized testing is absent.
- **Interdisciplinary readership interest**  The learned-allocation plus non-learned-verification pattern could interest dependable ML and real-time systems researchers. The present robot-specific synthetic benchmark limits broader reach.
- **Technical soundness**  The frozen formal comparison, common verifier, data-split discipline, and transparent negative result support the bounded FEV and rejection findings. The validation-only latency result and incomplete release package cannot support an independent optimized-runtime conclusion.
- **Readability for nonspecialists**  The manuscript is unusually clear for a technical IK paper and explains its limitations well. Accessibility would improve with a simpler explanation of FEV versus latency, a more explicit account of seed sensitivity versus replication, and a narrower operational definition of confidence.

## Recommendation posture

Major revision. This is not a final editorial recommendation and does not assess fit to a Nature journal. For an engineering journal, the work could become a solid and useful systems paper if the optimized runtime status is resolved or cleanly removed from the established case, and if the novelty of the learned gate is tested visibly against the strongest simple and adaptive alternatives.

## Risk and unsupported-claim register

- Optimized feasible-query latency is not independently established. The pilot is validation-only and `test_v3` has not started.
- Deployment packaging is not complete for all robot and seed combinations.
- The three seeds per robot are not independent runtime datasets.
- The point-mixture means do not estimate field prevalence or expected deployment performance.
- No collision, dynamics, controller tracking, sensing, torque, acceleration, jerk, contact, or hardware conclusion is supported.
- The data do not establish that ensemble diversity, disagreement features, history, or calibration are individually necessary.
- Cross-robot or cross-kinematic generalization is not assessed because the models are trained separately for each robot.
