# Recent IK references: verification batch B

Checked 2026-09-08. Scope: 11 existing keys in `paper/references.bib`. This is a literature-only note; no BibTeX, manuscript, benchmark, test, installation, or existing evidence was changed.

## Outcome and interpretation

- All 11 records exist. No DOI/title mismatch, omitted author, incorrect author order, or substantive title/venue/page discrepancy was found in the current BibTeX entries.
- Eight final-publication DOIs returned matching Crossref records. Initial HTTP 429 responses for four records resolved on a slower retry. Both arXiv DOI requests returned Crossref 404 but resolved correctly through DataCite and arXiv; **these are not invalid DOIs**. DiffusionSeeder has no DOI in the publisher's supplied citation.
- `nguyen2026sequential` has one status warning: its November 2026 issue assignment is confirmed, but the existing time-sensitive “In press as of 2 September 2026” wording was not independently reconfirmed. Its detailed method could not be checked in an accessible primary full text.
- Metadata verification and relevance are separate judgments. Verified publication metadata does not establish that an experiment meets this repository's online IK task contract.

For this triage, “directly relevant” means a paper actually addresses online IK execution, state/branch continuity, candidate selection, or learned initialization plus numerical correction. It does **not** mean the paper uses the same task constraints, tolerances, initialization, stopping/fallback policy, candidate budget, hardware, or end-to-end timing as the current benchmark. None of those equivalences is established here.

| Key | Core bibliography | Relevance judgment | Recommended role |
| --- | --- | --- | --- |
| `jayabalan2026hybrid` | Verified | Direct hybrid-IK method; restricted local validation | Optional hybrid-refinement comparator/context |
| `jlidi2026xgnn` | Verified | Direct learned warm-start plus DLS | Relevant warm-start related work |
| `nguyen2026sequential` | Verified; status wording warning | Strong subject match, primary content check incomplete | Prioritize for full-text review before detailed comparison |
| `nguyen2026setik` | Verified | Direct multi-candidate selection and branch continuity | Strong execution-oriented related work |
| `demby2024learning` | Verified | Direct learned IK family; contract evidence not checked | Learned-IK context, not a contract-aligned performance claim |
| `diprasetya2025kinenn` | Verified | Indirect kinematic-model/control background | Omit from a tight online-IK comparison unless needed |
| `go2025emiknet` | Verified | Direct multi-solution/multi-effector IK | Candidate-generation context; not evidence of temporal continuity |
| `zhang2025ikdiffuser` | Verified preprint | Direct generative IK, partial goals and guided initialization | Relevant recent preprint; label status |
| `yang2026mimik` | Verified preprint | Direct state-conditioned online IK and smooth execution | Strong recent preprint; distinguish its reported tolerance regime |
| `carvalho2025mpd` | Verified | Motion-planning background, not query-level IK | Background only, not an IK baseline |
| `huang2025diffusionseeder` | Verified | Trajectory seeding for motion optimization, not query-level IK | Background only, not an IK baseline |

## Per-reference checks

### 1. `jayabalan2026hybrid`

- Ordered authors: **Meenalochani Jayabalan; Karunamoorthy Loganathan; Palanikumar Kayaroganam**.
- Exact title: *A Novel Hybrid IK Architecture for Robotic Arms: Iterative Refinement of Soft-Computing Approximations with Validation on ABB IRB-1200 Robotic Arm*.
- Final journal article: **Machines 14(3), 292 (2026)**; published **4 March 2026**. DOI **10.3390/machines14030292**.
- Current BibTeX matches. `292` is an article identifier, although the publisher/Crossref citation presents it in the page position.
- Relevance: an ANFIS estimate initializes one of three Jacobian refiners. This is genuinely relevant to learning-plus-correction IK. However, the primary article's validation is confined to a representative 100 mm cubic workspace, explicitly chosen away from singular configurations. Its broad architecture claims should not be converted into demonstrated global-workspace robustness, branch continuity, or an identical online contract.
- Evidence: [Crossref record](https://api.crossref.org/works/10.3390/machines14030292), [publisher article, especially §§1.5, 3.2.1 and 4](https://www.mdpi.com/2075-1702/14/3/292). Publisher page was accessible through indexed primary text after a direct-open rate limit.

### 2. `jlidi2026xgnn`

- Ordered authors: **Ali Jlidi; Rabab Benotsmane; László Kovács**.
- Exact title: *Explainable Graph Neural Networks Towards Data-Driven Inverse Kinematics in Industrial Robot Motion Planning*.
- Final journal article: **Electronics 15(14), 3071 (2026)**; published **13 July 2026**. DOI **10.3390/electronics15143071**.
- Current BibTeX matches, including accents and author order.
- Relevance: explicitly a learned initializer for downstream DLS, not a standalone high-precision solver. The primary abstract reports reduced iterations/time and improved convergence on ABB IRB 2400 and UR5, plus calibration perturbation experiments. This supports warm-start positioning. It does not establish equivalence to previous-state warm starts, a particular task-priority hierarchy, or a common failure/recovery contract. Avoid quoting inference-only or paper-specific timing as a fair end-to-end comparison to this benchmark.
- Evidence: [Crossref record](https://api.crossref.org/works/10.3390/electronics15143071), [publisher article](https://www.mdpi.com/2079-9292/15/14/3071), [publisher version history confirming version of record](https://www.mdpi.com/2079-9292/15/14/3071/notes).

### 3. `nguyen2026sequential`

- Ordered authors: **Duc Tien Nguyen; Van Thanh Tri Nguyen; Duc Toan Luu; Truong Do; Vu Linh Nguyen**.
- Exact Crossref/Elsevier title: *Trajectory-based sequential learning for efficient and stable inverse kinematics of robotic manipulators*.
- Journal record: **Robotics and Autonomous Systems 205, 105669 (2026)**; cover issue **November 2026**. DOI **10.1016/j.robot.2026.105669**; PII **S0921889026003416**.
- Current core BibTeX matches; title capitalization is only informational. The issue is future-dated relative to this check. Crossref gives `published-print = 2026-11` and no online-publication date. Elsevier's accessible XML independently supplies the title, DOI, journal and November cover date, but no manuscript-stage flag, authors, abstract or full text.
- Status recommendation: retain a journal citation and the assigned volume/article number. Do not call it a preprint. Replace or omit the stale, independently unconfirmed “In press as of 2 September 2026” clause; “Assigned to the November 2026 issue” is supported.
- Relevance: the primary title establishes trajectory-based/sequential IK as a strong subject match. A third-party abstract describes previous-state conditioning and relative-joint prediction, but **that detailed technical description is not primary-source verified in this pass**. Do not use its accuracy/latency numbers or assert its exact online information contract until obtaining the publisher or author manuscript.
- Evidence: [Crossref record](https://api.crossref.org/works/10.1016/j.robot.2026.105669), [Elsevier primary XML metadata](https://api.elsevier.com/content/article/PII:S0921889026003416?httpAccept=text/xml), [publisher landing page](https://www.sciencedirect.com/science/article/pii/S0921889026003416) (direct access returned 403).

### 4. `nguyen2026setik`

- Ordered authors: **Duc Tien Nguyen; Van Thanh Tri Nguyen; Truong Do; Vu Linh Nguyen**.
- Exact title: *Multi-Solution Inverse Kinematics for Robotic Manipulators via Permutation-Invariant Set Prediction*.
- Final journal article: **IEEE Robotics and Automation Letters 11(8), 9803–9810 (2026)**; August issue. DOI **10.1109/LRA.2026.3703280**; IEEE document **11561052**.
- Current BibTeX matches. The institution's abbreviated author listing has apparent initials typos, so full ordered names should come from the publisher-deposited Crossref metadata, not be reconstructed from that web profile.
- Relevance: the accessible accepted-paper text formulates unordered candidate prediction, permutation-invariant supervision, and an execution-time picker that favors the candidate nearest the previously executed configuration. It is strong related work for separating candidate generation from executable branch-consistent commands. Still, fixed-cardinality set coverage, per-query accuracy and trajectory success need distinct treatment; shared tolerance and timing contracts are not established by this audit.
- Evidence: [Crossref record](https://api.crossref.org/works/10.1109/LRA.2026.3703280), [author institution publication listing](https://vinuni.edu.vn/people/nguyen-vu-linh-phd/), [accepted article text carrying the IEEE DOI and publication notice](https://www.researchgate.net/publication/407016674_Multi-Solution_Inverse_Kinematics_for_Robotic_Manipulators_via_Permutation-Invariant_Set_Prediction). The last source is the primary paper hosted on ResearchGate, not a ResearchGate-generated method summary; indexed paper text was accessible despite a direct-open rate limit.

### 5. `demby2024learning`

- Ordered authors: **Jacket Demby's; Ramy Farag; Guilherme N. DeSouza**.
- Exact title: *Inverse Kinematics of Robotic Manipulators Using a New Learning-by-Example Method*.
- Final proceedings article: **2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 9534–9541 (2024)**. DOI **10.1109/IROS58592.2024.10802048**.
- Current BibTeX matches. The unusual family name **Demby's** is genuine in the author lab's citation and Crossref (curly apostrophe there); do not “correct” it to Demby. Crossref and the author lab agree on pages 9534–9541. An older program/TOC surfaced with 9533–9540; do not replace the mutually corroborated final pages with that program range.
- Relevance: direct learned IK prior art. A primary full-text method/runtime audit was not completed, so do not assert that example conditioning is equivalent to this benchmark's current-state input or that its performance numbers meet the same success contract.
- Evidence: [Crossref record](https://api.crossref.org/works/10.1109/IROS58592.2024.10802048), [ViGIR author-lab publication page and supplied BibTeX](https://vigir1.ee.missouri.edu/Publications/publications.html) (indexed source text; direct open timed out).

### 6. `diprasetya2025kinenn`

- Ordered authors: **Mochammad Rizky Diprasetya; Johannes Pöppelbaum; Andreas Schwung**.
- Exact title: *KineNN: Kinematic Neural Network for inverse model policy based on homogeneous transformation matrix and dual quaternion*.
- Final journal article: **Robotics and Computer-Integrated Manufacturing 94, 102945 (2025)**; August issue. DOI **10.1016/j.rcim.2024.102945**.
- Current BibTeX matches. The `2024` within the DOI does not change the **2025** journal-volume year.
- Relevance: the primary article centers on learnable kinematic structures and parameter sharing between a world model and an inverse-model policy in model-based reinforcement learning. It is useful for kinematics-aware learning or transfer between robot architectures, but weak support for a contract-aligned online IK solver comparison. An invertible/IK-related architecture is not by itself evidence of branch-consistent query solving, uniform tolerances or bounded solve latency.
- Evidence: [Crossref record](https://api.crossref.org/works/10.1016/j.rcim.2024.102945), [publisher abstract and highlights](https://www.sciencedirect.com/science/article/pii/S0736584524002321), [institution-hosted article](https://publikationen.fhb.fh-swf.de/servlets/MCRFileNodeServlet/fhswf_derivate_00002757/Kinematic%20Neural%20Network%20for%20inverse%20model%20policy%20based%20on%20homogeneous%20transformation%20matrix%20and%20dual%20quaternion.pdf).

### 7. `go2025emiknet`

- Ordered authors: **Youn-Jae Go; Jun Moon**.
- Exact title: *EMIKNet: Expanding Multiple-Instance Inverse Kinematics Network for Multiple End-Effector and Multiple Solutions*.
- Final journal article: **IEEE Access 13, 25087–25096 (2025)**. DOI **10.1109/ACCESS.2025.3539022**. Primary article front matter gives publication **4 February 2025**, current version **10 February 2025**.
- Current BibTeX matches. Preserve the publisher's singular “Multiple End-Effector” title wording; do not silently grammaticalize it. ResearchGate's generic January label conflicts with the paper's own front matter and should not supply a publication month.
- Relevance: direct generative multi-solution/multi-effector IK using hierarchical Gaussian-mixture computation and joint-limit treatment. It is relevant for candidate-generation capacity and runtime/accuracy tradeoffs. Its multiple-solution mechanism does not itself establish continuity-aware online candidate selection, a strict hierarchy of task residuals, or matching benchmark contracts.
- Evidence: [Crossref record](https://api.crossref.org/works/10.1109/ACCESS.2025.3539022), [primary published article text hosted on ResearchGate](https://www.researchgate.net/publication/388696418_EMIKNet_Expanding_Multiple-instance_Inverse_Kinematics_Network_for_Multiple_End-effector_and_Multiple_Solutions), [IEEE landing page](https://ieeexplore.ieee.org/document/10872929/) (bot-verification page on direct access).

### 8. `zhang2025ikdiffuser`

- Ordered authors: **Zeyu Zhang; Ziyuan Jiao**.
- Exact arXiv/DataCite title: *IKDiffuser: a Diffusion-based Generative Inverse Kinematics Solver for Kinematic Trees*.
- **Preprint**, arXiv **2506.13087**, first submitted **16 June 2025**; latest displayed **v4, 14 January 2026**. DOI **10.48550/arXiv.2506.13087**. The arXiv comment remains “under review”; no final venue is identified in the inspected records.
- Current BibTeX is substantively correct. Keep year 2025 for the first-posted preprint and document the evaluated version if using v4 claims. Only title-case differences are present.
- Relevance: directly addresses IK across kinematic trees, partially specified end-effector goals, inference-time objective guidance, and initializing numerical optimization. This is especially pertinent when distinguishing target specification from candidate generation/refinement. However, objective-guided sampling is not automatically a hard task-priority guarantee, and variable end-effector masks are not automatically the same partial-task contract used here.
- Evidence: [arXiv abstract and version history](https://arxiv.org/abs/2506.13087), [DataCite record](https://api.datacite.org/dois/10.48550/arXiv.2506.13087). Crossref 404 is expected for this DataCite-registered DOI.

### 9. `yang2026mimik`

- Ordered authors: **Jiahao Yang; Shenhao Yan; Fan Feng; Chengsi Yao; Ge Wang; Zhixin Mai; Yiming Zhao; Yatong Han**.
- Exact title: *MimicIK: Real-Time Generative Inverse Kinematics from Teleoperation with FK Consistency*.
- **Preprint**, arXiv **2606.15148**, first submitted **13 June 2026**; latest displayed **v2, 16 June 2026**. DOI **10.48550/arXiv.2606.15148**. No final publication is identified in the inspected arXiv/DataCite records.
- Current BibTeX matches, including all eight authors. The local key `mimik` is merely an identifier; the title correctly says **MimicIK**.
- Relevance: strong online-execution match: current joints and target pose condition delta-joint prediction; the method targets smooth closed-loop behavior and uses FK consistency in training. The primary abstract reports a **10 mm** position-success threshold and 20 Hz deployment. These are its own settings, not evidence of meeting tighter position/orientation tolerances or a task-priority contract. FK regularization is not an exact output-level feasibility guarantee.
- Evidence: [arXiv abstract and version history](https://arxiv.org/abs/2606.15148), [DataCite record](https://api.datacite.org/dois/10.48550/arXiv.2606.15148).

### 10. `carvalho2025mpd`

- Ordered authors: **João Carvalho; An Thai Le; Piotr Kicki; Dorothea Koert; Jan Peters**.
- Exact final title: *Motion Planning Diffusion: Learning and Adapting Robot Motion Planning With Diffusion Models*.
- Final journal article: **IEEE Transactions on Robotics 41, 4881–4901 (2025)**. DOI **10.1109/TRO.2025.3593109**. Related preprint arXiv **2412.19948**, first posted December 2024, revised August 2025.
- Current BibTeX correctly cites the final 2025 article rather than the 2024 preprint. Do not substitute the later AAAI abstract reprint as the original publication.
- Relevance: trajectory-distribution priors, B-spline motion representations and cost-guided denoising for motion planning. This is useful general background on learned proposal distributions and differentiable refinement, but is **not a query-level online IK method or an interchangeable IK baseline**. Do not count its recency as justification for adding it to a tight IK related-work section.
- Evidence: [Crossref final record](https://api.crossref.org/works/10.1109/TRO.2025.3593109), [author preprint](https://arxiv.org/abs/2412.19948), [IEEE final article](https://doi.org/10.1109/TRO.2025.3593109) (indexed publisher abstract accessible; direct page required bot verification).

### 11. `huang2025diffusionseeder`

- Ordered authors: **Huang Huang; Balakumar Sundaralingam; Arsalan Mousavian; Adithyavairavan Murali; Ken Goldberg; Dieter Fox**.
- Exact title: *DiffusionSeeder: Seeding Motion Optimization with Diffusion for Rapid Motion Planning*.
- Final proceedings article: **Proceedings of The 8th Conference on Robot Learning**, **Proceedings of Machine Learning Research 270, 4392–4409 (2025)**. Publisher **PMLR**. Editors **Pulkit Agrawal; Oliver Kroemer; Wolfram Burgard**. No DOI is supplied by the official article citation.
- Current BibTeX matches the publisher's exported citation. **Event year is 2024**, held 6–9 November 2024; the volume was published 12 January 2025. Thus call it “CoRL 2024, proceedings published 2025” when discussing chronology; retaining the publisher's `year = {2025}` is justified for the proceedings record.
- Relevance: a diffusion model supplies trajectory seeds to cuRobo motion optimization from scene observations. This supports general learned-seeding motivation, not a direct online IK comparison. Planning latency and collision-free trajectory success cannot be treated as IK-query solve time and pose success.
- Evidence: [official article with complete BibTeX](https://proceedings.mlr.press/v270/huang25f.html), [official volume dates](https://proceedings.mlr.press/v270/).

## Optional BibTeX suggestions — not applied

No core metadata patch is required. The following are optional traceability/status improvements, not evidence of fabricated references:

```bibtex
% nguyen2026sequential: replace the unconfirmed time-sensitive status clause.
note = {Assigned to the November 2026 issue}

% zhang2025ikdiffuser: preserve the original preprint year; identify the version.
title = {{IKDiffuser}: a Diffusion-based Generative Inverse Kinematics Solver for Kinematic Trees}
note = {Preprint; first posted 16 June 2025; version 4 dated 14 January 2026; under review as listed on arXiv, checked 8 September 2026}

% yang2026mimik: optional explicit source/version fields.
url = {https://arxiv.org/abs/2606.15148}
note = {Preprint; version 2 dated 16 June 2026}

% demby2024learning: optional official conference-name expansion only.
booktitle = {2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}

% huang2025diffusionseeder: retain publisher year 2025; optional chronology note.
note = {CoRL 2024; proceedings published in 2025}
```

Keep article-number storage consistent with the repository's BibTeX style. Moving MDPI/Elsevier identifiers from `pages` into an `eid` field is a formatting choice that should be tested with that style; it is not necessary to correct the metadata.

## Selection and claim discipline

For a compact, execution-oriented discussion, SetIK and MimicIK are the most directly supported continuity-related entries in this batch; IKDiffuser and XGNN are useful for partial-goal/generative initialization and downstream numerical refinement. Nguyen sequential deserves priority for a primary-full-text check, not automatic detailed claims based on its promising title. Jayabalan offers a narrower local hybrid example. EMIKNet contributes multi-solution/multi-effector context. Demby remains valid learned-IK prior art, with the precise online input/timing contract still to inspect.

KineNN, MPD and DiffusionSeeder should not be used to pad recent-IK coverage. If retained, cite them only for their actual architectural/control or trajectory-planning contribution. None of the papers audited here can be called “contract-aligned with our benchmark” without a separate comparison of task constraints, error definitions, success thresholds, state/history access, candidate budgets, numerical refinement, termination, failure handling, and measured time boundary.
