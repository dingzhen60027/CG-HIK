# Recent IK reference verification — batch A

Checked 2026-09-08. Scope: 12 assigned keys in `paper/references.bib`, plus the separately requested PyRoki reference. This is a metadata and bounded relevance audit, not a benchmark or implementation evaluation. No IK, tests, benchmarks, installations, tracked-file changes, or manuscript edits were performed.

Method: read `nature-ref-verifier/SKILL.md` and its common-patterns reference; request each supplied DOI at Crossref before primary author/publisher sources. The browser could not open Crossref API URLs, but read-only `curl` requests obtained the records. Initial empty transient responses for GGIK and EAIK succeeded on retry. Nine supplied final-publication DOIs resolve to matching records. The three supplied arXiv DOIs return Crossref HTTP 404 because they are DataCite DOIs, not evidence of invalid papers; arXiv confirms each DOI. PyRoki was found by Crossref title lookup and then verified by its final DOI. Ordered author lists below retain the published form, including initials rather than inventing expanded middle names.

## Outcome and interpretation

- No critical DOI/title/author-identity mismatches were found in the 12 supplied entries.
- GeoFIK has a material status update: the coauthor's website says **ICRA 2026 accepted**. A final proceedings DOI/pages were not located in this bounded search; do not invent them or equate an acceptance statement with verified final proceedings metadata.
- PyRoki has verified final IROS 2025 metadata: DOI `10.1109/IROS60139.2025.11246651`, pp. 1312–1319.
- CppFlow's Crossref and author-site page span is anomalous (`12279–12785`), but the existing entry omits pages. Preserve the omission until the final printed pagination is checked; no replacement span is verified here.
- GGIK's 2025 volume year despite a 2024 DOI is correct. IKSel's final year is 2026 even though its stable citation key contains 2025. Zhang–Kikuuwe's final year is 2026 despite a 2025 DOI.
- Colan's bibliography is verified from Crossref plus an author-maintained bibliography, but direct IEEE abstract/full-text access was blocked; its relevance sentence is deliberately title-level and not a claimed full-text verification.

For task-contract positioning, **fixed solver acceptance thresholds, optimization costs, hard physical constraints, and user-specified task-tolerance sets are distinct**. The primary evidence below supports the stated limited connection; it does not establish that every cited solver implements ranged or set-valued task contracts.

## 1. `morgan2024cppflow`

- Title: **CppFlow: Generative Inverse Kinematics for Efficient and Robust Cartesian Path Planning**.
- Ordered authors: Jeremy Morgan; David Millard; Gaurav S. Sukhatme.
- Venue/year/status: *2024 IEEE International Conference on Robotics and Automation (ICRA)*, 2024; final proceedings article, not merely its 2023 preprint.
- DOI: `10.1109/ICRA57147.2024.10611724`; preprint `arXiv:2309.09102` (2023-09-16; v2 2024-06-03).
- Verification: supplied title, authors, venue, year, DOI match. Crossref and the author site both deposit `12279–12785`; that unusually long span is unresolved, not silently corrected.
- Relevance: Its author project defines valid Cartesian trajectories using less than 1 mm / 0.1° pose error plus other constraints, making it relevant to tolerance-conditioned time-to-valid-solution reporting, but those fixed acceptance limits do not by themselves establish a general task-tolerance formulation.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/ICRA57147.2024.10611724); [author project and citation](https://jstmn.github.io/cppflow-website/); [author preprint](https://arxiv.org/abs/2309.09102).
- Confidence: **Verified identity/publication; check suggested for pagination**.

## 2. `limoyo2025ggik`

- Title: **Generative Graphical Inverse Kinematics**.
- Ordered authors: Oliver Limoyo; Filip Marić; Matthew Giamou; Petra Alexson; Ivan Petrović; Jonathan Kelly.
- Venue/year/status: *IEEE Transactions on Robotics* **41**, 1002–1018 (2025); final journal article.
- DOI: `10.1109/TRO.2024.3521862`; preprint `arXiv:2209.08812` (first 2022-09-19; v5 2025-01-23).
- Verification: supplied fields agree; arXiv explicitly links this DOI and the 2025 final journal reference. The DOI's 2024 component is not the volume year.
- Relevance: GGIK learns joint-limit-aware solution distributions for diverse robots and supplies initializations to local optimization, supporting constraint-aware candidate generation and seed-quality context rather than proving user-defined task-tolerance guarantees.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/TRO.2024.3521862); [author preprint with final journal reference](https://arxiv.org/abs/2209.08812).
- Confidence: **Verified**.

## 3. `colan2024variablestep`

- Title: **Variable Step Sizes for Iterative Jacobian-Based Inverse Kinematics of Robotic Manipulators**.
- Ordered authors: Jacinto Colan; Ana Davila; Yasuhisa Hasegawa.
- Venue/year/status: *IEEE Access* **12**, 87909–87922 (2024); final journal article.
- DOI: `10.1109/ACCESS.2024.3418206`.
- Verification: all supplied bibliographic fields agree with Crossref and the first author's maintained bibliography. IEEE document `10568941` returned a JavaScript/bot-verification page; direct publisher abstract/full text was not verified. Secondary abstract mirrors were encountered but are not used to substantiate technical details here.
- Relevance: The verified title establishes variable step-size design for iterative Jacobian IK as a numerical-convergence comparator, but detailed constraint handling and tolerance semantics remain unverified in this audit.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/ACCESS.2024.3418206); [author-maintained bibliography, `colan2024variable`](https://raw.githubusercontent.com/jcolan/jcolan.github.io/master/files/biblio.bib); [publisher target—access blocked](https://ieeexplore.ieee.org/document/10568941/).
- Confidence: **Verified bibliography; technical relevance limited to title-level evidence**.

## 4. `elias2025ikgeo`

- Title: **IK-Geo: Unified robot inverse kinematics using subproblem decomposition**.
- Ordered authors: Alexander J. Elias; John T. Wen.
- Venue/year/status: *Mechanism and Machine Theory* **209**, article 105971 (July 2025); final journal article.
- DOI: `10.1016/j.mechmachtheory.2025.105971`; preprint `arXiv:2211.05737` (first 2022-11-10).
- Verification: supplied title, author order, year, volume, article number, DOI agree; title capitalization is immaterial. The author's publication page independently lists the final journal metadata.
- Relevance: Geometric subproblem decomposition provides fast IK with singular solutions and some least-squares extensions, making it relevant to online candidate enumeration and singularity robustness, not evidence of an application-level task-tolerance contract.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1016/j.mechmachtheory.2025.105971); [author publication page](https://alexanderelias.com/); [author preprint](https://arxiv.org/abs/2211.05737).
- Confidence: **Verified**.

## 5. `zhang2026viability`

- Title: **Ensuring Viability: A QP-based Inverse Kinematics for Handling Joint Range, Velocity and Acceleration Limits, as Well as Whole-body Collision Avoidance**.
- Ordered authors: Yachen Zhang; Ryo Kikuuwe.
- Venue/year/status: *Journal of Intelligent & Robotic Systems* **112**(1), article 16 (2026); final journal article.
- DOI: `10.1007/s10846-025-02335-z`.
- Verification: supplied fields agree; `16` is an article number, not a page span. Publisher says published 2026-01-05, version of record 2026-01-20; Crossref gives online publication 2026-01-05. The 2025 DOI component is not a year error.
- Relevance: This is directly relevant to online physical-constraint feasibility: an offline-constructed viability constraint is updated during online QP IK to retain feasible solutions under joint range/rate/acceleration and whole-body collision constraints.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1007/s10846-025-02335-z); [publisher article](https://link.springer.com/article/10.1007/s10846-025-02335-z).
- Confidence: **Verified**.

## 6. `yuan2025iksel`

- Title: **IKSel: Selecting Good Seed Joint Values for Fast Numerical Inverse Kinematics Iterations**.
- Ordered authors: Xinyi Yuan; Weiwei Wan; Kensuke Harada.
- Venue/year/status: *IEEE Transactions on Automation Science and Engineering* **23**, 4410–4427 (2026); final journal article.
- DOI: `10.1109/TASE.2026.3659225`; preprint `arXiv:2503.22234` (2025-03-28).
- Verification: supplied final fields agree. Keep final citation year 2026; a BibTeX key is an internal identifier and its `2025` does not require changing the publication year. The arXiv page itself still shows only the 2025 preprint.
- Relevance: KD-tree candidate retrieval, joint-adjustment ranking and seed reselection address fast numerical convergence and failures involving local minima or joint limits, supporting online seed selection rather than an explicit tolerance-set task model.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/TASE.2026.3659225); [author preprint](https://arxiv.org/abs/2503.22234).
- Confidence: **Verified**.

## 7. `ostermeier2025eaik`

- Title: **Automatic Geometric Decomposition for Analytical Inverse Kinematics**.
- Ordered authors: Daniel Ostermeier; Jonathan Külz; Matthias Althoff.
- Venue/year/status: *IEEE Robotics and Automation Letters* **10**(10), 9964–9971 (October 2025); final journal article.
- DOI: `10.1109/LRA.2025.3597897`; preprint `arXiv:2409.14815` (2024-09-23; v2 2025-08-21).
- Verification: supplied fields agree with Crossref; the author preprint and institutional author-version record corroborate title/order and final venue context.
- Relevance: Automatic geometric classification and decomposition reduce analytical-IK derivation and per-query work, providing an online efficiency baseline, while the inspected abstract does not establish arbitrary physical-constraint or task-tolerance support.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/LRA.2025.3597897); [author preprint](https://arxiv.org/abs/2409.14815); [TUM author-version record](https://mediatum.ub.tum.de/doc/1796718/xyfkst1ciei6yv4j19g2021ji.pdf).
- Confidence: **Verified** (no experimental speed claims independently reproduced).

## 8. `lopezcustodio2025geofik`

- Title: **GeoFIK: A Fast and Reliable Geometric Solver for the IK of the Franka Arm based on Screw Theory Enabling Multiple Redundancy Parameters**.
- Ordered authors: Pablo C. Lopez-Custodio; Yuhe Gong; Luis F. C. Figueredo.
- Venue/year/status: verified arXiv preprint, first 2025-03-06, v2 2026-03-16; **ICRA 2026 accepted according to coauthor Yuhe Gong's publication page**. Final proceedings DOI/pages not verified.
- DOI: `10.48550/arXiv.2503.03992` (DataCite; Crossref 404 is expected).
- Verification: supplied preprint fields agree, but its note omits the author-reported conference acceptance. A bounded Crossref title lookup returned no matching final record. Nottingham's search-indexed repository also describes an ICRA 2026 contribution, but its direct page failed, so it is not counted as accessible corroboration.
- Relevance: Franka-specific screw-theoretic IK with multiple redundancy variables and singularity handling supports fast posture/redundancy selection, rather than directly defining a task-tolerance region.
- Sources: [arXiv record](https://arxiv.org/abs/2503.03992); [coauthor publication page, GeoFIK entry](https://yuhegong.github.io/).
- Confidence: **Check suggested—update status cautiously; retain verified preprint identifier pending final metadata**.

## 9. `boschi2026singularity`

- Title: **Handling Transitions Across Singularities for UR-Like Serial Robots**.
- Ordered authors: Ivan Boschi; Alessandro De Toni; Roberto Di Leva; Edoardo Idà; Marco Carricato.
- Venue/year/status: *IEEE Robotics and Automation Letters* **11**(3), 2714–2721 (March 2026); final journal article.
- DOI: `10.1109/LRA.2026.3653295`.
- Verification: title, ordered identities, final venue/year/pages and DOI agree with Crossref and Bologna's repository. Crossref parses `Roberto Di` as given name and `Leva` as family, whereas Bologna identifies the family as `Di Leva`; preserve the supplied BibTeX `Di Leva, Roberto`. `Idà` / `Ida'` / unaccented `Ida` are source spelling variants, not missing authors. Direct IEEE page was blocked, but its indexed text and the accessible institutional abstract were available.
- Relevance: Branch switching across singularities preserves continuous, differentiable joint trajectories for UR-like robots, supporting trajectory-continuity discussion without establishing feasibility of arbitrary singularity-crossing task paths.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/LRA.2026.3653295); [Bologna publication and abstract](https://cris.unibo.it/handle/11585/1039115); [Di Leva institutional identity](https://cris.unibo.it/cris/rp/rp74211).
- Confidence: **Verified, with harmless name-normalization caveats**.

## 10. `yasutake2026hjcdik`

- Title: **HJCD-IK: GPU-Accelerated Inverse Kinematics through Batched Hybrid Jacobian Coordinate Descent**.
- Ordered authors: Cael Yasutake; Andrew H. Liu; Zachary Kingston; Brian Plancher.
- Venue/year/status: arXiv preprint first posted 2025-10-08, v2 2026-07-06; arXiv comments explicitly state **accepted to IROS 2026**. Final proceedings DOI/pages not verified.
- DOI: `10.48550/arXiv.2510.07514` (DataCite; Crossref 404 is expected).
- Verification: supplied title/authors/preprint year and acceptance note agree with arXiv; the key's 2026 and preprint year 2025 have different meanings. A supplemental Crossref title query gave an empty transport response, so no negative final-publication claim follows from that query.
- Relevance: Batched coordinate-descent initialization, Jacobian polishing and parallel collision filtering target the accuracy–latency trade-off for collision-free IK samples, making it a pertinent online-IK comparator without equating a Pareto frontier with task-level tolerance guarantees.
- Sources: [arXiv record and acceptance comment](https://arxiv.org/abs/2510.07514).
- Confidence: **Verified preprint metadata and author-reported acceptance; final publication unresolved**.

## 11. `wu2024ikspark`

- Title: **IKSPARK: Obstacle-Aware Inverse Kinematics via Convex Optimization**.
- Ordered authors: Liangting Wu; Roberto Tron.
- Venue/year/status: arXiv preprint first posted 2024-03-18, v2 2026-04-30; no final venue/DOI listed in the inspected arXiv record or located in the bounded title search.
- DOI: `10.48550/arXiv.2403.12235` (DataCite; Crossref 404 is expected).
- Verification: supplied current title, author order, preprint year and 2026 revision note agree. This audit does not assert that no final publication exists; only that one was not verified.
- Relevance: Semidefinite relaxation with rank recovery and convexified obstacle constraints provides constrained-IK and infeasibility-certificate context, but relaxation infeasibility certification must not be conflated with a universal online deadline or task-tolerance guarantee.
- Sources: [arXiv record and abstract](https://arxiv.org/abs/2403.12235).
- Confidence: **Verified preprint record; final venue unresolved**.

## 12. `tang2025etaik`

- Title: **ETA-IK: Execution-Time-Aware Inverse Kinematics for Dual-Arm Systems**.
- Ordered authors: Yucheng Tang; Xi Huang; Yongzhou Zhang; Tao Chen; Ilshat Mamaev; Björn Hein.
- Venue/year/status: *2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 7504–7511 (2025); final proceedings article.
- DOI: `10.1109/IROS60139.2025.11247583`; preprint `arXiv:2411.14381` (first 2024-11-21).
- Verification: all supplied bibliographic fields agree with Crossref; the lead author's publication page independently confirms the IROS 2025 venue and complete author list.
- Relevance: ETA-IK exploits dual-arm redundancy under a relative-pose constraint to minimize predicted motion execution time while implicitly considering collisions, so it is task-structure-aware but its execution-time objective is distinct from IK solver latency.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/IROS60139.2025.11247583); [author preprint](https://arxiv.org/abs/2411.14381); [lead-author publication page](https://yucheng-tang.github.io/).
- Confidence: **Verified**.

## 13. Additional requested reference: `kim2025pyroki` (suggested key)

- Title: **PyRoki: A Modular Toolkit for Robot Kinematic Optimization**.
- Ordered authors: Chung Min Kim; Brent Yi; Hongsuk Choi; Yi Ma; Ken Goldberg; Angjoo Kanazawa. The official site and arXiv identify the first two authors as equal contributors; asterisks are contribution markers, not part of their names.
- Venue/year/status: *2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, **1312–1319** (2025); final proceedings article.
- Final DOI: **`10.1109/IROS60139.2025.11246651`**. Preprint: `arXiv:2505.03728`, first posted 2025-05-06, DOI `10.48550/arXiv.2505.03728`.
- Verification: Crossref exact-title search found the final record; a subsequent request to that exact DOI resolved with matching full ordered authors/title/venue/pages. Official project citation confirms IROS 2025, although it omits final DOI/pages.
- Relevance: Composable kinematic variables and costs for pose, speed, collisions and motion retargeting make PyRoki pertinent to task-conditioned optimization design across CPU/GPU/TPU, but modular costs alone do not demonstrate hard satisfaction of arbitrary tolerance sets.
- Sources: [Crossref final DOI record](https://api.crossref.org/works/10.1109/IROS60139.2025.11246651); [official project and citation](https://pyroki-toolkit.github.io/); [author preprint](https://arxiv.org/abs/2505.03728).
- Confidence: **Verified final publication**.

## Remaining bounded-audit limitations

1. CppFlow's true printed page interval was not checked against the final PDF; preserve uncertainty despite matching deposited records.
2. GeoFIK and HJCD-IK acceptance is verified as an author statement, not final proceedings pagination/DOI. GeoFIK's supplied status note is incomplete.
3. IKSPARK remains a verified preprint for this audit; no final venue was established.
4. Colan's direct primary abstract/full text was unavailable, so no detailed claims about its experimental setup, stopping tolerances, or hard-constraint guarantees should be derived from this report.
5. These are source-grounded relevance summaries, not independent reproduction of any accuracy, latency, success-rate, completeness, or safety claim.
