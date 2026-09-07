# Reference verification: final task-contract manuscript

Checked 2026-09-08. Final bibliography: `paper/references.bib`.
There are 38 retained entries, 22 with bibliographic years 2024–2026. All are cited.
Four are explicitly cited as preprints rather than silently promoted to final
proceedings. The count uses bibliographic year, not the year embedded in a DOI.

## Verification procedure and meaning

Ordered authors, title, venue, year, DOI/official URL, volume and available pages
were compared with DOI-registry records and publisher, proceedings, author or
institutional sources. The archived batch notes distinguish a live primary page
from an indexed primary snapshot or inaccessible direct page. A resolver failure
was not treated as a nonexistent paper; arXiv DOIs were checked through DataCite.

Metadata verification is not a claim that every cited paper was reproduced, uses
the present contract, or supplies a directly comparable timing benchmark. Detailed
method claims in the manuscript are limited to the primary material inspected.
The paper makes no novelty claim about inventing tolerance-aware tasks or native
solver bounds.

## Coverage

The preserved notes in `reference_verification/task_contract_2026/` document
source URLs and field comparisons. Some notes include rejected candidates; the
following list defines the final retained set.

| Note | Retained keys |
|---|---|
| `verify_foundations.md` | `wampler1986dls`, `nakamura1986singularity`, `chiaverini1994review`, `beeson2015tracik`, `rakita2018relaxedik`, `ames2022ikflow`, `habekost2023cycleik`, `rice1976selection`, `siciliano1990tutorial`, `bischl2016aslib`, `virtanen2020scipy`, `paszke2019pytorch`, `pedregosa2011sklearn` |
| `verify_recent_a.md` | `morgan2024cppflow`, `limoyo2025ggik`, `colan2024variablestep`, `elias2025ikgeo`, `zhang2026viability`, `yuan2025iksel`, `ostermeier2025eaik`, `lopezcustodio2025geofik`, `boschi2026singularity`, `yasutake2026hjcdik`, `wu2024ikspark`, `tang2025etaik` |
| `verify_recent_b.md` | `jayabalan2026hybrid`, `demby2024learning`, `nguyen2026setik`, `yang2026mimik` |
| New-entry checks below | `moe2016setbased`, `berenson2011tsr`, `schuetz2014predictive`, `gamper2024switching`, `wolinski2025predictive`, `origanti2025lookahead`, `wingo2024linear`, `wolinski2024scaling`, `kim2025pyroki` |

The first eight new entries have complete fetched Crossref responses in
`new_refs_crossref.json`. PyRoki's final proceedings fields are corroborated by its
official project citation and final DOI. These are publication checks only; none
triggered an experimental run or baseline change.

## New-entry comparisons

| Key / ordered authors | Final bibliographic record | Primary corroboration and relevance |
|---|---|---|
| `moe2016setbased`: Signe Moe; Gianluca Antonelli; Andrew R. Teel; Kristin Y. Pettersen; Johannes Schrimpf | Frontiers in Robotics and AI 3, 16 (2016); DOI 10.3389/frobt.2016.00016 | [Published article](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00016/full): set/equality tasks and task-priority IK, not a new concept of this manuscript |
| `berenson2011tsr`: Dmitry Berenson; Siddhartha Srinivasa; James Kuffner | IJRR 30(12), 1435–1460 (2011); DOI 10.1177/0278364910396389 | [Publisher](https://journals.sagepub.com/doi/10.1177/0278364910396389): Task Space Regions and pose-constrained planning. Registry short title omits the subtitle; publisher title includes it |
| `schuetz2014predictive`: Christoph Schuetz; Thomas Buschmann; Joerg Baur; Julian Pfaff; Heinz Ulbrich | ICRA 2014, 5056–5061; DOI 10.1109/ICRA.2014.6907600 | [Author institution](https://portal.fis.tum.de/en/publications/predictive-online-inverse-kinematics-for-redundant-manipulators/): predictive online IK, no claim here of first preview |
| `gamper2024switching`: Hannes Gamper; Laura Rodrigo Pérez; Andreas Mueller; Alejandro Díaz Rosales; Mario Di Castro | RA-L 9(5), 4527–4534 (2024); DOI 10.1109/LRA.2024.3379860 | [Institution record](https://repository.tudelft.nl/record/uuid%3A0cc303bc-a5b0-429e-b48f-85ff89bd63a6): task priority/optimization switching |
| `wolinski2025predictive`: Łukasz Woliński; Marek Wojtyra | Mechanism and Machine Theory 209, 105988 (2025); DOI 10.1016/j.mechmachtheory.2025.105988 | [Publisher](https://www.sciencedirect.com/science/article/pii/S0094114X25000771): predictive QP and trajectory scaling; unlike fixed target timing here |
| `origanti2025lookahead`: Vamsi Krishna Origanti; Adrian Danzglock; Frank Kirchner | SII 2025, 990–997; DOI 10.1109/SII59315.2025.10871056 | [Author institution](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/15449): look-ahead nullspace management in dual-arm impedance control |
| `wingo2024linear`: Bruce Wingo; Ajay Suresha Sathya; Stéphane Caron; Seth Hutchinson; Justin Carpentier | RSS XX (2024); DOI 10.15607/RSS.2024.XX.110 | [Proceedings paper](https://roboticsproceedings.org/rss20/p110.pdf): linear-time differential IK, augmented-Lagrangian formulation |
| `wolinski2024scaling`: Łukasz Woliński; Marek Wojtyra | Mechanism and Machine Theory 191, 105493 (2024); DOI 10.1016/j.mechmachtheory.2023.105493 | [Publisher DOI](https://doi.org/10.1016/j.mechmachtheory.2023.105493) and Crossref: trajectory scaling for redundant manipulators. DOI year 2023 and volume year 2024 are compatible; no detailed performance comparison is asserted |
| `kim2025pyroki`: Chung Min Kim; Brent Yi; Hongsuk Choi; Yi Ma; Ken Goldberg; Angjoo Kanazawa | IROS 2025, 1312–1319; DOI 10.1109/IROS60139.2025.11246651 | [Official project/citation](https://pyroki-toolkit.github.io/): modular kinematic optimization toolkit; final proceedings cited instead of replacing them with the preprint |

## Corrections and explicit qualifications

- SciPy's full author field follows the official project citation: 34 named authors
  and the consortium. Crossref's expanded consortium-member metadata is not
  appended as if it were a second set of top-level authors.
- IKSel is a final 2026 article despite its historical local key containing 2025.
  Several DOI strings likewise contain a year preceding final journal publication.
- CppFlow's paper identity, authors, ICRA venue and DOI are verified. The deposited
  page range is inconsistent enough that pages are omitted rather than guessed.
- GeoFIK is cited by its verified preprint, with author-reported ICRA acceptance
  distinguished from final proceedings metadata that was not verified.
- HJCD-IK is cited by its first-posted preprint year and explicit IROS acceptance.
  IKSPARK and MimicIK also retain preprint status and official arXiv links.
- The two software records without DOI in official supplied citations (PyTorch and
  scikit-learn) retain their official proceedings/journal URLs; no DOI is invented.
- Wampler's exact middle initial/suffix has less direct byline coverage than the
  paper identity; existing `Charles W. Wampler II` is retained, corroborated by
  bibliographic indexes, not falsely described as re-read on the original first
  page. Rice's optional editor/imprint detail has similarly weaker direct-primary
  coverage, disclosed in the foundation note.
- Chiaverini's expanded author names were subsequently checked on the
  [original paper's first-page facsimile](https://stephanniec.github.io/stepholio/files/leastsqrinvkin.pdf):
  Stefano Chiaverini, Bruno Siciliano and Olav Egeland. This closes the byline
  qualification raised in the earlier batch note.
- Metadata-complete but less relevant generative motion-planning papers were not
  retained merely to increase the recent-reference count. No table equates their
  task requirements or runtime boundary with this study.

The generated `paper/generated/reference_inventory.json` records the exact final
BibTeX fields and hash. It is an index of the checked bibliography, not an
independent metadata source or proof of full-text review.
