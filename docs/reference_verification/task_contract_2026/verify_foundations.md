# Foundation-reference verification (13 entries)

Audit date: 2026-09-08. Input: `paper/references.bib`. This is preparatory literature work, not a manuscript revision. No original bibliography, manuscript, tracked code, experiments, tests, installations, or historical outputs were changed.

## Outcome and evidence standard

- All 11 DOI-bearing entries returned Crossref HTTP 200 and resolved to the intended paper/chapter. TRAC-IK and CycleIK initially returned HTTP 429; subsequent requests succeeded. No DOI/title substitution was found.
- The two URL-only entries, PyTorch and scikit-learn, match their official proceedings/journal pages and official BibTeX exports. Those exports give no DOI. This is **not** proof that no DOI exists anywhere; do not invent one or substitute an arXiv DOI for the published item.
- The principal completeness suggestion is the SciPy `author` field: the current `and others` deliberately abbreviates the record. The official project citation has 34 named authors followed by the group author `SciPy 1.0 Contributors`. Preserve that byline structure rather than blindly flattening Crossref's additional consortium-member records.
- TRAC-IK's official conference title includes `(Humanoids)`; adding it is optional metadata normalization, not a different venue. Case-only title differences are not substantive errors.
- Wampler's middle initial/suffix and Chiaverini et al.'s expanded given names have weaker direct-primary coverage in this audit. Retain the existing names pending original-byline confirmation; do not turn incomplete Crossref given names into a supposed correction.

`Crossref verified` below means the actual DOI REST endpoint was fetched and its metadata compared. `Primary page verified` means the named live publisher/author page was successfully opened and contained the relevant fields. `Indexed primary snapshot` means a search response supplied text from the publisher/author URL, but a direct open was inaccessible or did not expose the old record. An inaccessible URL is never counted as live verification. Full-text scientific claims were not audited here.

## Per-entry records

### 1. `wampler1986dls` — Check suggested: expanded author name

- Title: *Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods*.
- Ordered author: **Charles W. Wampler II** (current bibliography); Crossref registers only **Charles Wampler**. The middle initial and suffix are not independently confirmed from a live publisher/author byline in this audit.
- Venue/year: *IEEE Transactions on Systems, Man, and Cybernetics* **16**(1), 93–101 (1986). Crossref date is January 1986; an indexed author publication list uses Jan.–Feb. 1986 and `SMC-16`, compatible with the current numeric volume.
- DOI: `10.1109/TSMC.1986.289285`, Crossref HTTP 200, intended title matched.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/TSMC.1986.289285); [IEEE publisher target](https://ieeexplore.ieee.org/document/4075580/) opened but exposed no usable metadata; [author publication list](https://www3.nd.edu/~cwample1/publist.pdf), indexed primary snapshot only (direct requests returned 404), lists C. W. Wampler, title, volume, issue, pages, and year.
- Recommendation: retain current bibliographic fields and valid suffix syntax `Wampler, II, Charles W.`; no automatic author shortening. An original first page or accessible author archive is needed to close the exact suffix/byline check.

### 2. `nakamura1986singularity` — Crossref verified; publisher access limited

- Title: *Inverse Kinematic Solutions With Singularity Robustness for Robot Manipulator Control*.
- Ordered full authors: **Yoshihiko Nakamura; Hideo Hanafusa**.
- Venue/year: *Journal of Dynamic Systems, Measurement, and Control* **108**(3), 163–171 (1986). Crossref print/online date: 1986-09-01.
- DOI: `10.1115/1.3143764`, HTTP 200, all current core fields match.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1115/1.3143764); [ASME publisher target](https://asmedigitalcollection.asme.org/dynamicsystems/article/108/3/163/425826/Inverse-Kinematic-Solutions-With-Singularity), inaccessible with HTTP 403. A successfully opened [Japan Academy institutional publication list](https://www.japan-acad.go.jp/pdf/youshi/115/nakamura_yoshihiko.pdf), page 4, entry 3, independently corroborates author initials/order, exact title, venue, 108(3), 163–171, and 1986. It is institutional corroboration, not a live ASME byline inspection.
- Recommendation: retain current entry. No author-count or DOI mismatch is supported by the sources checked.

### 3. `chiaverini1994review` — Check suggested: full given-name coverage

- Title: *Review of the Damped Least-Squares Inverse Kinematics with Experiments on an Industrial Robot Manipulator*.
- Ordered full authors retained from input: **Stefano Chiaverini; Bruno Siciliano; Olav Egeland**. Crossref gives **S. Chiaverini; B. Siciliano; O. Egeland**, in that order.
- Venue/year: *IEEE Transactions on Control Systems Technology* **2**(2), 123–134 (1994). Crossref print date: June 1994.
- DOI: `10.1109/87.294335`, HTTP 200, title/order/core publication fields match.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/87.294335); [IEEE publisher target](https://ieeexplore.ieee.org/document/294335/) exposes no usable metadata. Indexed author-institution snapshots at [Naples repository](https://www.iris.unina.it/handle/11588/497713) and [Cassino repository](https://iris.unicas.it/handle/11580/8459) corroborate the record and expand Bruno and Stefano respectively; direct opens returned 403. The indexed [PRISMA author-lab publication list](https://prisma.dieti.unina.it/index.php/publications/intn-l-journal-papers) corroborated the 1994 citation with initials, but the live extracted page did not include its historical 1994 section. These are not counted as live byline verification.
- An indexed reproduction of the published first page at a non-author GitHub host displayed all three full names, but this was not used to close the requested publisher/author-source verification standard.
- Recommendation: retain current entry. Exact expanded names, especially Olav, remain a direct-primary byline recheck, not an identified error.

### 4. `beeson2015tracik` — Verified core metadata

- Title: *TRAC-IK: An Open-Source Library for Improved Solving of Generic Inverse Kinematics*.
- Ordered full authors: **Patrick Beeson; Barrett Ames** (Crossref).
- Venue/year: *2015 IEEE-RAS 15th International Conference on Humanoid Robots (Humanoids)*, 928–935 (2015), IEEE. Crossref date November 2015. Indexed IEEE metadata gives conference dates 3–5 November 2015, Seoul, South Korea; Xplore-added date 28 December 2015 is not the conference date.
- DOI: `10.1109/HUMANOIDS.2015.7363472`, initially HTTP 429 then HTTP 200, intended paper matched.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/HUMANOIDS.2015.7363472); [IEEE publisher record](https://ieeexplore.ieee.org/document/7363472/) indexed primary metadata; live [TRACLabs CRAFTSMAN publication list](https://traclabs.com/projects/craftsman/) corroborates P. Beeson, B. Ames, title, Humanoids, and 2015. The live [TRAC-IK project](https://traclabs.com/projects/trac-ik/) points to the author publication page, but [that author page](https://personal.traclabs.com/~pbeeson/publications/b2hd-Beeson-humanoids-15.html) was inaccessible (TLS EOF in direct request).
- Recommendation: no substantive fix. Optionally append `(Humanoids)` to `booktitle` to match Crossref/IEEE exactly.

### 5. `rakita2018relaxedik` — Verified

- Title: *RelaxedIK: Real-time Synthesis of Accurate and Feasible Robot Arm Motion*.
- Ordered full authors: **Daniel Rakita; Bilge Mutlu; Michael Gleicher**.
- Venue/year: *Robotics: Science and Systems XIV* (2018), Robotics: Science and Systems Foundation. Official BibTeX uses *Proceedings of Robotics: Science and Systems*, Pittsburgh, Pennsylvania, June 2018. Crossref online date: 2018-06-26. No page range supplied by the records checked.
- DOI: `10.15607/RSS.2018.XIV.043`, HTTP 200, match.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.15607/RSS.2018.XIV.043); live [official RSS proceedings and BibTeX](https://www.roboticsproceedings.org/rss14/p43.html).
- Recommendation: retain current entry. `Real-Time` versus `Real-time` is capitalization only; no pagination should be invented from the article identifier.

### 6. `ames2022ikflow` — Verified

- Title: *IKFlow: Generating Diverse Inverse Kinematics Solutions*.
- Ordered full authors: **Barrett Ames; Jeremy Morgan; George Konidaris**.
- Venue/year: *IEEE Robotics and Automation Letters* **7**(3), 7177–7184 (2022). Crossref print date: July 2022.
- DOI: `10.1109/LRA.2022.3181374`, HTTP 200, all current fields match.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1109/LRA.2022.3181374); live [author project page](https://sites.google.com/view/ikflow/home) verifies title/full author order. [Author-maintained PyPI project citation](https://pypi.org/project/ikflow/0.0.6/) indexed snapshot supplies the same journal, volume, issue, pages, year, DOI, and ordered authors. [IEEE publisher target](https://ieeexplore.ieee.org/document/9793576/) is the Crossref-declared target; its full live metadata was not independently inspected.
- Recommendation: retain the 2022 published-journal citation; [arXiv first posting in 2021](https://arxiv.org/abs/2111.08933) is not a reason to change the journal year.

### 7. `habekost2023cycleik` — Verified

- Title: *CycleIK: Neuro-inspired Inverse Kinematics*.
- Ordered full authors: **Jan-Gerrit Habekost; Erik Strahl; Philipp Allgeuer; Matthias Kerzel; Stefan Wermter**.
- Venue/year: chapter in *Artificial Neural Networks and Machine Learning – ICANN 2023*, *Lecture Notes in Computer Science* **14254**, 457–470 (2023), Springer. Live publisher page first-online date: 22 September 2023. All input fields match apart from title-case styling.
- DOI: `10.1007/978-3-031-44207-0_38`, initially HTTP 429 then HTTP 200, intended chapter matched.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1007/978-3-031-44207-0_38); live [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-44207-0_38), including full five-author list, LNCS volume, pages, and date.
- Recommendation: retain current entry. No need to change the appropriate `@incollection` type merely because the publisher also labels it a conference paper.

### 8. `rice1976selection` — Crossref verified; primary snapshot corroboration

- Title: *The Algorithm Selection Problem*.
- Ordered full author: **John R. Rice**.
- Venue/year: chapter in *Advances in Computers*, volume **15**, 65–118 (1976). Crossref publisher: Elsevier; input imprint: Academic Press.
- DOI: `10.1016/S0065-2458(08)60520-3`, HTTP 200, intended title matched. The `(08)` embedded in the DOI is **not** a publication-year correction; Crossref and the publisher-index snapshot both give 1976.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1016/S0065-2458(08)60520-3); [ScienceDirect chapter](https://www.sciencedirect.com/science/article/pii/S0065245808605203), indexed primary snapshot confirms John R. Rice, title, volume 15, 65–118, 1976, and DOI; direct open returned 403.
- Unresolved optional fields: the input editors **Morris Rubinoff; Marshall C. Yovits** and exact original **Academic Press** imprint were not supplied by the fetched chapter Crossref record or the accessible indexed chapter text. They are retained as input, not independently verified here. A volume-title-page or official volume record would close those fields.
- Recommendation: no change to core citation. Do not replace 1976 with a DOI-string year or delete plausible editor/imprint fields solely because the chapter registry omitted them.

### 9. `siciliano1990tutorial` — Verified

- Title: *Kinematic Control of Redundant Robot Manipulators: A Tutorial*.
- Ordered full author: **Bruno Siciliano**.
- Venue/year: *Journal of Intelligent and Robotic Systems* **3**(3), 201–212 (1990); publisher issue date September 1990. The current publisher site also styles the journal name with `&`; this is not a venue mismatch.
- DOI: `10.1007/BF00126069`, HTTP 200, match.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1007/BF00126069); live [Springer article](https://link.springer.com/article/10.1007/BF00126069) confirms title, full author, volume, pages, year, September issue date, and DOI. Issue number 3 is confirmed by Crossref.
- Recommendation: retain current entry.

### 10. `bischl2016aslib` — Crossref verified; primary snapshot corroboration

- Title: *ASlib: A Benchmark Library for Algorithm Selection*.
- Ordered full authors: **Bernd Bischl; Pascal Kerschke; Lars Kotthoff; Marius Lindauer; Yuri Malitsky; Alexandre Fréchette; Holger Hoos; Frank Hutter; Kevin Leyton-Brown; Kevin Tierney; Joaquin Vanschoren**.
- Venue/year: *Artificial Intelligence* **237**, 41–58 (2016), Elsevier. Crossref print date: August 2016.
- DOI: `10.1016/j.artint.2016.04.003`, HTTP 200. All current core fields, including all 11 authors and their order, match.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1016/j.artint.2016.04.003); [ScienceDirect publisher article](https://www.sciencedirect.com/science/article/pii/S0004370216300388), indexed primary snapshot independently supplies the complete author list, venue, volume, pages, August 2016, and DOI. Direct opening returned 403; no live full-text inspection is claimed.
- Recommendation: retain current entry.

### 11. `virtanen2020scipy` — Verified identity; completeness suggestion

- Title: *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*.
- Complete ordered citation byline (official SciPy citation; 34 named authors plus group): **Pauli Virtanen; Ralf Gommers; Travis E. Oliphant; Matt Haberland; Tyler Reddy; David Cournapeau; Evgeni Burovski; Pearu Peterson; Warren Weckesser; Jonathan Bright; Stéfan J. van der Walt; Matthew Brett; Joshua Wilson; K. Jarrod Millman; Nikolay Mayorov; Andrew R. J. Nelson; Eric Jones; Robert Kern; Eric Larson; C J Carey; İlhan Polat; Yu Feng; Eric W. Moore; Jake VanderPlas; Denis Laxalde; Josef Perktold; Robert Cimrman; Ian Henriksen; E. A. Quintero; Charles R. Harris; Anne M. Archibald; Antônio H. Ribeiro; Fabian Pedregosa; Paul van Mulbregt; SciPy 1.0 Contributors**.
- Venue/year: *Nature Methods* **17**(3), 261–272 (2020). Crossref online date 3 February 2020; print issue date 2 March 2020. `2019` in the DOI is not the publication year.
- DOI: `10.1038/s41592-019-0686-2`, HTTP 200, intended paper matched.
- Sources: [Crossref DOI record](https://api.crossref.org/works/10.1038/s41592-019-0686-2); live [official SciPy citation and BibTeX](https://scipy.org/citing-scipy/). The [original Nature article](https://www.nature.com/articles/s41592-019-0686-2) was available as an indexed publisher snapshot but direct opening failed at a cookie/identity redirect. The live [Nature author-correction page](https://www.nature.com/articles/s41592-020-0772-5) independently shows the same complete byline and distinguishes consortium membership. The correction concerns correspondence, affiliation, and text details; it is not a reason to substitute the correction DOI for the main article DOI.
- Crossref registry nuance: after the 34 individuals and group name, its `author` array additionally lists consortium members. Preserve this extra metadata below, but do **not** append it to the BibTeX author list as though all were separate top-level byline authors. The project-provided citation explicitly ends with the group.
- Recommendation: expand `and others` when a complete source bibliography is required. Existing abbreviation is not evidence of a fabricated or reordered author list. Suggested full entry appears below.

Crossref's additional consortium-member sequence, retained as registry metadata (77 names): Aditya Vijaykumar; Alessandro Pietro Bardelli; Alex Rothberg; Andreas Hilboll; Andreas Kloeckner; Anthony Scopatz; Antony Lee; Ariel Rokem; C. Nathan Woods; Chad Fulton; Charles Masson; Christian Häggström; Clark Fitzgerald; David A. Nicholson; David R. Hagen; Dmitrii V. Pasechnik; Emanuele Olivetti; Eric Martin; Eric Wieser; Fabrice Silva; Felix Lenders; Florian Wilhelm; G. Young; Gavin A. Price; Gert-Ludwig Ingold; Gregory E. Allen; Gregory R. Lee; Hervé Audren; Irvin Probst; Jörg P. Dietrich; Jacob Silterra; James T Webber; Janko Slavič; Joel Nothman; Johannes Buchner; Johannes Kulick; Johannes L. Schönberger; José Vinícius de Miranda Cardoso; Joscha Reimer; Joseph Harrington; Juan Luis Cano Rodríguez; Juan Nunez-Iglesias; Justin Kuczynski; Kevin Tritz; Martin Thoma; Matthew Newville; Matthias Kümmerer; Maximilian Bolingbroke; Michael Tartre; Mikhail Pak; Nathaniel J. Smith; Nikolai Nowaczyk; Nikolay Shebanov; Oleksandr Pavlyk; Per A. Brodtkorb; Perry Lee; Robert T. McGibbon; Roman Feldbauer; Sam Lewis; Sam Tygier; Scott Sievert; Sebastiano Vigna; Stefan Peterson; Surhud More; Tadeusz Pudlik; Takuya Oshima; Thomas J. Pingel; Thomas P. Robitaille; Thomas Spura; Thouis R. Jones; Tim Cera; Tim Leslie; Tiziano Zito; Tom Krauss; Utkarsh Upadhyay; Yaroslav O. Halchenko; Yoshiki Vázquez-Baeza.

### 12. `paszke2019pytorch` — Official proceedings verified

- Title: *PyTorch: An Imperative Style, High-Performance Deep Learning Library*.
- Ordered full authors: **Adam Paszke; Sam Gross; Francisco Massa; Adam Lerer; James Bradbury; Gregory Chanan; Trevor Killeen; Zeming Lin; Natalia Gimelshein; Luca Antiga; Alban Desmaison; Andreas Kopf; Edward Yang; Zachary DeVito; Martin Raison; Alykhan Tejani; Sasank Chilamkurthy; Benoit Steiner; Lu Fang; Junjie Bai; Soumith Chintala**.
- Venue/year: *Advances in Neural Information Processing Systems* **32** (NeurIPS 2019), Curran Associates, Inc. Official BibTeX has an empty `pages` field; the current omitted pages should remain omitted unless another official pagination source is acquired.
- DOI status: not supplied in the current bibliography or official proceedings BibTeX; no DOI-specific Crossref request was applicable.
- Sources: live [official NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html); [official BibTeX export](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Bibtex.bib), fetched directly with HTTP 200 after the browser text extractor rejected its BibTeX MIME type.
- Recommendation: retain current entry; all 21 author names and order match. Optional publisher enrichment is supported. Do not add a guessed DOI or guessed page range.

### 13. `pedregosa2011sklearn` — Official journal verified

- Title: *Scikit-learn: Machine Learning in Python*.
- Ordered full authors: **Fabian Pedregosa; Gaël Varoquaux; Alexandre Gramfort; Vincent Michel; Bertrand Thirion; Olivier Grisel; Mathieu Blondel; Peter Prettenhofer; Ron Weiss; Vincent Dubourg; Jake Vanderplas; Alexandre Passos; David Cournapeau; Matthieu Brucher; Matthieu Perrot; Édouard Duchesnay**.
- Venue/year: *Journal of Machine Learning Research* **12**(85), 2825–2830 (2011). Official journal BibTeX uses `number = {85}`; do not reject it merely because JMLR's numbering differs from conventional monthly issues.
- DOI status: not supplied in current bibliography, official journal page, or official BibTeX; no DOI-specific Crossref request was applicable.
- Sources: live [JMLR article](https://www.jmlr.org/papers/v12/pedregosa11a.html); [official BibTeX](https://www.jmlr.org/papers/v12/pedregosa11a.bib), fetched with HTTP 200 after the browser extractor rejected its BibTeX MIME type.
- Recommendation: retain current entry. The spelling `Vanderplas` matches this article's official metadata; the differently capitalized `VanderPlas` on SciPy is not grounds to normalize this author field without article-specific evidence.

## BibTeX suggestions — not applied to `paper/references.bib`

### Complete SciPy citation byline

This follows the official project author sequence and consortium treatment, retaining the already-correct issue number from Crossref and the project citation. It replaces intentional abbreviation with complete metadata rather than correcting a false identity.

```bibtex
@article{virtanen2020scipy,
  author  = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
             Haberland, Matt and Reddy, Tyler and Cournapeau, David and
             Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
             Bright, Jonathan and {van der Walt}, St{\'e}fan J. and
             Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
             Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
             Kern, Robert and Larson, Eric and Carey, C J and
             Polat, {\.I}lhan and Feng, Yu and Moore, Eric W. and
             {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
             Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
             Harris, Charles R. and Archibald, Anne M. and
             Ribeiro, Ant{\^o}nio H. and Pedregosa, Fabian and
             {van Mulbregt}, Paul and {SciPy 1.0 Contributors}},
  title   = {{SciPy} 1.0: Fundamental Algorithms for Scientific Computing in Python},
  journal = {Nature Methods},
  year    = {2020},
  volume  = {17},
  number  = {3},
  pages   = {261--272},
  doi     = {10.1038/s41592-019-0686-2}
}
```

### Optional exact conference naming

```bibtex
@inproceedings{beeson2015tracik,
  author    = {Beeson, Patrick and Ames, Barrett},
  title     = {{TRAC-IK}: An Open-Source Library for Improved Solving of Generic Inverse Kinematics},
  booktitle = {2015 IEEE-RAS 15th International Conference on Humanoid Robots (Humanoids)},
  year      = {2015},
  pages     = {928--935},
  doi       = {10.1109/HUMANOIDS.2015.7363472}
}
```

### Optional PyTorch publisher enrichment

The official export supplies `publisher = {Curran Associates, Inc.}`. It also supplies editors as initials (`H. Wallach; H. Larochelle; A. Beygelzimer; F. d'Alché-Buc; E. Fox; R. Garnett`) and an empty page range. Adding the publisher is supported; no editor-name expansion or pagination suggestion is made without further source work.

## Remaining verification limits

1. Confirm Wampler's exact original byline, especially `W.` and `II`, from an accessible publisher/author first page. Current suffix syntax is structurally correct BibTeX and should not be replaced with incomplete registry metadata.
2. Close Chiaverini et al.'s expanded-name checks using a live publisher/author source. Core identity, ordered initials, year, venue, volume, issue, and pages are already verified by Crossref.
3. Inspect Rice's official volume title page for the complete editor list and original imprint. Core chapter fields are DOI-verified and corroborated by indexed publisher metadata.
4. Publisher access restrictions prevent a claim of live primary metadata verification for Wampler, Chiaverini, ASME Nakamura, and the two Elsevier chapters/articles. Source-specific successful corroboration and indexed-only evidence are explicitly separated above. No entry is declared fabricated because a publisher page is inaccessible.
5. This audit verifies bibliographic identity, not paper relevance, mathematical claims, baseline implementation fidelity, metric comparability, or experimental results. No manuscript acceptance claim follows from it.
