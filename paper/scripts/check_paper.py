#!/usr/bin/env python3
"""Read-only manuscript/evidence QA, with derived JSON output. Never runs IK."""
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

from build_evidence import COMMIT, EVIDENCE, PAPER, ROOT, digest, load, write

UPDATED_DOCS = {
    "docs/FINAL_PAPER_CLAIM_MAP.md",
    "docs/FINAL_PAPER_CHANGELOG.md",
    "docs/RESEARCH.md",
}


def bibliography(text):
    """Parse brace-valued BibTeX fields, retaining nested LaTeX author accents."""
    entries = {}
    starts = list(re.finditer(r"(?m)^@(\w+)\{([^,]+),", text))
    for idx, start in enumerate(starts):
        body = text[start.end():starts[idx + 1].start() if idx + 1 < len(starts) else len(text)]
        fields = {}
        pos = 0
        while True:
            m = re.search(r"(\w+)\s*=\s*\{", body[pos:])
            if m is None:
                break
            name = m.group(1).lower()
            begin = pos + m.end()
            end, depth = begin, 1
            while end < len(body) and depth:
                char = body[end]
                escaped = end > 0 and body[end - 1] == "\\"
                if not escaped and char == "{":
                    depth += 1
                elif not escaped and char == "}":
                    depth -= 1
                end += 1
            assert depth == 0, start.group(2)
            fields[name] = " ".join(body[begin:end - 1].split())
            pos = end
        key = start.group(2)
        assert key not in entries, key
        entries[key] = dict(type=start.group(1), **fields)
    return entries


def verify_archive(path, expected):
    with tarfile.open(path) as archive:
        actual = {}
        for member in archive.getmembers():
            if member.isfile():
                actual[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    for name, old_hash in expected.items():
        assert actual[name] == old_hash, (str(path), name)
    return len(expected)


def main():
    delivery = load(EVIDENCE / "delivery_manifest.json")
    for name, expected in delivery["files"].items():
        assert digest(ROOT / name) == expected, name
    protected = load(EVIDENCE / "01_protocol/protected_files.json")
    paper_history = {p: h for p, h in protected.items() if p.startswith("paper/")}
    doc_history = {p: h for p, h in protected.items() if p in UPDATED_DOCS}
    unchanged = {p: h for p, h in protected.items() if p not in paper_history and p not in doc_history}
    for name, expected in unchanged.items():
        assert digest(ROOT / name) == expected, name
    archived_paper = verify_archive(PAPER / "history/cghik_paper_02aa287.tar.gz", paper_history)
    archived_docs = verify_archive(ROOT / "docs/history/task_contract_predecessor_02aa287.tar.gz", doc_history)

    snap = load(PAPER / "generated/evidence_snapshot.json")
    assert snap["evidence_commit"] == COMMIT
    for name, expected in snap["sources"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in snap["data_files"].items():
        assert digest(PAPER / name) == expected, name
    for name, expected in load(PAPER / "generated/figure_manifest.json")["files"].items():
        assert digest(PAPER / name) == expected, name

    manuscript = (PAPER / "main.tex").read_text()
    used = set(re.findall(r"\\ev\{([^}]+)\}", manuscript))
    assert used <= snap["numbers"].keys(), used - snap["numbers"].keys()
    definitions = dict(re.findall(
        r"\\expandafter\\def\\csname ev@([^\\]+)\\endcsname\{([^\n]*)\}",
        (PAPER / "generated/paper_numbers.tex").read_text()))
    assert definitions == snap["numbers"], "Generated number definitions differ"
    for name in re.findall(r"\\input\{([^}]+)\}", manuscript):
        assert (PAPER / (name if name.endswith(".tex") else name + ".tex")).is_file(), name
    expected_sections = [
        "Introduction", "Related Work", "Problem Formulation",
        "Task-Contract-Aware Solver Interface", "Experimental Protocol",
        "Results", "Discussion", "Limitations", "Conclusions",
    ]
    sections = re.findall(r"\\section\{([^}]+)\}", manuscript)
    assert sections[:9] == expected_sections, sections
    assert len(re.findall(r"\\subsection\{RQ[123]:", manuscript)) == 3
    intro = manuscript.split(r"\section{Introduction}", 1)[1].split(r"\section{Related Work}", 1)[0]
    intro = re.sub(r"\\label\{[^}]+\}|\\cite\w*\{[^}]+\}", "", intro)
    paragraphs = [p for p in re.split(r"\n\s*\n", intro) if p.strip()]
    assert len(paragraphs) == 7, len(paragraphs)
    words = re.findall(r"\b[A-Za-z]+(?:['-][A-Za-z]+)*\b", intro)
    assert 1200 <= len(words) <= 1500, len(words)
    narrative = manuscript.split(r"\section*{Data and code availability}")[0]
    assert not re.search(r"\b[Vv][2-9]\b", narrative), "Development version in narrative"

    bib = bibliography((PAPER / "references.bib").read_text())
    cited = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", manuscript):
        cited.update(x.strip() for x in group.split(","))
    assert cited == bib.keys(), (cited - bib.keys(), bib.keys() - cited)
    assert 35 <= len(bib) <= 45
    recent = sum(2024 <= int(e["year"]) <= 2026 for e in bib.values())
    assert 20 <= recent <= 25, recent
    audit = (ROOT / "docs/REFERENCE_VERIFICATION_TASK_CONTRACT.md").read_text()
    for key, entry in bib.items():
        assert key in audit, key
        assert all(entry.get(k) for k in ["author", "title", "year"]), key
        assert entry.get("doi") or entry.get("url"), key
        if entry["type"] == "misc":
            assert "preprint" in entry.get("note", "").lower(), key
        else:
            assert entry.get("journal") or entry.get("booktitle"), key
    write(PAPER / "generated/reference_inventory.json", {
        "bibliography_sha256": digest(PAPER / "references.bib"),
        "audit": "docs/REFERENCE_VERIFICATION_TASK_CONTRACT.md",
        "entries": bib,
        "note": "Index of checked source BibTeX, not independent metadata verification.",
    })

    log = (PAPER / "main.log").read_text()
    bad = re.findall(r"^.*(?:Warning|Overfull|Underfull|undefined|^!).*$", log, re.MULTILINE)
    assert not bad, bad
    info = subprocess.check_output(["pdfinfo", str(PAPER / "main.pdf")], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
    visual = load(PAPER / "generated/visual_review.json")
    assert visual["pdf_sha256"] == digest(PAPER / "main.pdf"), "Re-inspect rebuilt PDF"
    assert visual["reviewed_pages"] == list(range(1, pages + 1)), "Incomplete page inspection"
    assert visual["blocking_layout_issues"] == [], visual

    # Verify paper builds deterministically materialize their frozen numeric sources.
    generated_before = {p: digest(p) for p in (PAPER / "generated").glob("task_*_rows.tex")}
    generated_before[PAPER / "generated/paper_numbers.tex"] = digest(PAPER / "generated/paper_numbers.tex")
    generated_before[PAPER / "generated/evidence_snapshot.json"] = digest(PAPER / "generated/evidence_snapshot.json")
    import build_evidence
    build_evidence.main()
    for path, expected in generated_before.items():
        assert digest(path) == expected, str(path)

    result = dict(
        status="passed", evidence_commit=COMMIT, pdf_sha256=digest(PAPER / "main.pdf"),
        page_count=pages, introduction_paragraphs=len(paragraphs),
        introduction_english_words=len(words), references=len(bib),
        references_2024_2026=recent, preprint_references=sum(e["type"] == "misc" for e in bib.values()),
        used_numeric_keys=len(used), generated_numeric_keys=len(definitions),
        current_evidence_files_unchanged=len(delivery["files"]),
        historical_protected_files_unchanged=len(unchanged),
        historical_paper_files_archived=archived_paper,
        predecessor_docs_archived=archived_docs,
        deterministic_evidence_rebuild=True, visual_review=visual,
        solver_calls_in_paper_build=0,
    )
    write(PAPER / "generated/final_qa.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "visual_review"}, indent=2))


if __name__ == "__main__":
    main()
