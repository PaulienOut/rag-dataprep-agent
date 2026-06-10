from __future__ import annotations

import re

from rag_dataprep_agent.models import DocumentTypeResult, ParsedDocument


def detect_document_type(parsed: ParsedDocument) -> DocumentTypeResult:
    text = parsed.text[:12000].lower()
    filename = parsed.file.path.name.lower()
    evidence: list[str] = []
    scores = {
        "arxiv_paper": 0,
        "manual": 0,
        "eu_document": 0,
    }

    if re.search(r"\b\d{4}\.\d{4,5}v?\d*\b", filename) or "arxiv:" in text:
        scores["arxiv_paper"] += 4
        evidence.append("arXiv identifier detected")
    if "abstract" in text and "references" in text:
        scores["arxiv_paper"] += 2
        evidence.append("academic sections detected")

    manual_terms = ["user guide", "product guide", "tutorial", "quick start", "manual", "step-by-step"]
    manual_hits = [term for term in manual_terms if term in text or term in filename]
    if manual_hits:
        scores["manual"] += 2 + len(manual_hits)
        evidence.append(f"manual language detected: {', '.join(manual_hits[:3])}")
    if "table of contents" in text and ("click" in text or "select" in text):
        scores["manual"] += 2
        evidence.append("instructional table of contents detected")

    eu_terms = ["european council", "council of the european union", "brussels", "council conclusions"]
    eu_hits = [term for term in eu_terms if term in text or term in filename]
    if eu_hits:
        scores["eu_document"] += 2 + len(eu_hits)
        evidence.append(f"EU institutional language detected: {', '.join(eu_hits[:3])}")
    if re.search(r"\bst-\d{1,5}-\d{4}\b|\bcm-\d{1,5}-\d{4}\b", filename):
        scores["eu_document"] += 4
        evidence.append("EU document code detected in filename")

    document_type = max(scores, key=scores.get)
    score = scores[document_type]
    if score == 0:
        return DocumentTypeResult("unknown", 0.0, ["no strong type signal detected"])

    confidence = min(0.95, 0.45 + score * 0.1)
    return DocumentTypeResult(document_type, confidence, evidence)
