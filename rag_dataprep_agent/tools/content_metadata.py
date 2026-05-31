from __future__ import annotations

import re

from rag_dataprep_agent.models import ContentMetadata, ParsedDocument


def extract_content_metadata(parsed: ParsedDocument) -> ContentMetadata:
    text = parsed.text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = _metadata_title(parsed) or _first_title_like_line(lines)
    summary = _first_sentences(text, limit=2)
    subject = _metadata_subject(parsed)
    keywords = _keywords(text)
    return ContentMetadata(title=title, summary=summary, subject=subject, keywords=keywords)


def _metadata_title(parsed: ParsedDocument) -> str | None:
    title = parsed.pdf_metadata.get("Title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _metadata_subject(parsed: ParsedDocument) -> str | None:
    subject = parsed.pdf_metadata.get("Subject")
    if isinstance(subject, str) and subject.strip():
        return subject.strip()
    return None


def _first_title_like_line(lines: list[str]) -> str | None:
    for line in lines[:20]:
        clean = line.replace("[Page 1]", "").strip()
        if 8 <= len(clean) <= 160 and not clean.endswith("."):
            return clean
    return None


def _first_sentences(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return "No extractable text was found."
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    return " ".join(sentences[:limit])[:900]


def _keywords(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z-]{4,}", text.lower())
    stopwords = {
        "about",
        "after",
        "before",
        "between",
        "document",
        "documents",
        "their",
        "there",
        "these",
        "which",
        "would",
    }
    counts: dict[str, int] = {}
    for word in words:
        if word not in stopwords:
            counts[word] = counts.get(word, 0) + 1
    return [word for word, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]]
