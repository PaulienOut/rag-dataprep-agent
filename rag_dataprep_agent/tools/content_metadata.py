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
    document_metadata = _extract_document_metadata(text, lines)
    layout_metadata = _extract_layout_metadata(text)
    return ContentMetadata(
        title=title,
        summary=summary,
        subject=subject,
        keywords=keywords,
        document_metadata=document_metadata,
        layout_metadata=layout_metadata,
    )


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


def _extract_document_metadata(text: str, lines: list[str]) -> dict[str, str | None]:
    """Extract publication place and date from document."""
    metadata: dict[str, str | None] = {}
    
    # Get first page content
    first_page_match = re.search(r"\[Page 1\](.*?)(?:\[Page 2\]|$)", text, re.DOTALL)
    first_page = first_page_match.group(1) if first_page_match else ""
    
    # Extract place (e.g., "Brussels" from "Brussels, 20 March 2026")
    place_match = re.search(r"([A-Z][a-z]+),\s+\d+\s+[A-Za-z]+\s+\d{4}", first_page)
    if place_match:
        metadata["place"] = place_match.group(1)
    else:
        metadata["place"] = None
    
    # Extract date - try arxiv format first, then look for standalone date
    arxiv_match = re.search(r"arXiv:[^\s]+\s+\[[\w.]+\]\s+(\d+\s+[A-Za-z]+\s+\d{4})", text)
    if arxiv_match:
        metadata["date_of_publication"] = arxiv_match.group(1)
    else:
        # Try DD Month YYYY format (e.g., "20 March 2026")
        date_match = re.search(r"(\d+\s+[A-Za-z]+\s+\d{4})", first_page)
        if date_match:
            metadata["date_of_publication"] = date_match.group(1)
        else:
            # Try Month DD, YYYY format (e.g., "May 8, 2026")
            date_match = re.search(r"([A-Za-z]+\s+\d+,\s+\d{4})", first_page)
            if date_match:
                metadata["date_of_publication"] = date_match.group(1)
            else:
                metadata["date_of_publication"] = None
    
    return metadata


def _extract_layout_metadata(text: str) -> dict[str, str | None]:
    """Extract repeated headers and footers from pages 2+."""
    metadata: dict[str, str | None] = {"header": None, "footer": None}
    
    # Split by page markers
    pages = re.split(r"\[Page \d+\]", text)
    if len(pages) < 4:  # Need at least pages 2, 3, and 4 (indices 2, 3, 4)
        return metadata
    
    # Get lines from pages 2, 3, and 4
    page2_lines = [l.strip() for l in pages[2].split("\n") if l.strip()]
    page3_lines = [l.strip() for l in pages[3].split("\n") if l.strip()]
    page4_lines = [l.strip() for l in pages[4].split("\n") if l.strip()] if len(pages) > 4 else []
    
    if not page2_lines or not page3_lines:
        return metadata
    
    # Extract header (first line that's identical across multiple pages)
    if page2_lines[0] == page3_lines[0] and (not page4_lines or page4_lines[0] == page2_lines[0]):
        metadata["header"] = page2_lines[0]
    
    # Extract footer (look from the end backwards for identical or partially matching lines)
    # First check if last lines are identical
    if page2_lines[-1] == page3_lines[-1] and (not page4_lines or page4_lines[-1] == page2_lines[-1]):
        metadata["footer"] = page2_lines[-1]
    else:
        # Look for lines with common prefix (starting from the end, but stop before header)
        max_idx = len(page2_lines) - 1
        for line_idx in range(max_idx, -1, -1):
            # Skip the header line (usually first line)
            if metadata["header"] and line_idx == 0:
                continue
            
            line2 = page2_lines[line_idx]
            line3 = page3_lines[line_idx] if line_idx < len(page3_lines) else None
            line4 = page4_lines[line_idx] if line_idx < len(page4_lines) else None
            
            if not line3:
                continue
            
            # Skip if lines are identical
            if line2 == line3:
                continue
            
            # Find common prefix
            common = ""
            for c1, c2 in zip(line2, line3):
                if c1 == c2:
                    common += c1
                else:
                    break
            
            # Check if this is a meaningful common prefix (at least 3+ chars and meaningful)
            stripped = common.strip()
            if len(stripped) >= 3:
                # Check if line4 also starts with this prefix if it exists
                if not line4 or line4.startswith(common):
                    metadata["footer"] = stripped
                    break
    
    return metadata


