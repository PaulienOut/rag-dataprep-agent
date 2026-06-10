from __future__ import annotations

from pypdf import PdfReader

from rag_dataprep_agent.models import FileRecord, ParsedDocument


def parse_pdf(file_record: FileRecord, max_pages: int | None = None) -> ParsedDocument:
    reader = PdfReader(str(file_record.path))
    pages = reader.pages[:max_pages] if max_pages is not None else reader.pages
    page_texts = []

    for index, page in enumerate(pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append(f"[Page {index}]\n{text.strip()}")

    metadata = {
        key.lstrip("/"): value
        for key, value in dict(reader.metadata or {}).items()
        if value is not None
    }

    return ParsedDocument(
        file=file_record,
        text="\n\n".join(page_texts).strip(),
        page_count=len(reader.pages),
        pdf_metadata=metadata,
    )
