from pathlib import Path

from rag_dataprep_agent.models import FileRecord, ParsedDocument
from rag_dataprep_agent.tools.document_type_detection import detect_document_type


def _parsed(filename: str, text: str) -> ParsedDocument:
    return ParsedDocument(
        file=FileRecord(Path(filename), "pdf", 100, "2026-01-01T00:00:00+00:00"),
        text=text,
        page_count=1,
    )


def test_detects_arxiv_from_content_without_folder_name() -> None:
    parsed = _parsed("mixed-input/document.pdf", "arXiv:2605.10164 Abstract We propose a method. References")

    result = detect_document_type(parsed)

    assert result.document_type == "arxiv_paper"


def test_detects_manual_from_instructional_language() -> None:
    parsed = _parsed("mixed-input/document.pdf", "User Guide Table of Contents Click the File tab and select Options.")

    result = detect_document_type(parsed)

    assert result.document_type == "manual"


def test_detects_eu_document_from_content_without_folder_name() -> None:
    parsed = _parsed("mixed-input/document.pdf", "Council of the European Union Brussels Council conclusions")

    result = detect_document_type(parsed)

    assert result.document_type == "eu_document"
