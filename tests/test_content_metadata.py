from pathlib import Path

from rag_dataprep_agent.models import FileRecord, ParsedDocument
from rag_dataprep_agent.tools.content_metadata import extract_content_metadata


def _parsed(text: str) -> ParsedDocument:
    return ParsedDocument(
        file=FileRecord(Path("document.pdf"), "pdf", 100, "2026-01-01T00:00:00+00:00"),
        text=text,
        page_count=3,
    )


def test_extracts_publication_place_and_date_from_first_page() -> None:
    parsed = _parsed(
        "\n\n".join(
            [
                "[Page 1]\nCouncil of the European Union\nBrussels, 20 March 2026 (OR. en)\nDRAFT MINUTES",
                "[Page 2]\nAgenda item one",
            ]
        )
    )

    metadata = extract_content_metadata(parsed)

    assert metadata.document_metadata == {
        "place": "Brussels",
        "date_of_publication": "20 March 2026",
    }


def test_extracts_repeated_header_and_footer_after_first_page() -> None:
    parsed = _parsed(
        "\n\n".join(
            [
                "[Page 1]\nTitle\nOpening text",
                "[Page 2]\nCouncil of the European Union\nChapter one\nDocument footer",
                "[Page 3]\nCouncil of the European Union\nChapter two\nDocument footer",
                "[Page 4]\nCouncil of the European Union\nChapter three\nDocument footer",
            ]
        )
    )

    metadata = extract_content_metadata(parsed)

    assert metadata.layout_metadata == {
        "header": "Council of the European Union",
        "footer": "Document footer",
    }


def test_classifies_repeated_eu_document_code_as_footer() -> None:
    parsed = _parsed(
        "\n\n".join(
            [
                "[Page 1]\nEUCO 24/25\nEuropean Council\nBrussels, 18 December 2025",
                "[Page 2]\nConclusions - 18 December 2025\nEUCO 24/25    1\nEN\nI. UKRAINE",
                "[Page 3]\nConclusions - 18 December 2025\nEUCO 24/25    2\nEN\nII. MIDDLE EAST",
                "[Page 4]\nConclusions - 18 December 2025\nEUCO 24/25    3\nEN\nIII. SECURITY",
            ]
        )
    )

    metadata = extract_content_metadata(parsed)

    assert metadata.layout_metadata == {
        "header": "Conclusions - 18 December 2025",
        "footer": "EUCO 24/25",
    }


def test_extracts_arxiv_publication_date_from_paper_stamp() -> None:
    parsed = _parsed(
        "\n".join(
            [
                "[Page 1]",
                "The Rise of Negative Earnings and Demand",
                "Shifting Investment",
                "Jacob Toner Gosselin*",
                "Northwestern University",
                "May 8, 2026",
                "Abstract",
                "We document the rise of negative earnings.",
                "1",
                "arXiv:2605.02680v2  [econ.GN]  7 May 2026",
            ]
        )
    )

    metadata = extract_content_metadata(parsed)

    assert metadata.document_metadata == {
        "place": None,
        "date_of_publication": "7 May 2026",
    }


def test_extracts_standalone_paper_date_when_no_arxiv_stamp_exists() -> None:
    parsed = _parsed(
        "\n".join(
            [
                "[Page 1]",
                "A Working Paper",
                "Researcher Name",
                "May 8, 2026",
                "Abstract",
                "This paper studies a question.",
            ]
        )
    )

    metadata = extract_content_metadata(parsed)

    assert metadata.document_metadata == {
        "place": None,
        "date_of_publication": "May 8, 2026",
    }


def test_metadata_shapes_are_present_when_values_are_unknown() -> None:
    parsed = _parsed("[Page 1]\nA plain document with no obvious metadata.")

    metadata = extract_content_metadata(parsed)

    assert metadata.document_metadata == {
        "place": None,
        "date_of_publication": None,
    }
    assert metadata.layout_metadata == {
        "header": None,
        "footer": None,
    }


def test_keyword_extraction_prefers_topic_phrases_over_isolated_words() -> None:
    parsed = ParsedDocument(
        file=FileRecord(Path("document.pdf"), "pdf", 100, "2026-01-01T00:00:00+00:00"),
        text=(
            "[Page 1]\n"
            "Disposition Effect and Systematic Risk Exposure\n"
            "This study examines the disposition effect in short exposure positions. "
            "Systematic risk exposure shapes the disposition effect.\n"
            "[Page 2]\n"
            "The disposition effect remains important under integrated framing."
        ),
        page_count=2,
        pdf_metadata={"Title": "Disposition Effect and Systematic Risk Exposure"},
    )

    metadata = extract_content_metadata(parsed)

    assert "disposition effect" in metadata.keywords
    assert "systematic risk exposure" in metadata.keywords
    assert "disposition" not in metadata.keywords
    assert "effect" not in metadata.keywords


def test_keyword_extraction_uses_single_words_only_as_a_fallback() -> None:
    metadata = extract_content_metadata(_parsed("[Page 1]\nExcel"))

    assert metadata.keywords == ["excel"]
