from pathlib import Path

from pypdf import PdfWriter

from rag_dataprep_agent.tools.file_metadata import build_file_record
from rag_dataprep_agent.parsers.pdf_parser import parse_pdf


def test_parse_pdf_reads_basic_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Title": "Sample PDF"})
    with path.open("wb") as file:
        writer.write(file)

    parsed = parse_pdf(build_file_record(path))

    assert parsed.page_count == 1
    assert parsed.pdf_metadata["Title"] == "Sample PDF"
