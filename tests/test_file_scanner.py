from pathlib import Path

from rag_dataprep_agent.inventory.file_scanner import scan_files


def test_scan_files_finds_pdfs_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "unsorted"
    nested.mkdir()
    pdf = nested / "doc.pdf"
    txt = nested / "notes.txt"
    pdf.write_bytes(b"%PDF-1.4\n")
    txt.write_text("not supported", encoding="utf-8")

    records = scan_files(tmp_path)

    assert [record.path for record in records] == [pdf]
    assert records[0].file_type == "pdf"
