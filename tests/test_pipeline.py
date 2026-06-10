from pathlib import Path

from pypdf import PdfWriter

from rag_dataprep_agent.config import Settings
from rag_dataprep_agent.pipeline import prepare_documents


def test_prepare_documents_writes_manifest_for_pdf(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Title": "Sample Manual"})
    with path.open("wb") as file:
        writer.write(file)

    written = prepare_documents(
        input_path=tmp_path,
        output_dir=tmp_path / "prepared",
        settings=Settings(openai_api_key=None),
        use_llm=False,
        embed=True,
    )

    assert len(written) == 1
    assert written[0].exists()
