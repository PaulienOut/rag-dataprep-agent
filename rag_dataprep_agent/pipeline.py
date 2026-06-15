from __future__ import annotations

from pathlib import Path

from rag_dataprep_agent.config import Settings
from rag_dataprep_agent.inventory.file_scanner import scan_files
from rag_dataprep_agent.llm.agent import DocumentPrepAgent
from rag_dataprep_agent.llm.client import build_openai_client
from rag_dataprep_agent.models import PreparedDocument
from rag_dataprep_agent.observability import configure_logfire, info, instrument_openai_client, span
from rag_dataprep_agent.parsers.pdf_parser import parse_pdf
from rag_dataprep_agent.storage.manifests import write_manifest
from rag_dataprep_agent.tools.chunking import chunk_text
from rag_dataprep_agent.tools.document_type_detection import detect_document_type
from rag_dataprep_agent.tools.embeddings import deterministic_embedding
from rag_dataprep_agent.tools.file_metadata import file_record_to_metadata


def prepare_documents(
    input_path: str | Path,
    output_dir: str | Path,
    settings: Settings,
    use_llm: bool = False,
    embed: bool = False,
    max_files: int | None = None,
) -> list[Path]:
    configure_logfire(settings)
    client = build_openai_client(settings.openai_api_key) if use_llm else None
    instrument_openai_client(client, settings)
    agent = DocumentPrepAgent(client=client, metadata_model=settings.metadata_model)
    records = scan_files(input_path)
    if max_files is not None:
        records = records[:max_files]

    written: list[Path] = []
    with span(
        settings,
        "Prepare documents",
        input_path=str(input_path),
        output_dir=str(output_dir),
        file_count=len(records),
        use_llm=use_llm,
        embed=embed,
    ):
        for record in records:
            with span(settings, "Prepare document", source_path=str(record.path), file_size=record.size_bytes):
                parsed = parse_pdf(record)
                document_type = detect_document_type(parsed)
                content_metadata = agent.extract_content_metadata(parsed)
                chunks = chunk_text(parsed.text, settings.chunk_size, settings.chunk_overlap)

                if embed and chunks:
                    remote_embeddings = agent.embed_texts([chunk.text for chunk in chunks], settings.embedding_model)
                    embeddings = remote_embeddings or [deterministic_embedding(chunk.text) for chunk in chunks]
                    chunks = [
                        type(chunk)(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            start_char=chunk.start_char,
                            end_char=chunk.end_char,
                            embedding=embeddings[index],
                        )
                        for index, chunk in enumerate(chunks)
                    ]

                prepared = PreparedDocument(
                    source_path=str(record.path),
                    file_metadata=file_record_to_metadata(record),
                    pdf_metadata=parsed.pdf_metadata | {"page_count": parsed.page_count},
                    document_type=document_type,
                    content_metadata=content_metadata,
                    chunks=chunks,
                )
                written_path = write_manifest(prepared, output_dir)
                written.append(written_path)
                info(
                    settings,
                    "Prepared document",
                    source_path=str(record.path),
                    manifest_path=str(written_path),
                    document_type=document_type.document_type,
                    page_count=parsed.page_count,
                    chunk_count=len(chunks),
                    used_llm=client is not None,
                    embedded=embed,
                )
    return written
