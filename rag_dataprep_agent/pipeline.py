from __future__ import annotations

from pathlib import Path

from rag_dataprep_agent.config import Settings
from rag_dataprep_agent.inventory.file_scanner import scan_files
from rag_dataprep_agent.llm.agent import DocumentPrepAgent
from rag_dataprep_agent.llm.client import build_openai_client
from rag_dataprep_agent.models import PreparedDocument
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
    client = build_openai_client(settings.openai_api_key) if use_llm else None
    agent = DocumentPrepAgent(client=client, metadata_model=settings.metadata_model)
    records = scan_files(input_path)
    if max_files is not None:
        records = records[:max_files]

    written: list[Path] = []
    for record in records:
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
        written.append(write_manifest(prepared, output_dir))
    return written
