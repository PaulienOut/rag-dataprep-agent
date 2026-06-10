from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DocumentType = Literal["arxiv_paper", "manual", "eu_document", "unknown"]


@dataclass(frozen=True)
class FileRecord:
    path: Path
    file_type: str
    size_bytes: int
    modified_at: str


@dataclass(frozen=True)
class ParsedDocument:
    file: FileRecord
    text: str
    page_count: int
    pdf_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentTypeResult:
    document_type: DocumentType
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContentMetadata:
    title: str | None
    summary: str
    subject: str | None
    keywords: list[str] = field(default_factory=list)
    document_metadata: dict[str, str | None] = field(default_factory=dict)
    layout_metadata: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    embedding: list[float] | None = None


@dataclass(frozen=True)
class PreparedDocument:
    source_path: str
    file_metadata: dict[str, Any]
    pdf_metadata: dict[str, Any]
    document_type: DocumentTypeResult
    content_metadata: ContentMetadata
    chunks: list[DocumentChunk]
