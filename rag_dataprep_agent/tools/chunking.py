from __future__ import annotations

from rag_dataprep_agent.models import DocumentChunk


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or greater and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk-{len(chunks) + 1:04d}",
                    text=chunk,
                    start_char=start,
                    end_char=end,
                )
            )
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks
