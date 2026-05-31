from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rag_dataprep_agent.models import FileRecord


def build_file_record(path: str | Path) -> FileRecord:
    file_path = Path(path)
    stat = file_path.stat()
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return FileRecord(
        path=file_path,
        file_type=file_path.suffix.lower().lstrip(".") or "unknown",
        size_bytes=stat.st_size,
        modified_at=modified_at,
    )


def file_record_to_metadata(record: FileRecord) -> dict[str, object]:
    return {
        "filename": record.path.name,
        "path": str(record.path),
        "file_type": record.file_type,
        "size_bytes": record.size_bytes,
        "modified_at": record.modified_at,
    }
