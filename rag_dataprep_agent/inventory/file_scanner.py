from __future__ import annotations

from pathlib import Path

from rag_dataprep_agent.models import FileRecord
from rag_dataprep_agent.tools.file_metadata import build_file_record


SUPPORTED_SUFFIXES = {".pdf"}


def scan_files(input_path: str | Path) -> list[FileRecord]:
    path = Path(input_path)
    if path.is_file():
        candidates = [path]
    else:
        candidates = [item for item in path.rglob("*") if item.is_file()]

    return [
        build_file_record(candidate)
        for candidate in sorted(candidates)
        if candidate.suffix.lower() in SUPPORTED_SUFFIXES
    ]
