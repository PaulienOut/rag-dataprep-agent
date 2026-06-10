from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from rag_dataprep_agent.models import PreparedDocument


def write_manifest(prepared: PreparedDocument, output_dir: str | Path) -> Path:
    manifest_dir = Path(output_dir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(prepared.source_path).stem
    target = manifest_dir / f"{source_name}.json"
    target.write_text(json.dumps(_to_jsonable(prepared), indent=2), encoding="utf-8")
    return target


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
