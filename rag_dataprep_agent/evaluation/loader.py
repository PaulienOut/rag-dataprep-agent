from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationPair:
    document_id: str
    ground_truth_path: Path
    manifest_path: Path
    ground_truth: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class EvaluationDataset:
    pairs: list[EvaluationPair]
    missing_manifests: list[str]
    unexpected_manifests: list[str]


def load_evaluation_dataset(
    ground_truth_dir: str | Path,
    manifests_dir: str | Path,
) -> EvaluationDataset:
    ground_truth_path = _require_directory(ground_truth_dir, "ground-truth")
    manifests_path = _require_directory(manifests_dir, "manifest")

    ground_truth_files = {
        path.stem: path
        for path in sorted(ground_truth_path.glob("*.json"))
        if path.name != "combined.json"
    }
    manifest_files = {path.stem: path for path in sorted(manifests_path.glob("*.json"))}

    matched_ids = sorted(ground_truth_files.keys() & manifest_files.keys())
    pairs = [
        EvaluationPair(
            document_id=document_id,
            ground_truth_path=ground_truth_files[document_id],
            manifest_path=manifest_files[document_id],
            ground_truth=_load_json_object(ground_truth_files[document_id]),
            manifest=_load_json_object(manifest_files[document_id]),
        )
        for document_id in matched_ids
    ]

    return EvaluationDataset(
        pairs=pairs,
        missing_manifests=sorted(ground_truth_files.keys() - manifest_files.keys()),
        unexpected_manifests=sorted(manifest_files.keys() - ground_truth_files.keys()),
    )


def _require_directory(path: str | Path, label: str) -> Path:
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {directory}")
    return directory


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error

    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value
