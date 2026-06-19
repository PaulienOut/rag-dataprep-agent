import json
from pathlib import Path

import pytest

from rag_dataprep_agent.evaluation.loader import load_evaluation_dataset


def _write_json(directory: Path, name: str, value: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_loads_pairs_by_filename_and_skips_combined_overview(tmp_path: Path) -> None:
    ground_truth_dir = tmp_path / "ground_truth"
    manifests_dir = tmp_path / "manifests"
    _write_json(ground_truth_dir, "document-b.json", {"expected": "b"})
    _write_json(ground_truth_dir, "document-a.json", {"expected": "a"})
    _write_json(ground_truth_dir, "combined.json", [{"filename": "document-a.pdf"}])
    _write_json(manifests_dir, "document-a.json", {"generated": "a"})
    _write_json(manifests_dir, "document-b.json", {"generated": "b"})

    dataset = load_evaluation_dataset(ground_truth_dir, manifests_dir)

    assert [pair.document_id for pair in dataset.pairs] == ["document-a", "document-b"]
    assert dataset.pairs[0].ground_truth == {"expected": "a"}
    assert dataset.pairs[0].manifest == {"generated": "a"}
    assert dataset.missing_manifests == []
    assert dataset.unexpected_manifests == []


def test_reports_missing_and_unexpected_manifests(tmp_path: Path) -> None:
    ground_truth_dir = tmp_path / "ground_truth"
    manifests_dir = tmp_path / "manifests"
    _write_json(ground_truth_dir, "matched.json", {})
    _write_json(ground_truth_dir, "missing.json", {})
    _write_json(manifests_dir, "matched.json", {})
    _write_json(manifests_dir, "unexpected.json", {})

    dataset = load_evaluation_dataset(ground_truth_dir, manifests_dir)

    assert [pair.document_id for pair in dataset.pairs] == ["matched"]
    assert dataset.missing_manifests == ["missing"]
    assert dataset.unexpected_manifests == ["unexpected"]


def test_rejects_invalid_json_with_the_file_path(tmp_path: Path) -> None:
    ground_truth_dir = tmp_path / "ground_truth"
    manifests_dir = tmp_path / "manifests"
    ground_truth_dir.mkdir()
    invalid_path = ground_truth_dir / "document.json"
    invalid_path.write_text("{not valid JSON", encoding="utf-8")
    _write_json(manifests_dir, "document.json", {})

    with pytest.raises(ValueError, match=str(invalid_path)):
        load_evaluation_dataset(ground_truth_dir, manifests_dir)


def test_rejects_non_object_document_json(tmp_path: Path) -> None:
    ground_truth_dir = tmp_path / "ground_truth"
    manifests_dir = tmp_path / "manifests"
    _write_json(ground_truth_dir, "document.json", ["not", "an", "object"])
    _write_json(manifests_dir, "document.json", {})

    with pytest.raises(ValueError, match="expected a JSON object"):
        load_evaluation_dataset(ground_truth_dir, manifests_dir)
