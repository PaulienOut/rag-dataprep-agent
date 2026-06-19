import json
import sys
from pathlib import Path

import pytest

from rag_dataprep_agent.config import Settings
from rag_dataprep_agent.evaluation.runner import main


def _write_document(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "document_type": {"document_type": "manual"},
        "content_metadata": {
            "title": "User Guide",
            "subject": None,
            "summary": "A summary.",
            "keywords": ["tutorial"],
            "document_metadata": {"place": None, "date_of_publication": "2010"},
            "layout_metadata": {"header": None, "footer": None},
        },
    }
    (directory / name).write_text(json.dumps(document), encoding="utf-8")


def test_runner_writes_json_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    ground_truth_dir = tmp_path / "ground_truth"
    manifests_dir = tmp_path / "manifests"
    output_path = tmp_path / "reports" / "baseline.json"
    _write_document(ground_truth_dir, "guide.json")
    _write_document(manifests_dir, "guide.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--ground-truth-dir",
            str(ground_truth_dir),
            "--manifests-dir",
            str(manifests_dir),
            "--output",
            str(output_path),
            "--extraction-mode",
            "local",
        ],
    )

    main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["aggregate"]["manifest_coverage"] == 1.0
    assert report["aggregate"]["document_type_accuracy"] == 1.0
    assert report["configuration"]["extraction_mode"] == "local"
    assert report["configuration"]["ground_truth_dir"] == str(ground_truth_dir)
    assert report["configuration"]["manifests_dir"] == str(manifests_dir)
    assert "Wrote evaluation report" in capsys.readouterr().out


def test_runner_requires_api_key_for_llm_judge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ground_truth_dir = tmp_path / "ground_truth"
    manifests_dir = tmp_path / "manifests"
    _write_document(ground_truth_dir, "guide.json")
    _write_document(manifests_dir, "guide.json")
    monkeypatch.setattr(
        "rag_dataprep_agent.evaluation.runner.load_settings",
        lambda: Settings(openai_api_key=None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--ground-truth-dir",
            str(ground_truth_dir),
            "--manifests-dir",
            str(manifests_dir),
            "--llm-judge",
        ],
    )

    with pytest.raises(SystemExit, match="requires OPENAI_API_KEY"):
        main()
