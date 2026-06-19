from pathlib import Path

import pytest

from rag_dataprep_agent.evaluation.evaluator import evaluate_dataset
from rag_dataprep_agent.evaluation.judge import SummaryJudgeResult
from rag_dataprep_agent.evaluation.loader import EvaluationDataset, EvaluationPair


def _pair(document_id: str, expected: dict, generated: dict) -> EvaluationPair:
    return EvaluationPair(
        document_id=document_id,
        ground_truth_path=Path(f"{document_id}-expected.json"),
        manifest_path=Path(f"{document_id}-generated.json"),
        ground_truth=expected,
        manifest=generated,
    )


def _document(document_type: str, title: str, keywords: list[str]) -> dict:
    return {
        "document_type": {"document_type": document_type},
        "content_metadata": {
            "title": title,
            "subject": None,
            "summary": "A summary.",
            "keywords": keywords,
            "document_metadata": {
                "place": "Brussels",
                "date_of_publication": "18 April 2026",
            },
            "layout_metadata": {
                "header": None,
                "footer": "Document footer",
            },
        },
    }


class FakeSummaryJudge:
    model = "fake-judge"

    def judge(self, document_id: str, reference_summary: str, generated_summary: str) -> SummaryJudgeResult:
        return SummaryJudgeResult(
            factual_consistency=5,
            coverage=4,
            relevance=5,
            conciseness=4,
            overall_score=4.5,
            reason=f"Reviewed {document_id}",
            model=self.model,
        )


def test_evaluate_dataset_contains_inspectable_field_results() -> None:
    dataset = EvaluationDataset(
        pairs=[
            _pair(
                "document",
                _document("eu_document", "Organic Farming Rules", ["labeling"]),
                _document("eu_document", "Organic farming rules", ["labelling"]),
            )
        ],
        missing_manifests=[],
        unexpected_manifests=[],
    )

    report = evaluate_dataset(dataset)
    fields = report["documents"][0]["fields"]

    assert fields["document_type"]["exact_match"] is True
    assert fields["title"]["exact_match"] is True
    assert fields["keywords"]["f1"] == 1.0
    assert fields["date_of_publication"]["expected"] == "18 April 2026"
    assert fields["date_of_publication"]["normalized_expected"] == "2026-04-18"
    assert fields["summary"]["judge"] is None


def test_missing_manifest_counts_as_zero_in_aggregate_scores() -> None:
    document = _document("manual", "User Guide", ["tutorial"])
    dataset = EvaluationDataset(
        pairs=[_pair("matched", document, document)],
        missing_manifests=["missing"],
        unexpected_manifests=["extra"],
    )

    report = evaluate_dataset(dataset)
    aggregate = report["aggregate"]

    assert aggregate["expected_document_count"] == 2
    assert aggregate["matched_document_count"] == 1
    assert aggregate["manifest_coverage"] == 0.5
    assert aggregate["document_type_accuracy"] == 0.5
    assert aggregate["title_similarity"] == 0.5
    assert aggregate["keyword_f1"] == 0.5
    assert report["missing_manifests"] == ["missing"]
    assert report["unexpected_manifests"] == ["extra"]


def test_keyword_threshold_is_recorded_and_applied() -> None:
    expected = _document("manual", "Guide", ["labeling"])
    generated = _document("manual", "Guide", ["labelling"])
    dataset = EvaluationDataset(
        pairs=[_pair("document", expected, generated)],
        missing_manifests=[],
        unexpected_manifests=[],
    )

    report = evaluate_dataset(dataset, keyword_similarity_threshold=1.0)

    assert report["configuration"]["keyword_similarity_threshold"] == 1.0
    assert report["aggregate"]["keyword_f1"] == 0.0


def test_summary_judge_results_are_recorded_and_aggregated() -> None:
    document = _document("manual", "Guide", ["tutorial"])
    dataset = EvaluationDataset(
        pairs=[_pair("document", document, document)],
        missing_manifests=[],
        unexpected_manifests=[],
    )

    report = evaluate_dataset(dataset, summary_judge=FakeSummaryJudge())
    judge = report["documents"][0]["fields"]["summary"]["judge"]

    assert report["configuration"]["summary_judge_enabled"] is True
    assert report["configuration"]["summary_judge_model"] == "fake-judge"
    assert judge["overall_score"] == 4.5
    assert judge["reason"] == "Reviewed document"
    assert report["aggregate"]["summary_judged_document_count"] == 1
    assert report["aggregate"]["summary_judge_score"] == 4.5


def test_empty_dataset_has_zero_coverage_and_scores() -> None:
    report = evaluate_dataset(EvaluationDataset([], [], []))

    assert report["aggregate"]["manifest_coverage"] == 0.0
    assert report["aggregate"]["document_type_accuracy"] == 0.0
    assert report["aggregate"]["keyword_f1"] == 0.0
