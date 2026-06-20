from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from rag_dataprep_agent.evaluation.judge import SummaryJudge
from rag_dataprep_agent.evaluation.loader import EvaluationDataset, EvaluationPair
from rag_dataprep_agent.evaluation.metrics import compare_date, compare_keywords, compare_text

TEXT_FIELDS = {
    "title": ("content_metadata", "title"),
    "subject": ("content_metadata", "subject"),
    "place": ("content_metadata", "document_metadata", "place"),
    "header": ("content_metadata", "layout_metadata", "header"),
    "footer": ("content_metadata", "layout_metadata", "footer"),
}


def evaluate_dataset(
    dataset: EvaluationDataset,
    keyword_similarity_threshold: float = 0.85,
    summary_judge: SummaryJudge | None = None,
) -> dict[str, Any]:
    expected_document_count = len(dataset.pairs) + len(dataset.missing_manifests)
    document_results = [
        _evaluate_pair(pair, keyword_similarity_threshold, summary_judge)
        for pair in dataset.pairs
    ]
    document_results.extend(
        {
            "document_id": document_id,
            "status": "missing_manifest",
            "fields": None,
        }
        for document_id in dataset.missing_manifests
    )
    document_results.sort(key=lambda result: result["document_id"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "keyword_similarity_threshold": keyword_similarity_threshold,
            "summary_judge_enabled": summary_judge is not None,
            "summary_judge_model": summary_judge.model if summary_judge is not None else None,
        },
        "aggregate": _aggregate_results(document_results, expected_document_count, len(dataset.pairs)),
        "missing_manifests": dataset.missing_manifests,
        "unexpected_manifests": dataset.unexpected_manifests,
        "documents": document_results,
    }


def _evaluate_pair(
    pair: EvaluationPair,
    keyword_similarity_threshold: float,
    summary_judge: SummaryJudge | None,
) -> dict[str, Any]:
    expected_type = _optional_text(_nested(pair.ground_truth, "document_type", "document_type"))
    generated_type = _optional_text(_nested(pair.manifest, "document_type", "document_type"))

    fields: dict[str, Any] = {
        "document_type": {
            "expected": expected_type,
            "generated": generated_type,
            "exact_match": expected_type is not None and expected_type == generated_type,
        }
    }
    for field_name, path in TEXT_FIELDS.items():
        fields[field_name] = asdict(
            compare_text(
                _optional_text(_nested(pair.ground_truth, *path)),
                _optional_text(_nested(pair.manifest, *path)),
            )
        )

    expected_date = _optional_text(
        _nested(pair.ground_truth, "content_metadata", "document_metadata", "date_of_publication")
    )
    generated_date = _optional_text(
        _nested(pair.manifest, "content_metadata", "document_metadata", "date_of_publication")
    )
    fields["date_of_publication"] = asdict(compare_date(expected_date, generated_date))

    expected_keywords = _string_list(_nested(pair.ground_truth, "content_metadata", "keywords"))
    generated_keywords = _string_list(_nested(pair.manifest, "content_metadata", "keywords"))
    fields["keywords"] = {
        "expected": expected_keywords,
        "generated": generated_keywords,
        **asdict(
            compare_keywords(
                expected_keywords,
                generated_keywords,
                similarity_threshold=keyword_similarity_threshold,
            )
        ),
    }
    expected_summary = _optional_text(_nested(pair.ground_truth, "content_metadata", "summary"))
    generated_summary = _optional_text(_nested(pair.manifest, "content_metadata", "summary"))
    judge_result = None
    if summary_judge is not None and expected_summary is not None and generated_summary is not None:
        judge_result = asdict(summary_judge.judge(pair.document_id, expected_summary, generated_summary))
    fields["summary"] = {
        "expected": expected_summary,
        "generated": generated_summary,
        "judge": judge_result,
    }

    return {
        "document_id": pair.document_id,
        "status": "evaluated",
        "fields": fields,
    }


def _aggregate_results(
    document_results: list[dict[str, Any]],
    expected_document_count: int,
    matched_document_count: int,
) -> dict[str, Any]:
    evaluated = [result for result in document_results if result["status"] == "evaluated"]
    judged = [
        result["fields"]["summary"]["judge"]
        for result in evaluated
        if result["fields"]["summary"]["judge"] is not None
    ]
    denominator = expected_document_count

    def mean_score(field_name: str, score_name: str) -> float:
        if denominator == 0:
            return 0.0
        return sum(result["fields"][field_name][score_name] for result in evaluated) / denominator

    return {
        "expected_document_count": expected_document_count,
        "matched_document_count": matched_document_count,
        "manifest_coverage": matched_document_count / expected_document_count if expected_document_count else 0.0,
        "document_type_accuracy": mean_score("document_type", "exact_match"),
        "title_exact_match": mean_score("title", "exact_match"),
        "title_similarity": mean_score("title", "similarity"),
        "subject_similarity": mean_score("subject", "similarity"),
        "keyword_precision": mean_score("keywords", "precision"),
        "keyword_recall": mean_score("keywords", "recall"),
        "keyword_f1": mean_score("keywords", "f1"),
        "place_similarity": mean_score("place", "similarity"),
        "date_exact_match": mean_score("date_of_publication", "exact_match"),
        "date_similarity": mean_score("date_of_publication", "similarity"),
        "header_similarity": mean_score("header", "similarity"),
        "footer_similarity": mean_score("footer", "similarity"),
        "summary_judged_document_count": len(judged),
        "summary_judge_score": (
            sum(result["overall_score"] for result in judged) / len(judged)
            if judged
            else None
        ),
    }


def _nested(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
