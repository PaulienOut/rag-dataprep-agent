"""Ground-truth evaluation helpers."""

from rag_dataprep_agent.evaluation.evaluator import evaluate_dataset
from rag_dataprep_agent.evaluation.judge import OpenAISummaryJudge, SummaryJudge, SummaryJudgeResult
from rag_dataprep_agent.evaluation.loader import EvaluationDataset, EvaluationPair, load_evaluation_dataset
from rag_dataprep_agent.evaluation.metrics import (
    KeywordMatch,
    KeywordMetrics,
    TextComparison,
    compare_date,
    compare_keywords,
    compare_text,
    normalize_date,
    normalize_text,
    text_similarity,
)

__all__ = [
    "EvaluationDataset",
    "EvaluationPair",
    "KeywordMatch",
    "KeywordMetrics",
    "OpenAISummaryJudge",
    "SummaryJudge",
    "SummaryJudgeResult",
    "TextComparison",
    "compare_date",
    "compare_keywords",
    "compare_text",
    "evaluate_dataset",
    "load_evaluation_dataset",
    "normalize_date",
    "normalize_text",
    "text_similarity",
]
