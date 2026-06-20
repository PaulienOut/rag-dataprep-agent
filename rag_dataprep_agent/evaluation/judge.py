from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI


@dataclass(frozen=True)
class SummaryJudgeResult:
    factual_consistency: int
    coverage: int
    relevance: int
    conciseness: int
    overall_score: float
    reason: str
    model: str


class SummaryJudge(Protocol):
    model: str

    def judge(self, document_id: str, reference_summary: str, generated_summary: str) -> SummaryJudgeResult: ...


class OpenAISummaryJudge:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model

    def judge(
        self,
        document_id: str,
        reference_summary: str,
        generated_summary: str,
    ) -> SummaryJudgeResult:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate a generated document summary against a manually reviewed reference summary. "
                        "Score each criterion from 1 (poor) to 5 (excellent). "
                        "Factual consistency means consistency with the reference summary; you are not independently "
                        "checking the full source document. Coverage measures whether the generated summary includes "
                        "the important information in the reference. Relevance measures focus on the document topic. "
                        "Conciseness measures whether the summary is direct and avoids unnecessary detail. "
                        "Do not penalize harmless wording differences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Document: {document_id}\n\n"
                        f"Reference summary:\n{reference_summary}\n\n"
                        f"Generated summary:\n{generated_summary}"
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "summary_evaluation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "factual_consistency": {"type": "integer", "minimum": 1, "maximum": 5},
                            "coverage": {"type": "integer", "minimum": 1, "maximum": 5},
                            "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
                            "conciseness": {"type": "integer", "minimum": 1, "maximum": 5},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "factual_consistency",
                            "coverage",
                            "relevance",
                            "conciseness",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )
        try:
            data = json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise ValueError(f"summary judge returned invalid JSON for {document_id}") from error

        scores = {
            name: _score(data, name, document_id)
            for name in ("factual_consistency", "coverage", "relevance", "conciseness")
        }
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"summary judge returned no reason for {document_id}")

        return SummaryJudgeResult(
            **scores,
            overall_score=sum(scores.values()) / len(scores),
            reason=reason.strip(),
            model=self.model,
        )


def _score(data: dict[str, Any], name: str, document_id: str) -> int:
    value = data.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
        raise ValueError(f"summary judge returned invalid {name} score for {document_id}")
    return value
