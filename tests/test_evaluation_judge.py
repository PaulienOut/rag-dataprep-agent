import json
from types import SimpleNamespace

import pytest

from rag_dataprep_agent.evaluation.judge import OpenAISummaryJudge


class FakeResponses:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output_text = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return SimpleNamespace(output_text=output_text)


def test_openai_summary_judge_returns_structured_scores() -> None:
    responses = FakeResponses(
        {
            "factual_consistency": 5,
            "coverage": 4,
            "relevance": 5,
            "conciseness": 4,
            "reason": "Accurate and focused, with one detail omitted.",
        }
    )
    client = SimpleNamespace(responses=responses)
    judge = OpenAISummaryJudge(client, model="judge-model")

    result = judge.judge("document", "Reference summary.", "Generated summary.")

    assert result.overall_score == 4.5
    assert result.model == "judge-model"
    assert result.reason == "Accurate and focused, with one detail omitted."
    call = responses.calls[0]
    assert call["model"] == "judge-model"
    assert call["text"]["format"]["type"] == "json_schema"
    assert "Reference summary." in call["input"][1]["content"]


@pytest.mark.parametrize(
    "output",
    [
        "not JSON",
        {
            "factual_consistency": 6,
            "coverage": 4,
            "relevance": 5,
            "conciseness": 4,
            "reason": "Invalid score.",
        },
        {
            "factual_consistency": 5,
            "coverage": 4,
            "relevance": 5,
            "conciseness": 4,
            "reason": "",
        },
    ],
)
def test_openai_summary_judge_rejects_invalid_output(output: object) -> None:
    judge = OpenAISummaryJudge(SimpleNamespace(responses=FakeResponses(output)))

    with pytest.raises(ValueError, match="summary judge"):
        judge.judge("document", "Reference.", "Generated.")
