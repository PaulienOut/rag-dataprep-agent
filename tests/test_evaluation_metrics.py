import pytest

from rag_dataprep_agent.evaluation.metrics import (
    compare_date,
    compare_keywords,
    compare_text,
    normalize_date,
    normalize_text,
)


def test_normalize_text_handles_case_accents_punctuation_and_whitespace() -> None:
    assert normalize_text("  Règles-based,   Evaluation! ") == "regles based evaluation"


def test_compare_text_reports_null_cases_explicitly() -> None:
    assert compare_text(None, None).status == "both_null"
    assert compare_text("Brussels", None).status == "missing"
    assert compare_text(None, "Brussels").status == "unexpected"


def test_compare_text_separates_normalized_exact_match_from_similarity() -> None:
    exact = compare_text("Council of the EU", "council-of-the-eu")
    similar = compare_text("Council of the European Union", "Council of European Union")

    assert exact.exact_match is True
    assert exact.similarity == 1.0
    assert similar.exact_match is False
    assert 0.85 < similar.similarity < 1.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("18 April 2026", "2026-04-18"),
        ("April 18, 2026", "2026-04-18"),
        ("18th April 2026", "2026-04-18"),
        ("18/04/2026", "2026-04-18"),
        ("September, 2023", "2023-09"),
        ("2010", "2010"),
    ],
)
def test_normalize_date_supports_ground_truth_date_shapes(value: str, expected: str) -> None:
    assert normalize_date(value) == expected


def test_compare_date_matches_different_representations_of_the_same_date() -> None:
    comparison = compare_date("18 April 2026", "April 18, 2026")

    assert comparison.expected == "18 April 2026"
    assert comparison.generated == "April 18, 2026"
    assert comparison.normalized_expected == "2026-04-18"
    assert comparison.normalized_generated == "2026-04-18"
    assert comparison.exact_match is True
    assert comparison.similarity == 1.0


def test_keyword_metrics_use_one_to_one_matching_and_allow_spelling_variance() -> None:
    metrics = compare_keywords(
        expected=["organic farming", "EU rules", "labeling"],
        generated=["Organic Farming", "labelling", "agriculture"],
    )

    assert [(match.expected, match.generated) for match in metrics.matches] == [
        ("labeling", "labelling"),
        ("organic farming", "Organic Farming"),
    ]
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)


def test_keyword_metrics_do_not_match_unrelated_terms() -> None:
    metrics = compare_keywords(["organic farming"], ["database indexing"])

    assert metrics.matches == []
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_keyword_metrics_handle_empty_lists() -> None:
    both_empty = compare_keywords([], [])
    unexpected = compare_keywords([], ["extra"])
    missing = compare_keywords(["expected"], [])

    assert (both_empty.precision, both_empty.recall, both_empty.f1) == (1.0, 1.0, 1.0)
    assert (unexpected.precision, unexpected.recall, unexpected.f1) == (0.0, 1.0, 0.0)
    assert (missing.precision, missing.recall, missing.f1) == (0.0, 0.0, 0.0)


def test_keyword_threshold_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        compare_keywords([], [], similarity_threshold=1.1)
