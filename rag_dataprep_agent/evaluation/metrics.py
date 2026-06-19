from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Literal

NullStatus = Literal["both_null", "missing", "unexpected", "compared"]


@dataclass(frozen=True)
class TextComparison:
    expected: str | None
    generated: str | None
    normalized_expected: str | None
    normalized_generated: str | None
    status: NullStatus
    exact_match: bool
    similarity: float


@dataclass(frozen=True)
class KeywordMatch:
    expected: str
    generated: str
    similarity: float


@dataclass(frozen=True)
class KeywordMetrics:
    expected_count: int
    generated_count: int
    matches: list[KeywordMatch]
    precision: float
    recall: float
    f1: float


def normalize_text(value: str) -> str:
    """Normalize text without attempting semantic or synonym matching."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    words = re.findall(r"[^\W_]+", without_accents.casefold(), flags=re.UNICODE)
    return " ".join(words)


def text_similarity(expected: str, generated: str) -> float:
    normalized_expected = normalize_text(expected)
    normalized_generated = normalize_text(generated)
    if normalized_expected == normalized_generated:
        return 1.0
    if not normalized_expected or not normalized_generated:
        return 0.0
    return SequenceMatcher(None, normalized_expected, normalized_generated).ratio()


def compare_text(expected: str | None, generated: str | None) -> TextComparison:
    if expected is None and generated is None:
        return TextComparison(expected, generated, None, None, "both_null", True, 1.0)
    if expected is not None and generated is None:
        return TextComparison(expected, generated, normalize_text(expected), None, "missing", False, 0.0)
    if expected is None and generated is not None:
        return TextComparison(None, generated, None, normalize_text(generated), "unexpected", False, 0.0)

    assert expected is not None and generated is not None
    normalized_expected = normalize_text(expected)
    normalized_generated = normalize_text(generated)
    return TextComparison(
        expected=expected,
        generated=generated,
        normalized_expected=normalized_expected,
        normalized_generated=normalized_generated,
        status="compared",
        exact_match=normalized_expected == normalized_generated,
        similarity=text_similarity(expected, generated),
    )


def compare_date(expected: str | None, generated: str | None) -> TextComparison:
    if expected is None and generated is None:
        return TextComparison(expected, generated, None, None, "both_null", True, 1.0)
    if expected is not None and generated is None:
        return TextComparison(expected, generated, normalize_date(expected), None, "missing", False, 0.0)
    if expected is None and generated is not None:
        return TextComparison(None, generated, None, normalize_date(generated), "unexpected", False, 0.0)

    assert expected is not None and generated is not None
    normalized_expected = normalize_date(expected)
    normalized_generated = normalize_date(generated)
    return TextComparison(
        expected=expected,
        generated=generated,
        normalized_expected=normalized_expected,
        normalized_generated=normalized_generated,
        status="compared",
        exact_match=normalized_expected == normalized_generated,
        similarity=text_similarity(normalized_expected, normalized_generated),
    )


def normalize_date(value: str) -> str:
    compact = re.sub(r"\s+", " ", value.strip())
    compact = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", compact, flags=re.IGNORECASE)

    for date_format in (
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(compact, date_format).date().isoformat()
        except ValueError:
            continue

    for date_format in ("%B, %Y", "%B %Y", "%b, %Y", "%b %Y"):
        try:
            parsed = datetime.strptime(compact, date_format)
            return f"{parsed.year:04d}-{parsed.month:02d}"
        except ValueError:
            continue

    if re.fullmatch(r"\d{4}", compact):
        return compact
    return normalize_text(compact)


def compare_keywords(
    expected: list[str],
    generated: list[str],
    similarity_threshold: float = 0.85,
) -> KeywordMetrics:
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    expected_values = _unique_nonempty(expected)
    generated_values = _unique_nonempty(generated)
    candidates = [
        (text_similarity(expected_value, generated_value), expected_index, generated_index)
        for expected_index, expected_value in enumerate(expected_values)
        for generated_index, generated_value in enumerate(generated_values)
    ]

    used_expected: set[int] = set()
    used_generated: set[int] = set()
    matches: list[KeywordMatch] = []
    for similarity, expected_index, generated_index in sorted(candidates, reverse=True):
        if similarity < similarity_threshold:
            break
        if expected_index in used_expected or generated_index in used_generated:
            continue
        used_expected.add(expected_index)
        used_generated.add(generated_index)
        matches.append(
            KeywordMatch(
                expected=expected_values[expected_index],
                generated=generated_values[generated_index],
                similarity=similarity,
            )
        )

    match_count = len(matches)
    precision = _safe_ratio(match_count, len(generated_values), empty_score=1.0 if not expected_values else 0.0)
    recall = _safe_ratio(match_count, len(expected_values), empty_score=1.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return KeywordMetrics(
        expected_count=len(expected_values),
        generated_count=len(generated_values),
        matches=sorted(matches, key=lambda match: normalize_text(match.expected)),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _unique_nonempty(values: list[str]) -> list[str]:
    unique: dict[str, str] = {}
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in unique:
            unique[normalized] = value
    return list(unique.values())


def _safe_ratio(numerator: int, denominator: int, empty_score: float) -> float:
    return numerator / denominator if denominator else empty_score
