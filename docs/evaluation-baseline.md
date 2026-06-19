# Evaluation Baseline

This baseline was recorded on June 19, 2026 using the 14 manually reviewed
documents in `data/Selection/` and `data/ground_truth/`.

## Configuration

- Extraction mode: local deterministic extraction
- Summary judge: `gpt-4o-mini`
- Keyword similarity threshold: `0.85`
- Documents expected and evaluated: 14
- Missing or unexpected manifests: none

The local extraction baseline is intentional. It gives later LLM extraction or
prompt experiments a stable point of comparison.

## Deterministic Results

| Metric | Score |
| --- | ---: |
| Manifest coverage | 1.000 |
| Document type accuracy | 1.000 |
| Title exact match | 0.429 |
| Title similarity | 0.639 |
| Subject similarity | 0.429 |
| Keyword precision | 0.089 |
| Keyword recall | 0.107 |
| Keyword F1 | 0.097 |
| Place similarity | 0.714 |
| Date exact match | 0.357 |
| Date similarity | 0.572 |
| Header similarity | 0.643 |
| Footer similarity | 0.454 |

These scores should be treated as diagnostic signals, not as a single measure
of overall quality. The JSON report contains the expected and generated values
needed to inspect each low score.

## Summary Judge Results

The optional OpenAI judge evaluated all 14 generated summaries.

| Criterion | Mean score (1-5) |
| --- | ---: |
| Factual consistency with reference | 3.500 |
| Coverage | 2.929 |
| Relevance | 4.286 |
| Conciseness | 3.357 |
| Overall | 3.518 |

The judge compares against the manually reviewed reference summary. It does not
independently reread or fact-check the complete PDF. Scores can vary slightly
between runs because an LLM judge is nondeterministic.

## Initial Findings

1. Document-type detection is strong for the selected dataset.
2. Place extraction performs relatively well.
3. Generated summaries are generally relevant, but often omit important
   conclusions or details. Coverage is the weakest summary dimension.
4. Local keyword extraction is the clearest weakness. It often returns frequent
   individual words such as `effect`, `disposition`, and `exposure`, while the
   ground truth contains meaningful phrases such as `Disposition effect` and
   `Systematic risk exposure`.
5. Some low text scores may reflect metric strictness or valid wording
   alternatives. These cases require manual inspection before changing either
   extraction logic or metrics.

## Reproducing the Baseline

Run deterministic extraction and evaluation:

```bash
make evaluate
```

Run the same baseline with live summary judging:

```bash
make evaluate-llm
```

The detailed local reports are written to:

```text
evaluation-results/baseline.json
evaluation-results/baseline-llm.json
```

## Next Tuning Process

For each weak field:

1. Review low-scoring document examples.
2. Classify each case as poor output, strict metric, valid alternative, or
   ground-truth correction.
3. Change one extraction or metric variable at a time.
4. Rerun evaluation and compare with this baseline.

The first recommended experiment is improving keyword extraction to produce
topic phrases rather than isolated frequent words.
