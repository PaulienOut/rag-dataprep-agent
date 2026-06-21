# LLM Metadata Extraction Experiment

This experiment tested whether the existing OpenAI metadata extraction workflow
produces better keywords than local deterministic extraction.

## Configuration

- Extraction model: `gpt-4o-mini`
- Documents: 14 manually reviewed evaluation documents
- Evaluation metric and threshold: unchanged
- Summary judge: disabled
- Embeddings: disabled

The experiment made one live metadata extraction call per document. It did not
use an LLM judge, so the keyword comparison remains deterministic after
extraction.

## Results

| Approach | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Original local single words | 0.089 | 0.107 | 0.097 |
| Local phrase extraction V1 | 0.116 | 0.147 | 0.129 |
| `gpt-4o-mini` extraction | 0.584 | 0.586 | 0.582 |

The LLM keyword F1 is approximately six times the original baseline and more
than four times the local phrase experiment. It improved keyword F1 for all 14
documents relative to the original local extractor, with no regressions.

Two documents received a perfect keyword F1:

- `2605.00016v1`
- `2605.04004v1`

For example, `2605.00016v1` produced the expected topic phrases:

- `Disposition effect`
- `Systematic risk exposure`
- `Short exposure`
- `Integrated framing`
- `Prospect Theory`
- `Regret Theory`

## Broader Metadata Effects

The `--use-llm` option extracts all content metadata, not keywords alone.
Compared with the original local baseline, the same run also changed:

| Metric | Local baseline | `gpt-4o-mini` |
| --- | ---: | ---: |
| Title similarity | 0.639 | 0.996 |
| Subject similarity | 0.429 | 0.741 |
| Place similarity | 0.714 | 0.833 |
| Date similarity | 0.572 | 0.869 |
| Header similarity | 0.643 | 0.657 |
| Footer similarity | 0.454 | 0.695 |

Therefore this experiment supports LLM-based metadata extraction generally; it
does not isolate a prompt change affecting only keywords.

## Remaining Keyword Gaps

The lowest LLM keyword F1 scores were approximately `0.286` to `0.333`. Common
differences included:

- Splitting a phrase into separate keywords, such as `Oracle` and `Text`.
- Closely related wording that the character metric does not match, such as
  `EU regulations` versus `EU rules`.
- Related grammatical forms, such as `legislative` versus `legislation`.

These cases may involve extraction quality, metric strictness, or both. They
should be reviewed before changing the metric.

## Reproduce

```bash
make experiment-keywords-llm
```

This writes to separate experiment paths:

```text
prepared/experiments/keywords-llm-gpt-4o-mini/manifests/
evaluation-results/keywords-llm-gpt-4o-mini.json
```

It does not overwrite the baseline or local phrase experiment reports.

## Conclusion

The existing `gpt-4o-mini` workflow provides a substantial measured
improvement. Testing a stronger model is not the next necessary step. A better
next comparison would be cost and latency versus quality, or targeted prompt
refinement for the remaining low-scoring cases.
