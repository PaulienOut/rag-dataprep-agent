# Keyword Extraction Experiment V1

This experiment tested whether a simple local phrase-ranking extractor improves
keyword quality over the original single-word frequency baseline.

## Baseline

The original extractor returned the most frequent non-stopwords. Its recorded
keyword metrics were:

| Metric | Baseline |
| --- | ---: |
| Precision | 0.089 |
| Recall | 0.107 |
| F1 | 0.097 |

The baseline report was preserved before running the experiment. The experiment
uses separate manifest and report paths, so it does not overwrite the baseline.

## Change

The new local extractor:

- Generates two- and three-word keyword candidates.
- Ranks candidates by frequency.
- Gives extra weight to phrases found in the title and first page.
- Removes phrases that begin or end with common stopwords.
- Suppresses strongly overlapping phrase candidates.
- Uses single words only when too few phrase candidates are available.

No prompt, OpenAI model, or evaluation metric was changed.

## Results

| Metric | Baseline | Experiment | Change |
| --- | ---: | ---: | ---: |
| Precision | 0.089 | 0.116 | +0.027 |
| Recall | 0.107 | 0.147 | +0.040 |
| F1 | 0.097 | 0.129 | +0.033 |

Keyword F1 improved by approximately 33.6% relative to the baseline.

The strongest improvement was `2605.00016v1`, where the extractor produced
phrases including:

- `disposition effect`
- `integrated framing`
- `short exposure`

These matched manually authored ground-truth phrases that the original
single-word extractor missed.

## Limitations Found

The change did not improve every document. It introduced noisy phrases such as:

- `et al`
- URL fragments
- connective fragments such as `if you`

It also sometimes preferred a longer phrase when the ground truth contained a
useful single keyword. Five documents improved, five regressed, and four were
unchanged.

The result supports phrase extraction as a useful direction, but this first
implementation should be treated as an intermediate experiment rather than a
finished keyword solution.

## Reproduce

```bash
make experiment-keyword-phrases
```

This writes:

```text
prepared/experiments/keyword-phrases-v1/manifests/
evaluation-results/keyword-phrases-v1.json
```

The next refinement should filter citation and URL noise and combine selected
phrases with high-value title terms rather than excluding single keywords
almost completely.
