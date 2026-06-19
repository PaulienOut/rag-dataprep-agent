# Ground-Truth Evaluation Contract

This document defines what the document preparation evaluation measures. The
ground-truth files were created and reviewed manually.

## Evaluation Inputs

- Source PDFs: `data/Selection/`
- Ground truth: individual JSON files in `data/ground_truth/`
- Pipeline output: JSON manifests in `prepared/evaluation/manifests/`

`data/ground_truth/combined.json` is an overview file and is not evaluated as a
document.

## Document Matching

Documents are matched by filename stem:

```text
data/ground_truth/2605.00016v1.json
prepared/evaluation/manifests/2605.00016v1.json
```

A missing generated manifest is an evaluation failure for that document. A
generated manifest without matching ground truth is reported but does not
affect the main score.

## Evaluated Fields

| Field | Comparison |
| --- | --- |
| `document_type.document_type` | Exact match and aggregate accuracy |
| `content_metadata.title` | Normalized exact match and text similarity |
| `content_metadata.subject` | Normalized text similarity with explicit `null` handling |
| `content_metadata.keywords` | Normalized keyword precision, recall, and F1 with one-to-one character-similarity matching |
| `content_metadata.document_metadata.place` | Normalized text match |
| `content_metadata.document_metadata.date_of_publication` | Normalized date match when parseable, otherwise normalized text match |
| `content_metadata.layout_metadata.header` | Normalized text similarity with explicit `null` handling |
| `content_metadata.layout_metadata.footer` | Normalized text similarity with explicit `null` handling |
| `content_metadata.summary` | Optional OpenAI judge scoring factual consistency with the reference summary, coverage, relevance, and conciseness |

Deterministic metrics are always available. The summary judge is optional
because it makes live API calls and incurs OpenAI usage.

The summary judge compares generated and manually reviewed reference summaries.
It does not reread the full PDF, so factual consistency means consistency with
the reference summary rather than independent source-document verification.

Text normalization ignores capitalization, accents, punctuation, and repeated
whitespace. Keyword matching also permits small spelling variants using a
documented character-similarity threshold. It does not attempt synonym or
semantic matching.

## Null Values

`null` is a meaningful expected value:

- Expected `null`, generated `null`: correct.
- Expected value, generated `null`: missing value.
- Expected `null`, generated value: unexpected value.

This prevents missing metadata from receiving the same score as correct
metadata.

## Excluded Fields

The following fields are not part of metadata-quality scoring:

- File paths, sizes, and modification timestamps
- Raw PDF properties
- Detection confidence and evidence wording
- Chunk text, offsets, and embeddings

They are excluded because they are environment-dependent, implementation
details, or belong to a future retrieval evaluation.

## Report Requirements

Each evaluation report will contain:

- Evaluation timestamp and configuration
- Models and relevant pipeline settings
- Aggregate metrics
- Per-document field results
- Missing and unexpected manifests
- LLM-judge reasoning when summary judging is enabled

Reports should make failures inspectable, not only provide a single score.

## Baseline and Tuning

The first run establishes a baseline. Later experiments should change one
variable at a time, such as:

- Local extraction versus LLM extraction
- Metadata prompt
- Metadata model
- Amount of document text supplied to the model

Reports should be retained so changes can be compared with evidence.
