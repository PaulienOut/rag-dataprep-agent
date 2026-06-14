# Document Preparation Agent for RAG Knowledge Bases

An AI Engineering Buildcamp project for improving document readiness and data quality before files are used in retrieval-augmented generation (RAG) knowledge bases.

## The Problem

Data quality is a common challenge in RAG projects. Documents often arrive in mixed formats, with inconsistent metadata, unknown topics, noisy content, and sensitive information that may need to be handled before indexing.

## What It Does

The project aims to build a document preparation agent that inspects a file and applies the relevant preparation steps for a RAG knowledge base.

The first version supports PDF documents and focuses on three starter document classes:

- Arxiv papers
- Manuals
- EU documents

The folders in `data/` are labeled examples for development and testing. In real use, the documents do not need to be sorted into folders. The pipeline scans a mixed input folder and infers the document type from the filename, PDF metadata, and extracted text.

The agent currently supports:

- Metadata extraction from document properties and, when useful, from the document body or content. For example, an image may be described based on its visual content, while a Word document may provide title, subject, and other text-derived metadata. Across file types, the agent should extract common properties such as file type and last edited date when available.
- Document type detection for Arxiv papers, manuals, EU documents, or unknown documents.
- Content metadata extraction using local heuristics, with an optional OpenAI call for better title, subject, summary, and keywords.
- PDF parsing, text chunking, and optional embeddings so prepared documents can be added to a RAG index.

The exact metadata fields and classification categories will be refined as the project develops.

## Project Structure

- `rag_dataprep_agent/` - Python package containing the document preparation pipeline.
- `rag_dataprep_agent/inventory/` - scans input files without relying on folder names.
- `rag_dataprep_agent/parsers/` - parses supported document formats. Version 1 supports PDFs.
- `rag_dataprep_agent/tools/` - deterministic tools for metadata extraction, type detection, chunking, and local fallback embeddings.
- `rag_dataprep_agent/llm/` - OpenAI client, prompts, and a small agent wrapper that calls LLMs when enabled.
- `rag_dataprep_agent/storage/` - writes prepared JSON manifests.
- `tests/` - pytest tests for the deterministic parts of the pipeline.
- `data/` - example documents for the three starter classes.
- `prepared/` - generated output folder created when the pipeline runs.

## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file from the template and add your API key:

       cp .env.example .env

4. Install dependencies:

       uv sync

## Run the Pipeline

Run the deterministic local version on a small sample:

       uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --max-files 3

Create chunk embeddings as well:

       uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --max-files 3 --embed

Use OpenAI for richer content metadata extraction:

       uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --max-files 3 --use-llm

Use OpenAI for metadata extraction and embeddings:

       uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --max-files 3 --use-llm --embed

Enable Logfire monitoring for a run:

       uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --max-files 3 --logfire

The output is written as one JSON manifest per document in:

       prepared/manifests/

Each manifest contains source file metadata, PDF metadata, detected document type, extracted content metadata, chunks, and embeddings when enabled.

## Monitoring with Logfire

Logfire support is built into the CLI and pipeline. Monitoring is disabled by default so local tests and demo runs stay quiet.

1. Authenticate for local development:

       uv run logfire auth
       uv run logfire projects use <your-project>

   For production or CI, set a project write token instead:

       LOGFIRE_TOKEN=<your-write-token>

2. Enable monitoring for a single CLI run:

       uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --max-files 3 --logfire

   Or enable it from `.env`:

       LOGFIRE_ENABLED=true
       LOGFIRE_SERVICE_NAME=rag-dataprep-agent
       LOGFIRE_ENVIRONMENT=development

When enabled, Logfire records spans for the full preparation run and each document, plus summary events with document type, page count, chunk count, and manifest path. OpenAI metadata and embedding calls are also instrumented when `--use-llm` or remote embeddings are active.

## Testing

Run the test suite with:

       uv run pytest

The default tests do not call OpenAI. They test file scanning, PDF parsing, document type detection, chunking, and the pipeline using local deterministic behavior.

## Data

Put PDF documents in the `data/` folder, or pass the path to another PDF or folder when running the pipeline.
