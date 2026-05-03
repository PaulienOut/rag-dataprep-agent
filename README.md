# Document Preparation Agent for RAG Knowledge Bases

An AI Engineering Buildcamp project for improving document readiness and data quality before files are used in retrieval-augmented generation (RAG) knowledge bases.

## The Problem

Data quality is a common challenge in RAG projects. Documents often arrive in mixed formats, with inconsistent metadata, unknown topics, noisy content, and sensitive information that may need to be handled before indexing.

## What It Does

The project aims to build a document preparation agent that inspects a file and applies the relevant preparation steps for a RAG knowledge base.

The agent will support:

- Metadata extraction from document properties and, when useful, from the document body or content. For example, an image may be described based on its visual content, while a Word document may provide title, subject, and other text-derived metadata. Across file types, the agent should extract common properties such as file type and last edited date when available.
- Document classification with an NLP approach, initially focused on assigning topic categories.
- Parsing, chunking, and embedding so prepared documents can be added to a RAG index.
- Time permitting, pseudonymisation of sensitive information before downstream use.

The exact metadata fields and classification categories will be refined as the project develops.

## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file from the template and add your API key:

       cp .env.example .env

4. Install dependencies:

       uv sync

5. Start Jupyter:

       uv run jupyter notebook

## Notebooks

- `notebooks/01-setup.ipynb` - smoke test that confirms your environment works
- `notebooks/02-rag.ipynb` - a minimal RAG baseline you can adapt to your own data

## Data

Put your project data in the `data/` folder. See `notebooks/02-rag.ipynb` for how to load it.
