from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from rag_dataprep_agent.config import load_settings
from rag_dataprep_agent.observability import configure_logfire
from rag_dataprep_agent.pipeline import prepare_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PDFs for a RAG knowledge base.")
    parser.add_argument("input_path", help="PDF file or folder containing PDFs")
    parser.add_argument("--output-dir", default="prepared", help="Directory for prepared JSON manifests")
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI for content metadata extraction")
    parser.add_argument("--embed", action="store_true", help="Create embeddings for chunks")
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of files to process")
    parser.add_argument("--logfire", action="store_true", help="Enable Logfire monitoring for this run")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        parser.error(f"input path does not exist: {input_path}")

    settings = load_settings()
    if args.logfire:
        settings = replace(settings, logfire_enabled=True)
    configure_logfire(settings)
    written = prepare_documents(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        settings=settings,
        use_llm=args.use_llm,
        embed=args.embed,
        max_files=args.max_files,
    )
    print(f"Wrote {len(written)} manifest(s) to {Path(args.output_dir) / 'manifests'}")


if __name__ == "__main__":
    main()
