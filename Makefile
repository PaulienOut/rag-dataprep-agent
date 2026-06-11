run-sample:
	uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --max-files 3

pipeline:
	uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared

pipeline-full:
	uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --use-llm --embed

tests:
	uv run pytest

