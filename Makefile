.PHONY: run-sample pipeline pipeline-full tests prepare-evaluation evaluate evaluate-llm

run-sample:
	uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --max-files 3

pipeline:
	uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared

pipeline-full:
	uv run python -m rag_dataprep_agent.cli "data" --output-dir prepared --use-llm --embed

tests:
	uv run pytest

prepare-evaluation:
	uv run python -m rag_dataprep_agent.cli "data/Selection" --output-dir prepared/evaluation

evaluate: prepare-evaluation
	uv run python -m rag_dataprep_agent.evaluation.runner \
		--manifests-dir prepared/evaluation/manifests \
		--extraction-mode local

evaluate-llm: prepare-evaluation
	uv run python -m rag_dataprep_agent.evaluation.runner \
		--manifests-dir prepared/evaluation/manifests \
		--extraction-mode local \
		--llm-judge \
		--output evaluation-results/baseline-llm.json
