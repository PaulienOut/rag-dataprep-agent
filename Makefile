.PHONY: run-sample pipeline pipeline-full tests prepare-evaluation evaluate evaluate-llm experiment-keyword-phrases experiment-keywords-llm

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

experiment-keyword-phrases:
	uv run python -m rag_dataprep_agent.cli "data/Selection" \
		--output-dir prepared/experiments/keyword-phrases-v1
	uv run python -m rag_dataprep_agent.evaluation.runner \
		--manifests-dir prepared/experiments/keyword-phrases-v1/manifests \
		--extraction-mode local \
		--output evaluation-results/keyword-phrases-v1.json

experiment-keywords-llm:
	uv run python -m rag_dataprep_agent.cli "data/Selection" \
		--output-dir prepared/experiments/keywords-llm-gpt-4o-mini \
		--use-llm
	uv run python -m rag_dataprep_agent.evaluation.runner \
		--manifests-dir prepared/experiments/keywords-llm-gpt-4o-mini/manifests \
		--extraction-mode llm \
		--extraction-model gpt-4o-mini \
		--output evaluation-results/keywords-llm-gpt-4o-mini.json
