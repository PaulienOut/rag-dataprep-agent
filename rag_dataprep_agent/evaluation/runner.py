from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_dataprep_agent.config import load_settings
from rag_dataprep_agent.evaluation.evaluator import evaluate_dataset
from rag_dataprep_agent.evaluation.judge import OpenAISummaryJudge
from rag_dataprep_agent.evaluation.loader import load_evaluation_dataset
from rag_dataprep_agent.llm.client import build_openai_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated manifests against ground truth.")
    parser.add_argument("--ground-truth-dir", default="data/ground_truth")
    parser.add_argument("--manifests-dir", default="prepared/manifests")
    parser.add_argument("--output", default="evaluation-results/baseline.json")
    parser.add_argument("--keyword-threshold", type=float, default=0.85)
    parser.add_argument(
        "--extraction-mode",
        choices=("local", "llm", "unknown"),
        default="unknown",
        help="How the evaluated manifests were produced",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Use OpenAI to judge generated summaries")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model used for summary judging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_evaluation_dataset(args.ground_truth_dir, args.manifests_dir)
    summary_judge = None
    if args.llm_judge:
        settings = load_settings()
        client = build_openai_client(settings.openai_api_key)
        if client is None:
            raise SystemExit("--llm-judge requires OPENAI_API_KEY")
        summary_judge = OpenAISummaryJudge(client, model=args.judge_model)

    report = evaluate_dataset(
        dataset,
        keyword_similarity_threshold=args.keyword_threshold,
        summary_judge=summary_judge,
    )
    report["configuration"].update(
        {
            "ground_truth_dir": str(Path(args.ground_truth_dir)),
            "manifests_dir": str(Path(args.manifests_dir)),
            "extraction_mode": args.extraction_mode,
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    aggregate = report["aggregate"]
    print(f"Wrote evaluation report to {output_path}")
    print(
        "Coverage: "
        f"{aggregate['matched_document_count']}/{aggregate['expected_document_count']} | "
        f"Type accuracy: {aggregate['document_type_accuracy']:.3f} | "
        f"Keyword F1: {aggregate['keyword_f1']:.3f}"
    )
    if aggregate["summary_judge_score"] is not None:
        print(
            "Summary judge: "
            f"{aggregate['summary_judge_score']:.3f}/5 "
            f"across {aggregate['summary_judged_document_count']} document(s)"
        )


if __name__ == "__main__":
    main()
