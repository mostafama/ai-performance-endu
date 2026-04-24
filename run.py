"""
run.py
------
Unified command-line entry point for the EduQuestion3 pipeline.

Three subcommands:
  label     — Assign Bloom's taxonomy levels to questions using an LLM classifier
  query     — Query LLMs and save raw responses
  evaluate  — Score responses with LLM judge and lexical metrics

Usage examples:

  # Assign Bloom labels to all unlabelled questions in questions.csv
  python run.py label --questions questions.csv

  # Re-classify all questions from scratch
  python run.py label --questions questions.csv --relabel

  # Query using a named study run from config.py
  python run.py query --run gpt_gemini

  # Query with custom model + file arguments
  python run.py query --models gpt-4o-mini claude-sonnet-4.5 \
      --questions questions.csv --output my_responses.csv

  # Evaluate using a named config from config.py
  python run.py evaluate --config gpt_gemini

  # Evaluate with custom arguments
  python run.py evaluate --responses my_responses.csv \
      --questions questions.csv --output my_scores.csv
"""

import argparse

import config
from data_loader import load_questions
from labeler import QuestionLabeler
from querier import LLMQuerier
from evaluator import ResponseEvaluator


def run_label(args):
    """Handle the 'label' subcommand."""
    for questions_file in args.questions:
        print(f"\n=== Labelling: {questions_file} ===")
        QuestionLabeler(
            questions_csv=questions_file,
            model=args.model,
            relabel=args.relabel,
        ).run()


def run_query(args):
    """Handle the 'query' subcommand."""
    if args.run:
        run_names = list(config.STUDY_RUNS.keys()) if args.run == "all" else [args.run]
        for name in run_names:
            cfg = config.STUDY_RUNS[name]
            print(f"\n=== Querying: {name} ===")
            questions_df = load_questions(
                cfg["questions"],
                skip_datasets=args.skip_datasets or cfg.get("skip_datasets", []),
            )
            LLMQuerier(cfg["models"], questions_df, cfg["output"]).run()
    else:
        if not args.questions or not args.output:
            raise ValueError("--questions and --output are required when using --models")
        questions_df = load_questions(args.questions, skip_datasets=args.skip_datasets)
        LLMQuerier(args.models, questions_df, args.output).run()


def run_evaluate(args):
    """Handle the 'evaluate' subcommand."""
    if args.config:
        cfg_names = list(config.EVALUATION_CONFIGS.keys()) if args.config == "all" else [args.config]
        for name in cfg_names:
            cfg = config.EVALUATION_CONFIGS[name]
            print(f"\n=== Evaluating: {name} — {cfg['description']} ===")
            ResponseEvaluator(
                cfg["questions_csv"],
                cfg["responses_csv"],
                cfg["output_csv"],
                cfg["judge_model"],
            ).run()
    else:
        if not args.questions or not args.output:
            raise ValueError("--questions and --output are required when using --responses")
        ResponseEvaluator(args.questions, args.responses, args.output, args.judge).run()


def main():
    parser = argparse.ArgumentParser(
        description="EduQuestion3 — label questions, query LLMs, and evaluate responses"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- label subcommand ---
    lp = subparsers.add_parser(
        "label",
        help="Assign Bloom's taxonomy levels to questions"
    )
    lp.add_argument(
        "--questions", nargs="+", required=True,
        help="Question CSV file(s) to label"
    )
    lp.add_argument(
        "--model", default="gpt-4.1-mini",
        help="LLM to use as the Bloom classifier (default: gpt-4.1-mini)"
    )
    lp.add_argument(
        "--relabel", action="store_true",
        help="Re-classify questions that already have a bloom_level"
    )

    # --- query subcommand ---
    qp = subparsers.add_parser("query", help="Query LLMs with benchmark questions")
    qp_group = qp.add_mutually_exclusive_group(required=True)
    qp_group.add_argument(
        "--run",
        choices=list(config.STUDY_RUNS.keys()) + ["all"],
        help="Named study run from config.STUDY_RUNS",
    )
    qp_group.add_argument(
        "--models", nargs="+", choices=list(config.MODELS_CONFIG.keys()),
        help="Explicit list of models to query",
    )
    qp.add_argument("--questions", nargs="+", help="Question CSV/Excel file paths")
    qp.add_argument("--output", help="Output responses CSV path")
    qp.add_argument("--skip-datasets", nargs="+", default=[], dest="skip_datasets")

    # --- evaluate subcommand ---
    ep = subparsers.add_parser("evaluate", help="Score LLM responses")
    ep_group = ep.add_mutually_exclusive_group(required=True)
    ep_group.add_argument(
        "--config",
        choices=list(config.EVALUATION_CONFIGS.keys()) + ["all"],
        help="Named evaluation config from config.EVALUATION_CONFIGS",
    )
    ep_group.add_argument("--responses", help="Responses CSV path")
    ep.add_argument("--questions", nargs="+", help="Question CSV/Excel file paths")
    ep.add_argument("--output", help="Output scores CSV path")
    ep.add_argument("--judge", default="gpt-4.1-mini", help="Judge model name")

    args = parser.parse_args()

    if args.command == "label":
        run_label(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
