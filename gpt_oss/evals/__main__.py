"""
Main Evaluation Runner Module

This module provides the command-line interface for running model evaluations.
It supports multiple evaluation benchmarks and sampler backends, with configurable
parameters for model selection, reasoning effort, temperature, and more.

Supported Evaluations:
- basic: Simple sanity check evaluation
- gpqa: Graduate-Level Google-Proof Q&A benchmark
- aime25: American Invitational Mathematics Examination 2025
- healthbench: Medical question answering benchmark (including hard and consensus variants)

Supported Samplers:
- responses: OpenAI Responses API (recommended for reasoning models)
- chat_completions: OpenAI Chat Completions API (standard completion API)

Usage:
    python -m gpt_oss.evals --model gpt-oss-120b --eval gpqa --sampler responses
"""

import argparse
import json
from datetime import datetime

from . import report
from .basic_eval import BasicEval
from .gpqa_eval import GPQAEval
from .aime_eval import AIME25Eval
from .healthbench_eval import HealthBenchEval
from .chat_completions_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionsSampler,
)
from .responses_sampler import ResponsesSampler


def main():
    """
    Main entry point for running evaluations from the command line.

    This function:
    1. Parses command-line arguments to configure the evaluation
    2. Creates sampler instances for each model/reasoning-effort combination
    3. Creates a grading sampler (GPT-4.1) for evaluations that require grading
    4. Runs the specified evaluations on all model configurations
    5. Generates HTML reports and JSON result files for each run
    6. Saves all results to /tmp/ directory with timestamped filenames
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b,gpt-oss-20b",
        help="Select a model by name. Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low,medium,high",
        help="Reasoning effort (low, medium, high). Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["responses", "chat_completions"],
        default="responses",
        help="Sampler backend to use for models.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the API.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="gpqa,healthbench,healthbench_hard,healthbench_consensus,aime25",
        help="Select an eval by name. Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1584,
        help="Number of threads to run.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode"
    )
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()

    # Select the appropriate sampler class based on command-line argument
    # ResponsesSampler is recommended for reasoning models, ChatCompletionsSampler for standard models
    sampler_cls = ResponsesSampler if args.sampler == "responses" else ChatCompletionsSampler

    # Create sampler instances for all combinations of models and reasoning efforts
    # This allows testing multiple configurations in a single run
    # Example: "gpt-oss-120b,gpt-oss-20b" with "low,medium,high" creates 6 samplers
    models = {}
    for model_name in args.model.split(","):
        for reasoning_effort in args.reasoning_effort.split(","):
            models[f"{model_name}-{reasoning_effort}"] = sampler_cls(
                model=model_name,
                reasoning_model=True,
                reasoning_effort=reasoning_effort,
                temperature=args.temperature,
                base_url=args.base_url,
                max_tokens=131_072,  # Maximum output tokens supported by the model
            )

    print(f"Running with args {args}")

    # Create a grading sampler using GPT-4.1 for evaluations that require automated grading
    # This is used by HealthBench to grade responses against rubrics
    grading_sampler = ChatCompletionsSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        base_url="https://api.openai.com/v1",
    )

    def get_evals(eval_name, debug_mode):
        """
        Factory function to create evaluation instances based on name.

        Args:
            eval_name: Name of the evaluation (basic, gpqa, healthbench, etc.)
            debug_mode: If True, uses reduced dataset sizes for faster testing

        Returns:
            An Eval instance configured for the specified benchmark
        """
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "basic":
                # Simple sanity check: verifies the model can produce non-empty responses
                return BasicEval()
            case "gpqa":
                # GPQA: Graduate-level science questions with multiple-choice answers
                # n_repeats=8 provides variance estimates through repeated sampling
                return GPQAEval(
                    n_repeats=1 if args.debug else 8,
                    num_examples=num_examples,
                    debug=debug_mode,
                    n_threads=args.n_threads or 1,
                )
            case "healthbench":
                # HealthBench: Medical Q&A evaluated using detailed rubrics
                # Requires grader_model to assess responses against expert criteria
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=1,
                    n_threads=args.n_threads or 1,
                    subset_name=None,  # Full dataset
                )
            case "healthbench_hard":
                # HealthBench Hard: Subset of particularly challenging medical questions
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=1,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                )
            case "healthbench_consensus":
                # HealthBench Consensus: Questions with high physician agreement
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=1,
                    n_threads=args.n_threads or 1,
                    subset_name="consensus",
                )
            case "aime25":
                # AIME 2025: Advanced high school mathematics competition
                # Answers are integers 0-999, n_repeats=8 for variance estimation
                return AIME25Eval(
                    n_repeats=1 if args.debug else 8,
                    num_examples=num_examples,
                    n_threads=args.n_threads or 1,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    # Create evaluation instances for all requested eval types
    evals = {}
    for eval_name in args.eval.split(","):
        evals[eval_name] = get_evals(eval_name, args.debug)

    # Add debug suffix to output filenames when in debug mode
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}  # Tracks result files for final summary
    print(f"Running the following evals: {evals}")
    print(f"Running evals for the following models: {models}")

    # Generate timestamp for unique filenames
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")

    # Run all combinations of models and evaluations
    for model_name, sampler in models.items():
        # Replace slashes in model names to avoid path issues
        model_name = model_name.replace("/", "__")
        for eval_name, eval_obj in evals.items():
            # Execute the evaluation by calling the eval object with the sampler
            result = eval_obj(sampler)
            # The eval returns an EvalResult containing score, metrics, htmls, and convos

            # Create a unique filename stem for this evaluation run
            file_stem = f"{eval_name}_{model_name}_temp{args.temperature}"
            file_stem += f"_{date_str}"

            # Save HTML report for human-readable analysis
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(report.make_report(result))

            # Save summary metrics in JSON format
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            metrics = dict(sorted(metrics.items()))  # Sort metrics alphabetically
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            # Save complete results including all conversations and HTML outputs
            # This is useful for detailed analysis and debugging
            full_result_filename = f"/tmp/{file_stem}{debug_suffix}_allresults.json"
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename

    # Aggregate all results into a summary table
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        # Extract the primary metric (f1_score for some evals, score for others)
        result = result.get("f1_score", result.get("score", None))
        # Parse eval and model names from the filename
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    print(merge_metrics)
    return merge_metrics


if __name__ == "__main__":
    main()
