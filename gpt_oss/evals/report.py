"""
Report Generation and Result Aggregation Module

This module provides utilities for:
1. Aggregating individual evaluation results into summary statistics
2. Generating HTML reports for human review
3. Parallel processing of evaluation examples
4. Computing statistical metrics (mean, std, bootstrap estimates)

Key Components:
- aggregate_results(): Combines SingleEvalResults into an EvalResult
- map_with_progress(): Parallel execution with progress bar
- make_report(): Generates standalone HTML report
- message_to_html(): Formats conversation messages for display

Statistical Methods:
- Bootstrap standard deviation for variance estimation
- Support for custom statistics per metric
- Automatic mean/std computation for all metrics
"""

import os
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Callable

import jinja2
import numpy as np
from tqdm import tqdm

from .types import EvalResult, Message, SingleEvalResult


# Default HTML template for displaying individual evaluation examples
# Shows: prompt conversation, model response, correct answer, extracted answer, and score
HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""


def _compute_stat(values: list, stat: str):
    """
    Compute a statistical measure over a list of values.

    Supported statistics:
    - mean: Average of all values
    - std: Standard deviation
    - min/max: Minimum/maximum value
    - n_samples: Number of samples
    - bootstrap_std: Bootstrap estimate of standard error of the mean
        (more robust than std for small sample sizes)

    Args:
        values: List of numeric values
        stat: Name of the statistic to compute

    Returns:
        The computed statistic as a float
    """
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        # Bootstrap resampling to estimate standard error of the mean
        # Samples with replacement 1000 times and computes std of means
        # This provides a robust estimate of uncertainty
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate multiple SingleEvalResults into a final EvalResult.

    This function:
    1. Collects all metric values across examples
    2. Computes summary statistics for each metric
    3. Combines HTML outputs and conversations
    4. Returns an EvalResult with aggregated data

    Args:
        single_eval_results: List of results from individual examples
        default_stats: Statistics to compute for all metrics (default: mean and std)
        name2stats: Optional dict mapping specific metric names to custom statistics
            Example: {"accuracy": ("mean", "bootstrap_std", "n_samples")}

    Returns:
        EvalResult with:
        - score: Mean of individual scores
        - metrics: Dictionary of aggregated statistics
            - "metric_name" (if stat is "mean")
            - "metric_name:stat" (for other statistics)
        - htmls: List of HTML snippets for each example
        - convos: List of complete conversations
        - metadata: Combined metadata from all examples
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)  # Collect values for each metric
    htmls = []
    convos = []
    metadata = []

    # Collect all individual results
    for single_eval_result in single_eval_results:
        # Gather all metrics from this example
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        # Also collect the primary score
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)

    # Compute statistics for each metric
    final_metrics = {}
    for name, values in name2values.items():
        # Use custom stats if specified, otherwise use defaults
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            # Store as "metric" for mean, "metric:stat" for others
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)

    # Return aggregated result
    return EvalResult(
        score=final_metrics.pop("score", None),  # Extract primary score
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = 128,
    pbar: bool = True,
):
    """
    Apply a function to each element in parallel with a progress bar.

    This is the core parallelization function used by all evaluations to process
    examples concurrently. It provides:
    - Parallel execution using ThreadPool
    - Progress bar via tqdm
    - Debug mode support (sequential execution)

    Args:
        f: Function to apply to each element (e.g., evaluate_single_example)
        xs: List of inputs to process (e.g., list of evaluation examples)
        num_threads: Maximum number of parallel threads (default: 128)
        pbar: Whether to show progress bar (default: True)

    Returns:
        List of results from applying f to each element of xs
        Note: Results may be in a different order than inputs (uses imap_unordered)

    Example:
        def evaluate_example(example):
            response = model(example["prompt"])
            return score_response(response, example["answer"])

        results = map_with_progress(evaluate_example, dataset, num_threads=64)
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    # In debug mode, run sequentially for easier debugging
    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        # Parallel execution using ThreadPool
        # Use min(num_threads, len(xs)) to avoid creating unnecessary threads
        with ThreadPool(min(num_threads, len(xs))) as pool:
            # imap_unordered is faster than map as it doesn't preserve order
            return list(pbar_fn(pool.imap_unordered(f, xs), total=len(xs)))


# Set up Jinja2 environment for HTML template rendering
# - StrictUndefined: Raise errors for undefined variables (catches bugs)
# - autoescape: Automatically escape HTML/XML to prevent XSS
jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)

# Template for rendering a single message in the conversation
# Each message is displayed with role (user/assistant/system) and content
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Message) -> str:
    """
    Convert a message dictionary to an HTML snippet.

    Args:
        message: Dictionary with keys:
            - "role": Message role (user/assistant/system/developer)
            - "content": Message text
            - "variant": Optional variant label (e.g., "reasoning")

    Returns:
        HTML string displaying the message with appropriate styling
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


# Make message_to_html available within Jinja2 templates
jinja_env.globals["message_to_html"] = message_to_html


# Complete HTML report template
# Displays:
# 1. Summary table with overall score and all metrics
# 2. Individual examples with full conversations and results
_report_template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Generate a standalone HTML report from evaluation results.

    This creates a complete, self-contained HTML file that can be opened
    in a browser for reviewing evaluation results. The report includes:
    - Summary metrics table with score and all computed statistics
    - Individual examples showing prompts, responses, and scores
    - CSS styling for readable presentation
    - Color-coded message roles (user, assistant, system)

    Args:
        eval_result: The aggregated results from an evaluation run

    Returns:
        Complete HTML document as a string

    The generated report is saved to /tmp/ and can be opened directly
    in a web browser for human review of model performance.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )
