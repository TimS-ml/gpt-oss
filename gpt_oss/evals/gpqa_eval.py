"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

Paper: "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"
Authors: David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty,
         Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
URL: https://arxiv.org/abs/2311.12022

About GPQA:
- Graduate-level science questions in biology, physics, and chemistry
- Questions are "Google-proof": hard to solve by searching online
- Written and validated by domain experts with PhDs
- Designed to require deep scientific knowledge and reasoning

Dataset Variants:
- diamond: Main benchmark set (198 questions) - highest quality
- extended: Larger set with additional questions
- main: Balanced set between quality and size

Evaluation Details:
- Format: Multiple-choice (4 options: A, B, C, D)
- Scoring: Exact match on letter choice (1.0 correct, 0.0 incorrect)
- Answer Extraction: Uses abcd_grader module to parse model responses
- Multiple Attempts: By default, each question is asked 8 times (n_repeats=8)
  with randomized answer order to measure consistency

This benchmark tests:
- Deep scientific knowledge across multiple domains
- Ability to reason about complex graduate-level concepts
- Resistance to retrieval-based shortcuts
- Consistency across answer order permutations
"""

import random

import pandas

from . import report
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from .abcd_grader import extract_abcd


# Template for formatting multiple-choice questions
# Presents the question followed by four options labeled A-D
QUERY_TEMPLATE_MULTICHOICE = """
{Question}

(A) {A}
(B) {B}
(C) {C}
(D) {D}

Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'.
""".strip()


def format_multichoice_question(row):
    """
    Format a GPQA question with its answer choices.

    Args:
        row: Dictionary with keys: Question, A, B, C, D

    Returns:
        Formatted prompt string with question and all choices
    """
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


class GPQAEval(Eval):
    """
    Evaluation for the GPQA (Graduate-Level Google-Proof Q&A) benchmark.

    This evaluation tests models on graduate-level science questions that
    require deep domain knowledge. Answer choices are randomly permuted
    across repeated attempts to test consistency.

    Key Features:
    - Random answer order permutation for each repeat
    - Parallel processing for faster evaluation
    - Support for different dataset variants (diamond, extended, main)
    - Debug mode for focused testing on specific questions
    """

    def __init__(
        self,
        n_repeats: int = 8,
        variant: str = "diamond",
        num_examples: int | None = None,  # Restrict to subset for debugging
        debug: bool = False,
        n_threads: int = 1,
    ):
        """
        Initialize the GPQA evaluation.

        Args:
            n_repeats: Number of times to ask each question with different
                answer orderings (default: 8). This helps measure consistency.
            variant: Dataset variant to use:
                - "diamond": Main benchmark (198 questions, highest quality)
                - "extended": Larger set
                - "main": Balanced set
            num_examples: If set, randomly sample this many questions
                If None, use all questions in the variant
            debug: If True, uses only a specific test question for debugging
            n_threads: Number of parallel threads for processing
        """
        # Load the GPQA dataset from OpenAI's public blob storage
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
        rng = random.Random(0)  # Fixed seed for reproducibility

        # Debug mode: use only a specific test question
        if debug:
            examples = [row.to_dict() for _, row in df.iterrows() if "ESPRESSO spectrograph, please" in row["Question"]]
        else:
            examples = [row.to_dict() for _, row in df.iterrows()]
            # Optionally sample a subset for debugging
            if num_examples:
                assert n_repeats == 1, "n_repeats only supported for num_examples = None"
                examples = rng.sample(examples, num_examples)

        # Repeat each question n_repeats times
        examples = examples * n_repeats

        # Generate a random permutation for each repeat
        # This randomizes the order of answer choices (A/B/C/D)
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]

        self.examples = examples
        self.n_repeats = n_repeats
        self.n_threads = n_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """
        Run the GPQA evaluation on a given model.

        For each question:
        1. Permute the answer choices according to the stored permutation
        2. Determine which letter (A/B/C/D) corresponds to the correct answer
        3. Format the question with permuted choices
        4. Get model response
        5. Extract the chosen letter from the response
        6. Score 1.0 if correct, 0.0 otherwise

        Args:
            sampler: The model to evaluate

        Returns:
            EvalResult with accuracy, variance estimates, and full conversation logs
        """
        def fn(row: dict):
            """Evaluate a single GPQA question."""
            # Get the four answer choices (1 correct, 3 incorrect)
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]

            # Apply the random permutation to shuffle answer order
            choices = [choices[i] for i in row["permutation"]]

            # Find which position (0-3) the correct answer ended up in
            correct_index = choices.index(row["Correct Answer"])
            # Convert to letter A/B/C/D
            correct_answer = "ABCD"[correct_index]

            # Build the prompt with permuted choices
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )

            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]

            # Get model's response
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list

            # Extract the model's answer choice (A/B/C/D)
            extracted_answer = extract_abcd(response_text)

            # Score: 1.0 if model chose the correct letter, 0.0 otherwise
            score = 1.0 if extracted_answer == correct_answer else 0.0

            # Generate HTML visualization
            html = report.jinja_env.from_string(report.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )

            # Build full conversation
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]

            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        # Process all examples in parallel
        results = report.map_with_progress(fn, self.examples, num_threads=self.n_threads)

        # Aggregate results with mean, std, and bootstrap estimates
        return report.aggregate_results(results)


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1], "r") as f:
        results = json.load(f)

    passes = 0
    for convo, html in zip(results["convos"], results["htmls"]):
        message = convo[-1]["content"]
        import re

        # the ground truth is in <p>Correct Answer: A</p> in the html
        ground_truth = re.search(r"<p>Correct Answer: (A|B|C|D)</p>", html)
        ground_truth = ground_truth.group(1)
        extracted_answer = extract_abcd(message)
        if extracted_answer == ground_truth:
            passes += 1
        elif len(message) > 15:
            print("no match:", message)
            print("ground truth:", ground_truth)
            print("extracted answer:", extracted_answer)
            print("--------------------------------")

    pass_rate = passes / len(results["convos"])
    print(f"pass@1: {pass_rate}")