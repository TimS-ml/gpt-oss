"""
AIME 2025 Evaluation

Benchmark: American Invitational Mathematics Examination (AIME) 2025
Dataset: https://huggingface.co/datasets/opencompass/AIME2025

About AIME:
- The AIME is a prestigious high school mathematics competition in the US
- It's the second stage in qualifying for the International Math Olympiad (IMO)
- Questions are significantly harder than typical standardized test questions
- All answers are integers from 0 to 999 (inclusive)

Evaluation Details:
- Dataset: 30 problems total (15 from AIME 2025-I, 15 from AIME 2025-II)
- Scoring: Exact match on integer answers (1.0 correct, 0.0 incorrect)
- Answer Format: Models are instructed to put final answer in \\boxed{}
- Multiple Attempts: By default, each problem is attempted 8 times (n_repeats=8)
  to estimate variance in model performance

This evaluation tests:
- Advanced mathematical reasoning
- Multi-step problem solving
- Ability to provide exact numerical answers
- Consistency across multiple attempts

Typical Topics:
- Algebra, geometry, number theory, combinatorics, probability
- Requires creativity and insight, not just computation
"""
import random
import re
import pandas
from . import report

from .types import Eval, EvalResult, SamplerBase, SingleEvalResult


# Prompt template for AIME questions
# Instructs the model to show reasoning and format the final answer with \boxed{}
AIME_TEMPLATE = """
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
"""


def format_aime_question(row):
    """
    Format an AIME question with instructions for the model.

    Args:
        row: Dictionary with "question" key

    Returns:
        Formatted prompt string with question and answer format instructions
    """
    return AIME_TEMPLATE.format(question=row["question"])


def extract_boxed_text(text):
    """
    Extract the answer from LaTeX \\boxed{} notation.

    AIME problems typically ask for answers in \\boxed{}, which models often use.
    This function extracts content from:
    - \\boxed{answer}
    - \\framebox{answer}

    If no boxed content is found, falls back to extracting the last integer in
    the response (since AIME answers are always integers).

    Args:
        text: Model response text

    Returns:
        Extracted answer as a string, or empty string if nothing found

    Examples:
        >>> extract_boxed_text("The answer is \\\\boxed{42}")
        '42'
        >>> extract_boxed_text("\\\\boxed{123, 456}")  # Takes last element if comma-separated
        '456'
        >>> extract_boxed_text("I calculated 789")  # Fallback to last integer
        '789'
    """
    # Look for \boxed{} or \framebox{} notation
    pattern = r'boxed{(.*?)}|framebox{(.*?)}'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Process matches in reverse order (prefer last occurrence)
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    # If the boxed content has commas, take the last element
                    # (handles cases like \boxed{x=5, y=10} where we want 10)
                    return group.split(',')[-1].strip()

    # Fallback: extract the last integer from the entire text
    # This handles cases where the model didn't use \boxed{}
    pattern = r'\d+'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]

    return ""


def normalize_number(s):
    """
    Extract the integer portion from the start of a string.

    AIME answers are always integers 0-999, so we normalize by extracting
    just the digits from the beginning.

    Args:
        s: String that may contain an integer

    Returns:
        String of digits, or None if no digits found at the start

    Examples:
        >>> normalize_number("42")
        '42'
        >>> normalize_number("123abc")
        '123'
        >>> normalize_number("abc")
        None
    """
    match = re.match(r"\d+", s)  # Match digits from the start
    if not match:
        return None
    return match.group(0)


class AIME25Eval(Eval):
    """
    Evaluation for the AIME 2025 mathematics competition.

    This evaluation loads problems from both AIME 2025-I and 2025-II exams
    and tests the model's ability to solve advanced mathematics problems.

    Key Features:
    - Supports multiple attempts per problem (n_repeats) for variance estimation
    - Parallel processing for faster evaluation
    - Exact integer matching for scoring
    - Includes random permutation tracking (for potential future use)
    """

    def __init__(
        self,
        n_repeats: int = 4,
        num_examples: int | None = None,  # Restrict to subset for debugging
        n_threads: int = 1,
    ):
        """
        Initialize the AIME 2025 evaluation.

        Args:
            n_repeats: Number of times to attempt each problem (default: 4)
                Use 8 for production runs to get good variance estimates
                Use 1 for debugging or quick tests
            num_examples: If set, randomly sample this many problems
                If None, use all 30 problems
            n_threads: Number of parallel threads for processing
        """
        # Load both AIME 2025 exams from HuggingFace
        path1 = f"https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-I.jsonl"
        df1 = pandas.read_json(path1, lines=True)
        path2 = f"https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-II.jsonl"
        df2 = pandas.read_json(path2, lines=True)

        # Combine problems from both exams
        examples = [row.to_dict() for _, row in df1.iterrows()] + [row.to_dict() for _, row in df2.iterrows()]

        # Normalize answers to ensure they're all integers
        examples = [{
            "question": row["question"],
            "answer": normalize_number(row["answer"]) if isinstance(row["answer"], str) else row["answer"],
        } for row in examples]

        # Use fixed random seed for reproducibility
        rng = random.Random(0)

        # Optionally sample a subset of examples for debugging
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)

        # Repeat each problem n_repeats times to estimate variance
        examples = examples * n_repeats

        # Add random permutation to each example (currently unused, but available for future use)
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]

        self.examples = examples
        self.n_repeats = n_repeats
        self.n_threads = n_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """
        Run the AIME evaluation on a given model.

        For each problem:
        1. Format the question with instructions
        2. Get model response
        3. Extract integer answer from \boxed{} or from text
        4. Compare to ground truth
        5. Score 1.0 for exact match, 0.0 otherwise

        Args:
            sampler: The model to evaluate

        Returns:
            EvalResult with:
            - score: Mean accuracy across all attempts
            - metrics: Including std, bootstrap_std for variance estimation
            - htmls: Visualizations of each attempt
            - convos: Full conversation histories
        """
        def fn(row: dict):
            """Evaluate a single AIME problem."""
            # Format the problem with instructions
            prompt_messages = [
                sampler._pack_message(
                    content=format_aime_question(row), role="user"
                )
            ]

            # Get model's response
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list

            # Extract the integer answer from the response
            extracted_answer = extract_boxed_text(response_text)
            correct_answer = int(row["answer"])

            # Convert extracted answer to integer for comparison
            # All AIME answers are integers 0-999
            try:
                extracted_answer = int(extracted_answer)
            except (ValueError, TypeError):
                # If extraction/conversion fails, mark as None (will score 0.0)
                extracted_answer = None

            # Score: 1.0 for exact match, 0.0 otherwise
            score = 1.0 if extracted_answer == correct_answer else 0.0

            # Generate HTML visualization for this attempt
            html = report.jinja_env.from_string(report.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )

            # Build full conversation history
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]

            # Return result with character count metric
            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        # Process all examples in parallel
        results = report.map_with_progress(fn, self.examples, num_threads=self.n_threads)

        # Aggregate results across all attempts
        # Will compute mean, std, bootstrap_std automatically
        return report.aggregate_results(results)

