"""
Basic Evaluation - Sanity Check

This is the simplest possible evaluation, designed to verify that:
1. The evaluation framework is working correctly
2. The model can generate non-empty responses
3. The sampler can successfully communicate with the model

Not a real benchmark - just a smoke test!

The evaluation:
- Asks a simple question: "hi"
- Expects any non-empty response
- Scores 1.0 if response length > 0, otherwise 0.0

This is useful for:
- Testing new model deployments
- Debugging sampler configurations
- Verifying the evaluation pipeline works end-to-end
"""
from . import report

from .types import Eval, EvalResult, SamplerBase, SingleEvalResult


class BasicEval(Eval):
    """
    Minimal evaluation that checks if the model can respond to a simple greeting.

    This is not a real benchmark - it's a sanity check to ensure the evaluation
    framework and model are working correctly.
    """

    def __init__(self):
        """Initialize with a single trivial example."""
        self.examples = [{
            "question": "hi",
            "answer": "hi, how can i help?",  # Expected answer (not actually used for grading)
        }]

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """
        Run the basic evaluation.

        For each example:
        1. Send the question to the model
        2. Check if the response is non-empty
        3. Score 1.0 for non-empty, 0.0 for empty
        4. Track response length as a metric

        Args:
            sampler: The model to evaluate

        Returns:
            EvalResult with score and metrics
        """
        def fn(row: dict):
            """Evaluate a single example."""
            # Send the question to the model
            sampler_response = sampler([
                sampler._pack_message(content=row["question"], role="user")
            ])
            response_text = sampler_response.response_text
            extracted_answer = response_text  # In BasicEval, we don't parse the answer
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list

            # Simple scoring: 1.0 if non-empty, 0.0 if empty
            score = 1.0 if len(extracted_answer) > 0 else 0.0

            # Generate HTML for this example
            html = report.jinja_env.from_string(report.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )

            # Build the full conversation history
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]

            # Return result with character count as an additional metric
            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        # Process all examples (just 1 in this case) sequentially
        results = report.map_with_progress(fn, self.examples, num_threads=1)

        # Aggregate into final result
        return report.aggregate_results(results)

