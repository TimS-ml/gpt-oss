"""
Metal inference backend for the Responses API.

This backend uses Apple's Metal Performance Shaders (MPS) for inference
on macOS devices with Apple Silicon or AMD GPUs. Features:
- Metal-accelerated kernels for efficient GPU compute on macOS
- Automatic KV cache management with LCP optimization
- Batch token generation (generates multiple tokens at once, returns one by one)
- Native macOS integration

Performance characteristics:
- Good performance on Apple Silicon (M1/M2/M3)
- Lower memory overhead than some alternatives
- Efficient for local development on Mac
- Best suited for: macOS development, Apple Silicon deployment

Usage:
    python -m gpt_oss.responses_api.serve --inference-backend metal --checkpoint /path/to/model

Note: This backend is only available on macOS systems with Metal support.
"""

from typing import Callable

from gpt_oss.metal import Context, Model


# Configuration parameters
MAX_OUTPUT_TOKENS = 100  # Number of tokens to generate in each batch


def setup_model(checkpoint: str) -> Callable[[list[int], float], int]:
    """
    Initialize the Metal backend.

    Loads the model using Metal Performance Shaders and creates
    an inference context for token generation.

    Args:
        checkpoint: Path to the model checkpoint directory

    Returns:
        Inference function ready for use by the API server
    """

    # Load model weights
    model = Model(checkpoint)

    # Create inference context (manages KV cache and generation state)
    context = Context(model)

    seed = 0  # Random seed for sampling (currently fixed)
    output_tokens = []  # Buffer for batched token generation

    def infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        """
        Generate the next token using Metal-accelerated inference.

        This implementation generates tokens in batches (for efficiency) but
        returns them one at a time. The Context handles LCP caching internally,
        reusing computation when the input prefix matches cached tokens.

        Args:
            tokens: Full token sequence including input
            temperature: Sampling temperature (0 = greedy)
            new_request: If True, clears the output buffer and starts fresh

        Returns:
            Next token ID
        """
        nonlocal output_tokens

        # Clear output buffer on new requests
        if new_request:
            output_tokens = []

        # Generate a batch of tokens if buffer is empty
        if len(output_tokens) == 0:
            # Reset context and feed input tokens
            # The Context automatically handles LCP caching - if the input tokens
            # match what's already in the KV cache, computation is reused
            context.reset()
            for t in tokens:
                context.append(t)

            # Generate multiple tokens at once (more efficient than one-by-one)
            output_tokens = context.sample(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=temperature,
                seed=seed
            )

        # Return one token from the buffer
        return int(output_tokens.pop(0))

    return infer_next_token
