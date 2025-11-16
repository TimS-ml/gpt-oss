"""
vLLM inference backend for the Responses API.

This backend uses vLLM (https://github.com/vllm-project/vllm), a fast and
easy-to-use library for LLM inference. Features:
- PagedAttention for efficient memory management
- Prefix caching to reuse computation for shared prefixes
- Tensor parallelism for multi-GPU inference
- Continuous batching (though not fully utilized here)

NOTE: This implementation generates one token at a time to match the
streaming interface expected by the Responses API. For batch processing
or multi-request serving, vLLM's native API would be more efficient.

Performance characteristics:
- Very good out-of-the-box performance
- Wide model compatibility (supports most HuggingFace models)
- Efficient prefix caching reduces redundant computation
- Best for: Multi-GPU deployment, experimenting with different models

Usage:
    # Single GPU
    python -m gpt_oss.responses_api.serve --inference-backend vllm --checkpoint /path/to/model

    # Multi-GPU (set TP environment variable)
    TP=4 python -m gpt_oss.responses_api.serve --inference-backend vllm --checkpoint /path/to/model
"""

import os
from typing import Callable, List, Optional

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

DEFAULT_TEMPERATURE = 0.0  # Greedy decoding by default
TP = os.environ.get("TP", 2)  # Tensor parallelism degree (number of GPUs)

def load_model(checkpoint: str):
    """
    Create and initialize the vLLM engine.

    Enables prefix caching so that repeated prefixes across calls
    can reuse KV cache for faster prefill (useful for multi-turn conversations).

    Args:
        checkpoint: Path to the model checkpoint/weights

    Returns:
        vLLM LLM engine instance
    """

    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TP,        # Number of GPUs for tensor parallelism
        enable_prefix_caching=True,     # Reuse KV cache for shared prefixes
        disable_log_stats=True,         # Reduce log verbosity
    )

    return llm


def get_infer_next_token(llm: LLM):
    """
    Create an inference function that generates one token at a time.

    Uses vLLM's TokensPrompt to pass token IDs directly (avoiding re-tokenization).
    Leverages prefix caching to reuse computation for shared prefixes across calls.

    Args:
        llm: Initialized vLLM engine

    Returns:
        Function with signature: (tokens, temperature, new_request) -> int
        that generates the next token given input tokens
    """

    def infer_next_token(
        tokens: List[int],
        temperature: float = DEFAULT_TEMPERATURE,
        new_request: bool = False,  # Kept for interface compatibility; unused by vLLM
    ) -> int:
        """
        Generate the next token given input tokens.

        Args:
            tokens: List of token IDs representing the input sequence
            temperature: Sampling temperature (0 = greedy)
            new_request: Whether this is a new request (unused)

        Returns:
            Next token ID

        Raises:
            ValueError: If tokens list is empty
            RuntimeError: If vLLM fails to generate a token
        """
        if not tokens:
            raise ValueError("tokens must contain at least one input token id")

        # Configure sampling parameters
        sampling = SamplingParams(
            temperature=float(temperature),
            max_tokens=1,            # Generate only the next token
            n=1,                     # Single output sequence
            # Additional controls like top_p, top_k can be added here
        )

        # Generate using token IDs directly (no text encoding)
        # Prefix caching will automatically reuse KV cache for shared prefixes
        outputs = llm.generate(
            TokensPrompt(prompt_token_ids=tokens),
            sampling_params=sampling,
        )

        # Extract the generated token
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty outputs")

        gen = outputs[0].outputs[0]
        if not gen.token_ids:
            # Model stopped (e.g., generated EOS token)
            raise RuntimeError("No next token was generated (possibly EOS).")

        next_tok = int(gen.token_ids[0])
        return next_tok

    return infer_next_token


def setup_model(checkpoint: str) -> Callable[[List[int], float, bool], int]:
    """
    Initialize the vLLM backend.

    Args:
        checkpoint: Path to model checkpoint/weights directory

    Returns:
        Inference function ready for use by the API server
    """
    llm = load_model(checkpoint)
    infer_next_token = get_infer_next_token(llm)
    return infer_next_token
