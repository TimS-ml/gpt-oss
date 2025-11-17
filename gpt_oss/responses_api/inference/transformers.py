"""
HuggingFace Transformers inference backend for the Responses API.

This backend uses the HuggingFace Transformers library for inference.
It's the most widely compatible backend, supporting virtually any model
available on HuggingFace Hub.

Features:
- Wide model compatibility (any HuggingFace model)
- Easy to use and widely documented
- Automatic device mapping for multi-GPU setups
- BF16 for memory efficiency

Limitations:
- Not the most optimized for production serving
- Generates one token at a time (no batching)
- No advanced optimizations like prefix caching

This backend is best for:
- Experimenting with different models from HuggingFace
- Quick prototyping
- Development and testing
- Models not supported by other backends

Usage:
    # Single GPU
    python -m gpt_oss.responses_api.serve --inference-backend transformers --checkpoint /path/to/model

    # Multi-GPU (automatic with device_map="auto")
    python -m gpt_oss.responses_api.serve --inference-backend transformers --checkpoint /path/to/model

Note: This implementation is simple but not the most efficient. For production
      deployments, consider using vLLM or Triton backends instead.
"""

import os
from typing import Callable, List

# Transformers imports
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch


DEFAULT_TEMPERATURE = 0.0  # Greedy decoding by default
TP = os.environ.get("TP", 2)  # Tensor parallelism (currently unused)

def load_model(checkpoint: str):
    """
    Load a model using HuggingFace Transformers Auto API.

    Uses bfloat16 for memory efficiency and automatic device mapping
    for multi-GPU setups.

    Args:
        checkpoint: Path to model checkpoint or HuggingFace model ID

    Returns:
        Loaded PreTrainedModel instance
    """

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,  # Use BF16 for efficiency
        device_map="auto",           # Automatically distribute across GPUs
    )

    return model


def get_infer_next_token(model: PreTrainedModel):
    """
    Create an inference function that generates one token at a time.

    Uses model.generate() with max_new_tokens=1 to match the streaming
    interface expected by the Responses API.

    Args:
        model: Loaded PreTrainedModel

    Returns:
        Function with signature: (tokens, temperature, new_request) -> int
    """

    def infer_next_token(
        tokens: List[int],
        temperature: float = DEFAULT_TEMPERATURE,
        new_request: bool = False,  # Kept for interface compatibility; unused
    ) -> int:
        """
        Generate the next token given input tokens.

        Args:
            tokens: List of token IDs representing the input sequence
            temperature: Sampling temperature (0 = greedy)
            new_request: Whether this is a new request (unused)

        Returns:
            Next token ID
        """
        # Convert to tensor and move to model's device
        tokens = torch.tensor([tokens], dtype=torch.int64, device=model.device)

        # Generate one token
        # do_sample=True enables sampling, False uses greedy decoding
        output = model.generate(
            tokens,
            max_new_tokens=1,
            do_sample=temperature != 0,
            temperature=temperature if temperature != 0 else None,
        )

        # Extract and return the newly generated token
        return output[0, -1].tolist()

    return infer_next_token


def setup_model(checkpoint: str) -> Callable[[List[int], float, bool], int]:
    """
    Initialize the Transformers backend.

    Args:
        checkpoint: Path to model checkpoint or HuggingFace model ID

    Returns:
        Inference function ready for use by the API server
    """
    model = load_model(checkpoint)
    infer_next_token = get_infer_next_token(model)
    return infer_next_token
