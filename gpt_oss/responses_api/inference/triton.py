"""
Triton inference backend for the Responses API.

This backend uses custom Triton kernels for high-performance inference
on NVIDIA GPUs. Features:
- CUDA graph optimization for minimal overhead
- KV cache management for efficient multi-turn conversations
- Longest Common Prefix (LCP) optimization to reuse computation
- Support for multi-GPU deployment via PyTorch distributed

Performance characteristics:
- Very fast inference (optimized CUDA kernels)
- Low latency for single-token generation
- Efficient memory usage with KV caching
- Best suited for deployment on NVIDIA GPUs

Usage:
    # Single GPU
    python -m gpt_oss.responses_api.serve --inference-backend triton

    # Multi-GPU with torchrun
    torchrun --nproc-per-node=4 -m gpt_oss.responses_api.serve --inference-backend triton
"""

import datetime
import os
from typing import Callable

# Enable expandable CUDA memory segments for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.distributed as dist

from gpt_oss.triton.model import Cache, ModelConfig, Transformer

DEFAULT_TEMPERATURE = 0.0  # Greedy decoding by default
CONTEXT = 16_384  # Maximum context length (16K tokens)
CONCURRENT_SESSIONS = 1  # Number of concurrent inference sessions per GPU

# GPU rank for multi-GPU setups (set via RANK environment variable)
rank = int(
    os.environ.get("RANK", 0)
)


def load_model(checkpoint: str):
    """
    Load the model from checkpoint onto the GPU.

    Args:
        checkpoint: Path to the SafeTensors checkpoint directory

    Returns:
        Tuple of (model, device)
    """
    print(f"[{rank}] loading model...")

    # Set up CUDA device for this process
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)  # Disable gradients for inference
    device = torch.device(f"cuda:{rank}")

    # Load model weights from checkpoint
    model = Transformer.from_checkpoint(checkpoint, device=device)

    print(f"[{rank}] loaded")
    return model, device


def get_infer_next_token(model, device):
    """
    Create an optimized inference function using CUDA graphs.

    This function sets up:
    1. KV caches for all transformer layers
    2. A CUDA graph for single-token generation (eliminates kernel launch overhead)
    3. LCP-based cache reuse to avoid redundant computation

    Returns:
        Inference function with signature (tokens, temperature, new_request) -> int
    """
    # Create KV caches for all transformer blocks
    # Each cache stores keys and values for efficient attention computation
    caches = [
        Cache(CONCURRENT_SESSIONS, CONTEXT, model.config.num_key_value_heads)
        for _ in range(len(model.block))
    ]

    # Pre-allocated tensor for single-token input (used in CUDA graph)
    input_token = torch.zeros(1, dtype=torch.int32, device=device)

    # Track what tokens have been processed so far (for LCP optimization)
    tokens_so_far = []

    # Warm up the model and create CUDA graph for single-token generation
    # CUDA graphs capture the entire execution graph, eliminating kernel launch overhead
    model.prefill(torch.zeros(1, 4, dtype=torch.int32, device=device), caches)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(input_token[None, :], caches=caches)[0]

    def lcp(cache: list[int], inp: list[int]) -> list[int]:
        """
        Find the longest common prefix between two token sequences.

        This allows us to reuse cached computation when the input
        matches the beginning of a previous computation.

        Args:
            cache: Previously processed tokens
            inp: New input tokens

        Returns:
            Common prefix of both sequences
        """
        i = 0
        max_len = min(len(cache), len(inp))
        while i < max_len and cache[i] == inp[i]:
            i += 1
        return cache[:i]

    def sample_next_token(
        logits: torch.Tensor, temperature: float = DEFAULT_TEMPERATURE
    ) -> int:
        """
        Sample the next token from logits.

        Args:
            logits: Model output logits
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Selected token ID
        """
        if temperature == 0.0:
            # Greedy decoding: select most probable token
            return torch.argmax(logits[-1, :], dim=-1).item()
        # Sampling: apply temperature and sample from distribution
        probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
        return torch.multinomial(probs[-1, :], num_samples=1).item()

    @torch.inference_mode()
    def infer_next_token(
        tokens: list[int],
        temperature: float = DEFAULT_TEMPERATURE,
        new_request: bool = False,
    ) -> int:
        """
        Generate the next token given input tokens.

        Uses LCP optimization to avoid re-processing shared prefixes,
        prefills multiple tokens at once if needed, then uses the CUDA
        graph for fast single-token generation.

        Args:
            tokens: Full token sequence including input
            temperature: Sampling temperature
            new_request: Whether this is a new request (unused)

        Returns:
            Next token ID
        """
        nonlocal tokens_so_far

        # Find common prefix with previously processed tokens
        tokens_so_far = lcp(tokens_so_far, tokens)

        # Truncate caches to match the common prefix
        for cache in caches:
            cache.truncate(len(tokens_so_far))

        # Extract only the new tokens that need processing
        all_tokens = tokens  # Keep for debugging
        tokens = tokens[len(tokens_so_far) :]

        # If we have multiple new tokens, prefill them efficiently
        if len(tokens) > 1:
            model.prefill(
                torch.as_tensor(tokens[:-1], dtype=torch.int32, device=device)[None, :],
                caches,
            )

        if len(tokens) == 0:
            # This shouldn't happen - would mean we're at the same position
            breakpoint()

        # Process the last token using the CUDA graph (fast path)
        input_token[-1] = tokens[-1]
        graph.replay()

        # Sample next token from the output logits
        next_tok = sample_next_token(logits, temperature=temperature)

        return next_tok

    return infer_next_token


def setup_model(checkpoint: str) -> Callable[[list[int], float], int]:
    """
    Initialize the Triton backend.

    Args:
        checkpoint: Path to model checkpoint directory

    Returns:
        Inference function ready for use by the API server
    """
    model, device = load_model(checkpoint)
    infer_next_token = get_infer_next_token(model, device)
    return infer_next_token
