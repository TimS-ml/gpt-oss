"""
Simple text generation script for gpt-oss models.

This script demonstrates basic token generation using various inference backends.
It's designed for testing and demonstration purposes only - for production use,
see gpt_oss.chat which includes full Harmony parser support and tool integration.

Usage:
    Single GPU (Triton):
        python -m gpt_oss.generate -p "Why did the chicken cross the road?" model/

    Multi-GPU with tensor parallelism (PyTorch):
        torchrun --nproc-per-node=4 -m gpt_oss.generate -p "Tell me a story" model/

    vLLM backend:
        python -m gpt_oss.generate -b vllm --tensor-parallel-size=2 -p "Hello" model/

Supported Backends:
    - torch: PyTorch implementation with tensor parallelism
    - triton: Optimized single-GPU inference with Triton kernels
    - vllm: High-throughput inference using the vLLM engine

Note: This script generates raw tokens without Harmony parsing, so the output
      may include special tokens and formatting markers.
"""

import argparse

from gpt_oss.tokenizer import get_tokenizer


def main(args):
    """
    Main generation function that initializes the appropriate backend and runs inference.

    This function:
    1. Initializes the selected inference backend (torch, triton, or vllm)
    2. Loads the model from the checkpoint directory
    3. Tokenizes the input prompt
    4. Generates tokens one at a time
    5. Prints each generated token with its log probability

    Args:
        args: Command-line arguments containing:
            - checkpoint: Path to model weights directory
            - backend: Inference backend to use (torch/triton/vllm)
            - prompt: Input text to generate from
            - temperature: Sampling temperature (0.0 for greedy)
            - limit: Maximum number of tokens to generate (0 for unlimited)
            - context_length: Maximum context length for Triton backend
            - tensor_parallel_size: Number of GPUs for vLLM backend
    """
    # Initialize the appropriate inference backend based on user selection
    match args.backend:
        case "torch":
            # PyTorch backend: supports tensor parallelism across multiple GPUs
            # Use with torchrun for multi-GPU: torchrun --nproc-per-node=N script.py
            from gpt_oss.torch.utils import init_distributed
            from gpt_oss.torch.model import TokenGenerator as TorchGenerator
            device = init_distributed()  # Initialize distributed training and get device
            generator = TorchGenerator(args.checkpoint, device=device)

        case "triton":
            # Triton backend: optimized single-GPU inference with custom CUDA kernels
            # Provides better performance than PyTorch for single-GPU scenarios
            from gpt_oss.torch.utils import init_distributed
            from gpt_oss.triton.model import TokenGenerator as TritonGenerator
            device = init_distributed()
            generator = TritonGenerator(args.checkpoint, context=args.context_length, device=device)

        case "vllm":
            # vLLM backend: high-throughput inference with PagedAttention
            # Excellent for serving and batch processing
            from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator
            generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=args.tensor_parallel_size)

        case _:
            raise ValueError(f"Invalid backend: {args.backend}")

    # Initialize tokenizer and encode the input prompt
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(args.prompt)

    # Set up token generation limits (None means no limit)
    max_tokens = None if args.limit == 0 else args.limit

    # Generate tokens one at a time until hitting stop tokens or max_tokens limit
    # The generator yields (token_id, log_probability) tuples
    for token, logprob in generator.generate(
        tokens,
        stop_tokens=[tokenizer.eot_token],  # Stop at end-of-text token
        temperature=args.temperature,  # Controls randomness (0.0 = greedy/deterministic)
        max_tokens=max_tokens,
        return_logprobs=True  # Include log probabilities in output
    ):
        tokens.append(token)  # Add generated token to context for next iteration
        token_text = tokenizer.decode([token])  # Decode token to text
        print(
            f"Generated token: {repr(token_text)}, logprob: {logprob}"
        )


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Text generation example")

    # Required argument: path to model checkpoint directory
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint directory containing model weights",
    )

    # Optional: input prompt text
    parser.add_argument(
        "-p",
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="How are you?",
        help="Input text prompt for the language model to complete",
    )

    # Optional: sampling temperature (0.0 = deterministic, higher = more random)
    parser.add_argument(
        "-t",
        "--temperature",
        metavar="TEMP",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy/deterministic, 1.0 for standard sampling)",
    )

    # Optional: maximum number of tokens to generate
    parser.add_argument(
        "-l",
        "--limit",
        metavar="LIMIT",
        type=int,
        default=0,
        help="Maximum number of tokens to generate (0 for unlimited)",
    )

    # Optional: choose inference backend
    parser.add_argument(
        "-b",
        "--backend",
        metavar="BACKEND",
        type=str,
        default="torch",
        choices=["triton", "torch", "vllm"],
        help="Inference backend: 'torch' for multi-GPU, 'triton' for optimized single-GPU, 'vllm' for serving",
    )

    # Optional: tensor parallelism size for vLLM
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs to use for tensor parallelism with vLLM backend",
    )

    # Optional: maximum context length for Triton
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Maximum context length (sequence length) for Triton backend",
    )

    # Parse arguments and run main function
    args = parser.parse_args()
    main(args)
