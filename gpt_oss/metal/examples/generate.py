#!/usr/bin/env python
"""
Simple Text Generation Script for gpt-oss Metal Backend

This script demonstrates basic text generation using the gpt-oss model with
Apple Silicon Metal acceleration. It provides a straightforward interface for:
- Loading a model from Metal-optimized binary format
- Tokenizing an input prompt
- Generating a fixed number of tokens with streaming output

Unlike the chat.py example, this script uses raw text generation without
structured message formatting or special tokens, making it suitable for
simple completion tasks.

Metal Optimizations:
- GPU-accelerated inference on Apple Silicon (M1/M2/M3)
- Efficient memory management with unified memory architecture
- Low-latency token generation with Metal Performance Shaders
"""

import argparse
import sys

from gpt_oss.metal import Context, Model


# Command-line argument parser for generation parameters
parser = argparse.ArgumentParser(description='Chat with gpt-oss', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', metavar='PATH', type=str, help='Path to gpt-oss model in Metal inference format')
parser.add_argument('-p', '--prompt', type=str, required=True, help='Text prompt for generation')
parser.add_argument('-l', '--limit', type=int, default=100, help='Number of tokens to generate')
parser.add_argument('--context-length', type=int, default=0, help='Maximum context length (0 = use model default)')


def main(args):
    """
    Main generation function that produces text completions.

    This function:
    1. Loads the model from Metal-optimized binary format
    2. Tokenizes the input prompt and adds it to the context
    3. Generates tokens one at a time using GPU acceleration
    4. Streams decoded tokens to stdout until limit is reached

    Args:
        args: Command-line arguments (typically sys.argv[1:])

    Note:
        The Metal backend maintains the KV cache on the GPU for efficient
        autoregressive generation with low per-token latency.
    """
    options = parser.parse_args(args)

    # Load model from Metal-optimized binary format
    # The model is loaded entirely into unified memory for fast access
    model = Model(options.model)

    # Create a context for managing tokens and KV cache on Apple Silicon GPU
    # The context handles tokenization, caching, and inference state
    context = Context(model, context_length=options.context_length)

    # Tokenize and add the prompt to the context
    # This performs a forward pass to populate the KV cache for the prompt
    context.append(options.prompt)

    # Display the tokenized prompt (useful for debugging)
    print(context.tokens)

    # Remember how many tokens are in the prompt (not generated)
    prompt_tokens = context.num_tokens

    # Get tokenizer for decoding generated tokens to text
    tokenizer = model.tokenizer

    # Generate tokens up to the specified limit using Metal acceleration
    while context.num_tokens - prompt_tokens < options.limit:
        # Sample next token from the model using default temperature (1.0)
        # This runs Metal compute shaders for attention and sampling on the GPU
        token = context.sample()

        # Add the sampled token to context (updates KV cache on GPU)
        context.append(token)

        # Decode and print the token as UTF-8 text
        # Use streaming output with flush for real-time display
        print(str(tokenizer.decode(token), encoding="utf-8"), end='', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
