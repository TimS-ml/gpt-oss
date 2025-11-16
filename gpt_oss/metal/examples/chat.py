#!/usr/bin/env python
"""
Interactive Chat Interface for gpt-oss Metal Backend

This script provides an interactive chat interface using the gpt-oss model with
Apple Silicon Metal acceleration. It implements a structured chat format with:
- System/user/assistant roles
- Special tokens for message formatting (<|start|>, <|message|>, <|end|>)
- Multi-channel output (analysis and final channels)
- Streaming token generation with visual formatting

The chat format follows a structured protocol where messages are delimited by
special tokens and the assistant can output reasoning in separate channels.

Metal Optimizations:
- Uses Apple Silicon GPU acceleration via Metal Performance Shaders
- Efficient context management with configurable context window
- Streaming inference with low latency on unified memory architecture
"""

import argparse
import sys

from datetime import date
from gpt_oss.metal import Context, Model


# Default system prompt that configures the model's behavior
# Includes reasoning effort setting and channel instructions
DEFAULT_PROMPT = f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {date.today().isoformat()}

reasoning effort high

# Valid channels: analysis, final. Channel must be included for every message."""


# Command-line argument parser with helpful defaults
parser = argparse.ArgumentParser(description="Chat with gpt-oss", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model", metavar="PATH", type=str, help="Path to gpt-oss model in Metal inference format")
parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="System prompt")
parser.add_argument(
    "--context-length", type=int, default=0, help="The maximum context length (0 = use model default)"
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="Sampling temperature (higher = more random)"
)
parser.add_argument(
    "--seed", type=int, default=0, help="Sampling seed for reproducibility"
)


# ANSI color codes for terminal output formatting
GREY = "\33[90m"    # Used for analysis channel output (thinking/reasoning)
BOLD = "\33[1m"     # Used for role labels (User:, Assistant:)
RESET = "\33[0m"    # Reset formatting to default


def main(args):
    """
    Main chat loop that handles user interaction and model inference.

    This function:
    1. Loads the model from disk using Metal-optimized format
    2. Initializes a conversation context with the system prompt
    3. Enters an interactive loop where user messages are processed
    4. Streams model responses with proper formatting and channel handling

    Args:
        args: Command-line arguments (typically sys.argv[1:])

    Note:
        The Metal backend loads the entire model into unified memory for
        fast inference on Apple Silicon GPUs.
    """
    options = parser.parse_args(args)

    # Load model from Metal-optimized binary format
    # The model includes weights, tokenizer, and configuration
    model = Model(options.model)
    tokenizer = model.tokenizer

    # Encode special tokens used for structured message formatting
    # These tokens structure the conversation into roles and content
    start_token = tokenizer.encode_special_token("<|start|>")       # Marks start of a message block
    message_token = tokenizer.encode_special_token("<|message|>")   # Separates role from content
    end_token = tokenizer.encode_special_token("<|end|>")           # Marks end of a message block
    return_token = tokenizer.encode_special_token("<|return|>")     # Signals end of assistant turn
    channel_token = tokenizer.encode_special_token("<|channel|>")   # Separates channel name from content

    # Initialize conversation context with Metal-accelerated inference
    # Context manages the KV cache and token history on the GPU
    context = Context(model, context_length=options.context_length)

    # Add system prompt to context using structured message format:
    # <|start|>system<|message|>{prompt}<|end|>
    context.append(start_token)
    context.append("system")
    context.append(message_token)
    context.append(options.prompt)
    context.append(end_token)

    # Main interactive chat loop
    while True:
        # Add user message to context in structured format:
        # <|start|>user<|message|>{user_input}<|end|>
        context.append(start_token)
        context.append("user")
        context.append(message_token)
        message = input(f"{BOLD}User:{RESET} ").rstrip()
        context.append(message)
        context.append(end_token)

        # Print assistant label and prepare for streaming response
        print(f"{BOLD}Assistant:{RESET} {GREY}", end="", flush=True)

        # Begin assistant response with channel specification
        # Format: <|start|>assistant<|channel|>{channel_name}<|message|>{content}
        context.append(start_token)
        context.append("assistant")
        context.append(channel_token)

        # State machine for parsing structured assistant response
        inside_start_block = True      # True when parsing role name
        inside_channel_block = True     # True when parsing channel name
        role = "assistant"
        channel = ""

        # Stream tokens from the model using Metal-accelerated inference
        while True:
            # Sample next token from the model on Apple Silicon GPU
            # Uses configured temperature and seed for reproducible sampling
            token = context.sample(
                temperature=options.temperature,
                seed=options.seed,
            )
            # Add sampled token to context (updates KV cache on GPU)
            context.append(token)

            # Handle special tokens and state transitions
            if token == return_token:
                # <|return|> signals end of assistant turn
                print(flush=True)
                break
            elif token == start_token:
                # <|start|> begins a new message block (possibly multi-turn)
                inside_start_block = True
                role = ""
                channel = ""
            elif token == message_token:
                # <|message|> separates metadata from content
                inside_start_block = False
                inside_channel_block = False
                # Apply grey color for analysis channel (reasoning/thinking)
                if channel == "analysis":
                    print(f"{GREY}", end="", flush=True)
            elif token == end_token:
                # <|end|> marks end of a message block
                print(f"{RESET}", flush=True)
            elif token == channel_token:
                # <|channel|> separates role from channel name
                inside_channel_block = True
            elif token < tokenizer.num_text_tokens:
                # Regular text token - decode and display based on current state
                if inside_channel_block:
                    # Accumulate channel name (e.g., "analysis" or "final")
                    channel += str(tokenizer.decode(token), encoding="utf-8")
                elif inside_start_block:
                    # Accumulate role name (usually not displayed in output)
                    role += str(tokenizer.decode(token), encoding="utf-8")
                else:
                    # Output actual message content to terminal
                    # Use buffer.write for proper byte handling
                    sys.stdout.buffer.write(tokenizer.decode(token))
                    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
