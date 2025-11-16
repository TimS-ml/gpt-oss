"""
Tokenizer module for GPT-OSS models.

This module provides the tokenizer configuration for gpt-oss models, which use
a custom extension of the o200k_base tokenizer with additional special tokens
required for the Harmony response format.

The Harmony format is OpenAI's structured response format that supports:
- Multi-channel outputs (final answer, chain-of-thought, tool calls)
- Tool use and function calling
- Structured message formatting
"""

import tiktoken


def get_tokenizer():
    """
    Create and return a tokenizer for gpt-oss models.

    The tokenizer extends the o200k_base encoding (used by GPT-4o and other models)
    with additional special tokens required for the Harmony response format.

    Special Tokens:
        <|startoftext|> (199998): Marks the beginning of text
        <|endoftext|> (199999): Marks the end of text
        <|return|> (200002): Indicates a return/yield point
        <|constrain|> (200003): Marks constraints or restrictions
        <|channel|> (200005): Denotes a communication channel
        <|start|> (200006): Generic start marker
        <|end|> (200007): Generic end marker
        <|message|> (200008): Message boundary marker
        <|call|> (200012): Function/tool call marker
        <|reserved_*|>: Reserved tokens for future use (200000-200001, 200004,
                       200009-200011, 200013-201087)

    Returns:
        tiktoken.Encoding: A configured tokenizer instance with the o200k_harmony
                          encoding that includes all base vocabulary plus Harmony
                          special tokens.

    Example:
        >>> tokenizer = get_tokenizer()
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(tokens)
    """
    # Load the base o200k encoding (used by GPT-4o and similar models)
    o200k_base = tiktoken.get_encoding("o200k_base")

    # Create a custom encoding by extending o200k_base with Harmony special tokens
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,  # Use the same pattern string for tokenization
        mergeable_ranks=o200k_base._mergeable_ranks,  # Use the same vocabulary merges
        special_tokens={
            # Include all base special tokens
            **o200k_base._special_tokens,

            # Add Harmony-specific special tokens
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {
            # Add a range of reserved tokens for future extensions (200013-201087)
            f"<|reserved_{i}|>": i for i in range(200013, 201088)
        },
    )
    return tokenizer
