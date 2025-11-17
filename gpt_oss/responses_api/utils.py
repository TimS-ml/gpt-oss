"""
Utility functions for the Responses API.

This module contains helper functions and test data used by the API
implementation. Currently includes fake token sequences for testing
without a real model.
"""

import time

# Predefined token sequences for testing/stubbing
# These represent encoded model outputs in the Harmony tokenization format
# The tokens include special control tokens (200xxx series) that mark:
# - Message boundaries
# - Reasoning/analysis sections
# - Tool calls
# - Different channels (final, analysis, commentary)

# First sequence (commented out) - represents a simple tool call response
fake_tokens = [
    200005,  # Start message delimiter
    35644,   # Message header token
    200008,  # Content start
    23483,
    316,
    1199,
    1114,
    717,
    170154,
    13,
    200007,  # Content end
    200006,  # Message end
    173781,  # Channel/recipient marker
    200005,  # Next message start
    35644,
    316,
    28,
    44580,
    775,
    170154,
    464,
    91,
    542,
    141043,
    91,
    29,
    4108,
    200008,
    10848,
    7693,
    7534,
    28499,
    18826,
    18583,
    200012,  # End of sequence
]

# Active token sequence - represents a simple Q&A interaction
# Decodes to something like:
# <|start|><channel:final><|end|>User: The answer is 1 + 1 = 2. Period.<|end|><channel:final><|end|><final><|end|>1 + 1 = 2. √²<eos>
fake_tokens = [
    200005,  # Start delimiter
    35644,   # Message header
    200008,  # Content start
    1844,    # "User"
    31064,   # ":"
    25,      # " "
    392,     # "The"
    4827,    # " answer"
    382,     # " is"
    220,     # " "
    17,      # "1"
    659,     # " +"
    220,     # " "
    17,      # "1"
    16842,   # " ="
    12295,   # " 2"
    81645,   # ". Period"
    13,      # "."
    51441,   # (special token)
    6052,    # (special token)
    13,      # "."
    200007,  # Content end
    200006,  # Message end
    173781,  # Channel marker
    200005,  # Next message start
    17196,   # "final"
    200008,  # Content start
    17,      # "1"
    659,     # " +"
    220,     # " "
    17,      # "1"
    314,     # " ="
    220,     # " "
    19,      # "2"
    13,      # "."
    9552,    # " √"
    238,     # superscript marker
    242,     # "²"
    200002,  # End of sequence (EOS)
]
# fake_tokens = [200005, 35644, 200008, 976, 1825, 31064, 25, 392, 25216, 29400, 290, 11122, 306, 52768, 2117, 16842, 1416, 1309, 316, 2281, 198, 68, 290, 2208, 11122, 13, 1416, 679, 261, 1114, 717, 170154, 484, 44390, 261, 5100, 1621, 26, 581, 1757, 2005, 198, 75, 480, 483, 5100, 392, 137956, 2117, 11, 13180, 4050, 7801, 4733, 290, 11122, 5377, 484, 290, 1114, 7377, 13, 1416, 1309, 260, 198, 78, 1199, 290, 1114, 4584, 364, 58369, 2421, 717, 170154, 483, 5100, 392, 137956, 2117, 11, 13180, 4050, 200007, 200006, 173781, 200005, 12606, 815, 260, 198, 78, 28, 117673, 3490]
# fake_tokens = [
#     198,
#     200005,
#     35644,
#     200008,
#     23483,
#     316,
#     1199,
#     1114,
#     717,
#     170154,
#     13,
#     200007,
#     200006,
#     173781,
#     200005,
#     12606,
#     815,
#     316,
#     32455,
#     106847,
#     316,
#     28,
#     44580,
#     775,
#     170154,
#     464,
#     91,
#     542,
#     141043,
#     91,
#     29,
#     4108,
#     200008,
#     10848,
#     7693,
#     7534,
#     28499,
#     18826,
#     18583,
#     200012,
#     198,
# ]

# Global queue of tokens to return - cycles through fake_tokens repeatedly
token_queue = fake_tokens.copy()


def stub_infer_next_token(tokens: list[int], temperature: float = 0.0) -> int:
    """
    Stub implementation of token generation for testing.

    Returns pre-defined fake tokens one at a time, cycling through the
    fake_tokens array repeatedly. Useful for testing the API without
    a real model backend.

    Args:
        tokens: The input token sequence (ignored in stub)
        temperature: Sampling temperature (ignored in stub)

    Returns:
        The next token from the fake_tokens sequence

    Note:
        Sleeps for 0.1s per token to simulate generation latency
    """
    global token_queue
    next_tok = token_queue.pop(0)
    # When we run out of tokens, reset the queue to start over
    if len(token_queue) == 0:
        token_queue = fake_tokens.copy()
    time.sleep(0.1)  # Simulate generation delay
    return next_tok
