"""
Ollama inference backend for the Responses API.

This backend connects to a locally running Ollama service to perform inference.
Ollama (https://ollama.ai) is a tool that makes it easy to run LLMs locally.

Features:
- Works with any model supported by Ollama
- Streaming text generation from Ollama
- Token-level streaming via background thread
- Automatic timeout and error handling

Limitations:
- Does NOT use prompt caching (each request is independent)
- Can be slow between conversation turns
- Requires Ollama service running on localhost:11434
- Token-by-token streaming adds some overhead

This backend is primarily for:
- Testing and development
- Running alternative models via Ollama
- Quick prototyping without GPU access

Usage:
    # First, start Ollama and pull a model:
    ollama pull mistral

    # Then start the server:
    python -m gpt_oss.responses_api.serve --inference-backend ollama --checkpoint mistral

Note: The checkpoint parameter should be the Ollama model name (e.g., "mistral", "llama2")
"""

import json
import threading
import time
from typing import Callable, Optional

import requests
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

EOS_TOKEN = 200002  # End-of-sequence token (used on hard timeout)

# Configuration parameters
POLL_INTERVAL_S = 0.01  # How often to check for new tokens (10ms)
CALL_MAX_WAIT_S = 0.250  # Max time to wait in a single infer_next_token call (250ms)
NO_TOKEN_TIMEOUT_S = 15.0  # Timeout for inactivity before returning EOS
FIRST_BYTE_TIMEOUT_S = 30.0  # Timeout for first token from Ollama

# Shared state (global variables for managing the streaming connection)
_token_buffer: list[int] = []  # Queue of tokens received from Ollama
_buffer_lock = threading.Lock()  # Protect token buffer from concurrent access
_stream_thread: Optional[threading.Thread] = None  # Background thread for streaming
_stream_done = threading.Event()  # Signals when streaming is complete
_stream_error: Optional[Exception] = None  # Stores any error from the stream
_last_progress_ts: float = 0.0  # Last time we made progress (enqueued/dequeued tokens)
_previous_request_tokens: list[int] = []  # Previously processed tokens (unused currently)


def lcp(cache: list[int], inp: list[int]) -> list[int]:
    """
    Find longest common prefix between two token sequences.

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


def _now():
    """Get current monotonic time in seconds."""
    return time.monotonic()


def _touch_progress():
    """Update the last progress timestamp to current time."""
    global _last_progress_ts
    _last_progress_ts = _now()


def _reset_stream_state():
    """Reset all streaming state for a new request."""
    global _token_buffer, _stream_thread, _stream_error
    with _buffer_lock:
        _token_buffer = []
    _stream_done.clear()
    _stream_thread = None
    _stream_error = None
    _touch_progress()


def setup_model(checkpoint: str) -> Callable[[list[int], float, bool], int]:
    """
    Initialize the Ollama backend.

    Creates an inference function that:
    1. Decodes token IDs to text using Harmony encoding
    2. Sends text to Ollama's /api/generate endpoint
    3. Streams responses in a background thread
    4. Re-tokenizes the streaming text into tokens
    5. Returns tokens one at a time from a buffer

    Args:
        checkpoint: Ollama model name (e.g., "mistral", "llama2")

    Returns:
        Inference function ready for use by the API server
    """
    # Load Harmony tokenizer for encoding/decoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    model_name = checkpoint

    def _start_stream(token_ids: list[int], temperature: float):
        """
        Start streaming generation from Ollama in a background thread.

        Decodes tokens to text, sends to Ollama, and accumulates the
        streaming response back into tokens.

        Args:
            token_ids: Input tokens to decode and send
            temperature: Sampling temperature

        Returns:
            Thread handle for the background stream
        """
        prompt_text = encoding.decode(token_ids)

        def run():
            nonlocal prompt_text, temperature
            global _stream_error
            global _previous_request_tokens

            accum_text = ""
            last_len = 0  # number of tokens already emitted

            try:
                url = "http://localhost:11434/api/generate"

                payload = {
                    "model": model_name,
                    "prompt": prompt_text,
                    "stream": True,
                    "options": {"temperature": temperature},
                    "raw": True,
                }

                with requests.post(url, json=payload, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        obj = json.loads(line)

                        if isinstance(obj.get("response"), str):
                            accum_text += obj["response"]
                            toks = encoding.encode(accum_text, allowed_special="all")
                            if len(toks) > last_len:
                                new_toks = toks[last_len:]
                                with _buffer_lock:
                                    _token_buffer.extend(new_toks)
                                last_len = len(toks)
                                _touch_progress()

                        if obj.get("done", False):
                            _token_buffer.append(EOS_TOKEN)
                            last_len = len(toks)
                            _touch_progress()
                            break

                _stream_done.set()

            except Exception as e:
                _stream_error = e
                _stream_done.set()

        t = threading.Thread(target=run, name="ollama-stream", daemon=True)
        t.start()
        return t

    def infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        """
        - Starts a new Ollama stream on new_request.
        - Forwards tokens as they arrive.
        - Only emits EOS_TOKEN if we exceed an inactivity timeout.
        """
        global _stream_thread

        if new_request:
            _reset_stream_state()
            _stream_thread = _start_stream(token_ids=tokens, temperature=temperature)
            # Wait for first byte within FIRST_BYTE_TIMEOUT_S (without emitting EOS early)
            start = _now()
            while _now() - start < FIRST_BYTE_TIMEOUT_S:
                with _buffer_lock:
                    if _token_buffer:
                        tok = _token_buffer.pop(0)
                        _touch_progress()
                        return tok
                if _stream_error is not None:
                    raise RuntimeError(f"Ollama stream error: {_stream_error!r}")
                # If Ollama finished instantly with no output, continue loop until timeout
                time.sleep(POLL_INTERVAL_S)
            # Hard first-byte timeout -> emit EOS so the server can stop this request
            return EOS_TOKEN

        if _stream_error is not None:
            raise RuntimeError(f"Ollama stream error: {_stream_error!r}")

        # Normal path: wait up to CALL_MAX_WAIT_S for a token to arrive
        wait_start = _now()
        while _now() - wait_start < CALL_MAX_WAIT_S:
            with _buffer_lock:
                if _token_buffer:
                    tok = _token_buffer.pop(0)
                    _touch_progress()
                    return tok
            # No token yet; if we've been idle too long overall, end with EOS
            if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
                return EOS_TOKEN
            time.sleep(POLL_INTERVAL_S)

        # Still no token in this call slice. Do NOT send EOS unless we've timed out.
        if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
            return EOS_TOKEN

        # Tell caller to call us again; block minimally by returning *nothing new*.
        # We must return an int; safest is to wait a tiny bit longer for a token.
        # If still none, keep returning only after short waits. Avoid EOS here.
        # One more short wait to reduce hot-looping:
        time.sleep(POLL_INTERVAL_S)
        with _buffer_lock:
            if _token_buffer:
                tok = _token_buffer.pop(0)
                _touch_progress()
                return tok

        # As a last resort for this call slice, return EOS only on true inactivity timeout.
        if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
            return EOS_TOKEN

        # If we reach here, we still haven't got a token—ask the caller to call again soon.
        # Return a harmless token that the server will replace/ignore if your interface supports it.
        # If your interface does NOT allow a sentinel, keep the short-blocking behavior above.
        return (
            EOS_TOKEN if False else 0
        )  # replace `0` with a PAD/NOOP token your server ignores

    return infer_next_token
