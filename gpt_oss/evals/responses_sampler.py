"""
Responses API Sampler

This module implements a sampler that uses the OpenAI Responses API, which is
a newer API format designed specifically for reasoning models. The Responses API
differs from Chat Completions in several ways:

Key Differences from Chat Completions:
1. Uses "developer" role instead of "system" for instructional messages
2. Uses "input" parameter instead of "messages"
3. Uses "max_output_tokens" instead of "max_tokens"
4. Supports explicit reasoning control via reasoning.effort parameter
5. Returns output in a different format with potential multiple output blocks

When to Use:
- Recommended for reasoning models (e.g., o1, o3, gpt-oss with reasoning)
- Provides better control over reasoning behavior
- More efficient for extended reasoning tasks

API Response Structure:
The API may return multiple output blocks:
- Text blocks with reasoning traces
- Final answer blocks
All blocks are appended to the conversation history.
"""

import time
from typing import Any

import openai
from openai import OpenAI

from .types import MessageList, SamplerBase, SamplerResponse


class ResponsesSampler(SamplerBase):
    """
    Sampler that uses the OpenAI Responses API.

    This is the recommended sampler for reasoning models, as it provides:
    - Better control over reasoning effort
    - More efficient handling of long reasoning chains
    - Native support for multi-turn reasoning

    Example Usage:
        sampler = ResponsesSampler(
            model="gpt-oss-120b",
            reasoning_model=True,
            reasoning_effort="high",
            temperature=1.0,
            max_tokens=131_072,
            base_url="http://localhost:8000/v1"
        )

        response = sampler([{"role": "user", "content": "Solve this problem..."}])
    """

    def __init__(
        self,
        model: str,
        developer_message: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 131_072,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        base_url: str = "http://localhost:8000/v1",
    ):
        """
        Initialize the Responses API sampler.

        Args:
            model: Model identifier (e.g., "gpt-oss-120b", "o1-preview")
            developer_message: Optional developer message (similar to system message
                in Chat Completions, but uses "developer" role)
            temperature: Sampling temperature [0.0, 2.0]. Default 1.0 for reasoning models
            max_tokens: Maximum output tokens (note: input + output should stay within limits)
            reasoning_model: Whether this model supports reasoning
            reasoning_effort: For reasoning models: "low", "medium", or "high"
            base_url: API endpoint URL
        """
        # Set very long timeout for reasoning models that may take significant time
        self.client = OpenAI(base_url=base_url, timeout=24*60*60)
        self.model = model
        self.developer_message = developer_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"  # Format for image inputs (if supported)
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        """
        Format a message in the Responses API format.

        Args:
            role: Message role ("user", "assistant", "developer")
            content: Message content

        Returns:
            Dictionary with "role" and "content" keys
        """
        return {"role": role, "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Generate a response using the Responses API.

        This method:
        1. Prepends developer message if configured
        2. Calls the Responses API with appropriate parameters
        3. Handles reasoning effort configuration
        4. Processes multiple output blocks (text, reasoning, etc.)
        5. Appends all output to the conversation history
        6. Retries on rate limits with exponential backoff

        Args:
            message_list: Conversation history

        Returns:
            SamplerResponse with generated text, full conversation, and metadata
        """
        # Prepend developer message if configured
        if self.developer_message:
            message_list = [
                self._pack_message("developer", self.developer_message)
            ] + message_list

        trial = 0  # Track retry attempts for exponential backoff

        # Retry loop for handling transient errors and rate limits
        while True:
            try:
                # Construct the API request parameters
                # Note: Responses API uses "input" and "max_output_tokens"
                # instead of "messages" and "max_tokens"
                request_kwargs = {
                    "model": self.model,
                    "input": message_list,  # "input" instead of "messages"
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,  # "max_output_tokens" instead of "max_tokens"
                }

                # Add reasoning configuration for reasoning models
                if self.reasoning_model:
                    request_kwargs["reasoning"] = (
                        {"effort": self.reasoning_effort} if self.reasoning_effort else None
                    )

                # Call the Responses API
                response = self.client.responses.create(**request_kwargs)

                # Process all output blocks and append them to the conversation
                # The Responses API may return multiple output blocks:
                # - Reasoning traces (intermediate thinking)
                # - Final answers
                # All are appended to maintain full context
                for output in response.output:
                    if hasattr(output, "text"):
                        # Simple text output (most common case)
                        message_list.append(self._pack_message(getattr(output, "role", "assistant"), output.text))
                    elif hasattr(output, "content"):
                        # Structured content output (multimodal or complex responses)
                        for c in output.content:
                            # Content blocks are processed here
                            # (Implementation detail: c.text is handled elsewhere)
                            pass

                # Return the complete response
                # output_text is the concatenated final answer from all blocks
                return SamplerResponse(
                    response_text=response.output_text,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )

            except openai.BadRequestError as e:
                # BadRequestError indicates a problem with our request (not retryable)
                # Return empty response rather than crashing the evaluation
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )

            except Exception as e:
                # Other exceptions (rate limits, network errors) should be retried
                # Use exponential backoff: wait 1s, 2s, 4s, 8s, etc.
                exception_backoff = 2**trial  # Note: typo "expontial" in original
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # If an unknown error occurs that's not caught above, it will propagate
