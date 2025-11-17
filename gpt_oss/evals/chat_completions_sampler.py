"""
Chat Completions API Sampler

This module implements a sampler that uses the OpenAI Chat Completions API
(or compatible APIs) for model inference. This is the standard API format
used by most language models.

The sampler supports:
- Standard chat completion models (GPT-3.5, GPT-4, etc.)
- Reasoning models with configurable reasoning_effort
- Custom system messages
- Automatic retry with exponential backoff on rate limits
- Appending reasoning traces to conversation history

API Compatibility:
- Works with OpenAI API (https://api.openai.com/v1)
- Works with local servers implementing the same interface
- Compatible with vLLM, TGI, and other OpenAI-compatible servers
"""

import time
from typing import Any

import openai
from openai import OpenAI

from .types import MessageList, SamplerBase, SamplerResponse

# Standard system message for OpenAI API
OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."

# More detailed system message mimicking ChatGPT interface
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionsSampler(SamplerBase):
    """
    Sampler that uses the OpenAI Chat Completions API.

    This sampler implements the standard chat completions endpoint, which is
    widely supported across different model providers. It handles:
    - System message injection
    - Temperature-based sampling
    - Reasoning model support with configurable effort
    - Automatic retry logic for rate limits and transient errors
    - Token usage tracking

    Example Usage:
        # For evaluation models
        sampler = ChatCompletionsSampler(
            model="gpt-oss-120b",
            reasoning_model=True,
            reasoning_effort="high",
            base_url="http://localhost:8000/v1"
        )

        # For grading models (e.g., HealthBench)
        grader = ChatCompletionsSampler(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            base_url="https://api.openai.com/v1"
        )
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        base_url: str = "http://localhost:8000/v1",
    ):
        """
        Initialize the Chat Completions sampler.

        Args:
            model: Model identifier (e.g., "gpt-oss-120b", "gpt-4.1-2025-04-14")
            system_message: Optional system message prepended to all conversations
            temperature: Sampling temperature [0.0, 2.0]. Higher = more random
            max_tokens: Maximum number of tokens to generate
            reasoning_model: Whether this is a reasoning model (supports reasoning_effort)
            reasoning_effort: For reasoning models: "low", "medium", or "high"
            base_url: API endpoint URL (OpenAI or compatible server)
        """
        # Set a very long timeout (24 hours) for reasoning models that may take time
        self.client = OpenAI(base_url=base_url, timeout=24 * 60 * 60)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.image_format = "url"  # Format for image inputs (if supported)

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        """
        Format a message in the expected API format.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content (text or multimodal)

        Returns:
            Dictionary with "role" and "content" keys
        """
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Generate a completion for the given conversation.

        This method:
        1. Prepends system message if configured
        2. Calls the Chat Completions API
        3. Handles reasoning model parameters if applicable
        4. Retries on rate limits with exponential backoff
        5. Appends reasoning trace to conversation if present
        6. Returns the response with metadata

        Args:
            message_list: Conversation history to complete

        Returns:
            SamplerResponse containing the generated text, full conversation,
            and usage metadata
        """
        # Prepend system message if configured
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list

        trial = 0  # Track retry attempts for exponential backoff

        # Retry loop for handling transient errors and rate limits
        while True:
            try:
                # Call the appropriate API based on model type
                if self.reasoning_model:
                    # Reasoning models support the reasoning_effort parameter
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        reasoning_effort=self.reasoning_effort,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    # Standard models don't support reasoning_effort
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                # Extract the response from the API result
                choice = response.choices[0]
                content = choice.message.content

                # If the model produced reasoning traces, append them to the conversation
                # This allows future messages to reference the reasoning
                if getattr(choice.message, "reasoning", None):
                    message_list.append(self._pack_message("assistant", choice.message.reasoning))

                # Validate that we got a non-empty response
                if not content:
                    raise ValueError("OpenAI API returned empty response; retrying")

                # Return successful response with usage metadata
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )

            except openai.BadRequestError as e:
                # BadRequestError indicates a problem with our request (not retryable)
                # Return a placeholder response rather than crashing the evaluation
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )

            except Exception as e:
                # Other exceptions (rate limits, network errors) should be retried
                # Use exponential backoff: wait 1s, 2s, 4s, 8s, etc.
                exception_backoff = 2 ** trial
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # If an unknown error occurs that's not caught above, it will propagate
