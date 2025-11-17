"""
Base Tool Abstract Class for GPT-OSS

This module defines the foundational Tool interface that all tools in GPT-OSS must implement.
Tools are the mechanism by which the model can interact with external systems, execute code,
browse the web, and perform other actions beyond text generation.

Key Concepts:
-------------
1. Tool Invocation: The model sends a Message with a recipient field matching the tool's name
2. Message Processing: The tool processes the message and yields response messages
3. Harmony Integration: All messages follow the OpenAI Harmony format with Author, Content, etc.
4. Channel Consistency: Tools maintain channel consistency across multi-turn interactions

Architecture:
-------------
- Tools are async generators that yield Message objects
- Each tool has a unique name used for routing messages
- Tools provide instruction text that describes their capabilities to the model
- Tools can maintain state across invocations (e.g., browser history)
"""

from abc import ABC, abstractmethod
from uuid import UUID, uuid4
from typing import AsyncIterator

from openai_harmony import (
    Author,
    Role,
    Message,
    TextContent,
)


def _maybe_update_inplace_and_validate_channel(
    *, input_message: Message, tool_message: Message
) -> None:
    """
    Ensures channel consistency between input and output messages.

    In multi-channel conversations (e.g., parallel reasoning chains), messages from
    tools must maintain the same channel as the triggering message. This function:
    - Auto-sets the channel on tool messages if unset
    - Raises an error if the tool message has a different, explicitly-set channel

    Args:
        input_message: The message that triggered the tool
        tool_message: The response message from the tool

    Raises:
        ValueError: If tool_message has a different channel than input_message
    """
    # If the channel of a new message produced by tool is different from the originating message,
    # we auto-set the new message's channel, if unset, or raise an error.
    if tool_message.channel != input_message.channel:
        if tool_message.channel is None:
            # Auto-set the channel to match the input
            tool_message.channel = input_message.channel
        else:
            # Explicitly different channel is an error
            raise ValueError(
                f"Messages from tool should have the same channel ({tool_message.channel=}) as "
                f"the triggering message ({input_message.channel=})."
            )


class Tool(ABC):
    """
    Something the model can call.

    Tools expose APIs that are shown to the model in a syntax that the model
    understands and knows how to call (from training data). Tools allow the
    model to do things like run code, browse the web, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        An identifier for the tool. The convention is that a message will be routed to the tool
        whose name matches its recipient field.
        """

    @property
    def output_channel_should_match_input_channel(self) -> bool:
        """
        A flag which indicates whether the output channel of the tool should match the input channel.
        """
        return True

    async def process(self, message: Message) -> AsyncIterator[Message]:
        """
        Public entry point for processing messages sent to this tool.

        This method handles the message routing and channel consistency logic, then delegates
        to the concrete implementation in _process(). Tools should NOT override this method.

        Flow:
        1. Message arrives with recipient=tool.name (already validated by router)
        2. This method calls _process() to generate response messages
        3. Channel consistency is enforced on each yielded message
        4. Messages are yielded back to the conversation manager

        Args:
            message: The incoming message from the model or another source.
                     message.recipient should match this tool's name.
                     message.content contains the tool invocation (e.g., code to execute, URL to browse)

        Yields:
            Message objects with author.role=TOOL to be added to the conversation.
            These are typically sent back to the assistant for further processing.

        Implementation Notes:
        - Do not override this method; override `_process` below (to avoid interfering with tracing).
        - For blocking operations, use `call_on_background_thread` to get a coroutine.
        - For testing, use `evaluate_generator` to get the results synchronously.
        """
        async for m in self._process(message):
            # Enforce channel consistency if required by the tool
            if self.output_channel_should_match_input_channel:
                _maybe_update_inplace_and_validate_channel(input_message=message, tool_message=m)
            yield m

    @abstractmethod
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """
        Core tool implementation that concrete tools must override.

        This method contains the actual tool logic (e.g., executing code, fetching web pages).
        It receives a message from the model and yields response messages.

        Args:
            message: The message to process. Content format depends on the tool:
                     - PythonTool: message.content[0].text contains Python code
                     - SimpleBrowserTool: message.recipient has function name, content has JSON args
                     - ApplyPatchTool: message.content contains patch text

        Yields:
            Message objects with:
            - author.role = Role.TOOL
            - author.name = self.name
            - content = tool output (e.g., execution results, web page content)
            - recipient = "assistant" (so the model processes the response)

        Implementation Notes:
        - This is an async generator, so use 'yield' to return messages
        - For long-running operations, consider yielding intermediate progress messages
        - Handle errors gracefully and return error messages via error_message() helper
        """
        if False:  # This is to convince the type checker that this is an async generator.
            yield  # type: ignore[unreachable]
        _ = message  # Stifle "unused argument" warning.
        raise NotImplementedError

    @abstractmethod
    def instruction(self) -> str:
        """
        Returns a text description of the tool's capabilities for the model.

        This instruction text is typically included in the system prompt or as part of
        the tool configuration sent to the model. It teaches the model:
        - What the tool does
        - How to invoke it (format of messages)
        - What to expect in return
        - Any constraints or limitations

        The model learns from these instructions during training and uses them to
        decide when and how to call tools during inference.

        Returns:
            A string describing the tool's functionality and usage. Examples:
            - PythonTool: Describes Python execution environment, available packages, etc.
            - SimpleBrowserTool: Describes search(), open(), find() functions and citation format
            - ApplyPatchTool: Describes patch format with *** Begin Patch markers

        Example Instruction (for a Python tool):
            '''
            Use this tool to execute Python code. Code is run in a Docker container with
            Python 3.11. The container is stateless, so you must include all code in each
            invocation. Use print() statements to see output.
            '''
        """
        raise NotImplementedError

    def instruction_dict(self) -> dict[str, str]:
        """
        Returns the tool instruction as a dictionary mapping tool name to instruction text.

        This is a convenience method used by systems that manage multiple tools and need
        to build a comprehensive instruction set for the model.

        Returns:
            Dict with single entry: {self.name: self.instruction()}
        """
        return {self.name: self.instruction()}

    def error_message(
        self, error_message: str, id: UUID | None = None, channel: str | None = None
    ) -> Message:
        """
        Constructs a standardized error message from this tool.

        Error messages are sent back to the assistant when tool execution fails.
        They follow the same Message format as successful responses but contain
        error information in the content.

        Args:
            error_message: Human-readable description of what went wrong
            id: Optional message UUID (auto-generated if not provided)
            channel: Optional channel for multi-channel conversations

        Returns:
            A Message object with:
            - author.role = Role.TOOL
            - author.name = self.name
            - content = error text
            - recipient = "assistant"

        Usage:
            try:
                result = dangerous_operation()
            except Exception as e:
                yield self.error_message(f"Operation failed: {str(e)}")

        Note:
            TODO: Consider using a dedicated SystemError content type instead of TextContent
        """
        return Message(
            id=id if id else uuid4(),
            author=Author(role=Role.TOOL, name=self.name),
            content=TextContent(text=error_message), # TODO: Use SystemError instead
            channel=channel,
        ).with_recipient("assistant")

