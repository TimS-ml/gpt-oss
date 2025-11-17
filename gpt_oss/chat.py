"""
Interactive chat interface for gpt-oss models with Harmony format and tool support.

This module implements a full-featured chat interface that:
- Parses and generates Harmony-formatted responses (OpenAI's structured format)
- Supports multi-channel outputs (final answer, chain-of-thought reasoning, tool calls)
- Integrates with various tools (web browsing, Python execution, code patching)
- Handles distributed inference across multiple GPUs
- Provides both human-readable and raw output modes

The Harmony format enables structured responses where the model can:
- Show its reasoning process in a separate channel (chain-of-thought)
- Make tool calls with structured parameters
- Provide a final polished answer in the main channel
- All within a single response, properly formatted and parsed

Supported Tools:
- SimpleBrowserTool: Web search and browsing with citation support
- PythonTool: Execute Python code in a sandboxed Docker container
- apply_patch: Apply unified diff patches to modify files

Usage:
    # Basic chat
    python -m gpt_oss.chat model/

    # With web browsing
    python -m gpt_oss.chat --browser model/

    # With Python execution
    python -m gpt_oss.chat --python model/

    # With code patching
    python -m gpt_oss.chat --apply-patch model/

    # Multi-GPU with tensor parallelism
    torchrun --nproc-per-node=4 -m gpt_oss.chat --backend torch model/
"""

import atexit
import argparse
import asyncio
import datetime
import os
from pathlib import Path

# Try to use GNU readline for better line editing (Mac/Linux)
# Falls back to standard readline on Windows
try:
    import gnureadline as readline
except ImportError:
    import readline

import torch
import termcolor

# Import tool implementations
from gpt_oss.tools import apply_patch
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import YouComBackend
from gpt_oss.tools.python_docker.docker_tool import PythonTool

# Import Harmony format classes for structured message handling
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)


# Map user-friendly reasoning effort names to Harmony enum values
# Higher reasoning effort = more chain-of-thought tokens = better quality but slower
REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def get_user_input():
    """
    Get user input in a distributed setting where multiple GPU processes are running.

    In distributed inference (e.g., with torchrun), only rank 0 (the main process)
    should read from stdin. Other ranks wait and receive the input via broadcast
    to ensure all processes have the same input for consistent generation.

    Returns:
        str: The user's input text, synchronized across all distributed processes.

    Note:
        In single-GPU mode (no distributed), this simply calls input().
        In multi-GPU mode, only rank 0 reads input and broadcasts to other ranks.
    """
    # Get the current process rank (0 if not using distributed training)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Only the main process (rank 0) reads from stdin
    if rank == 0:
        user_input = input()
    else:
        user_input = ""  # Other ranks start with empty string

    # Broadcast the input from rank 0 to all other ranks
    user_input_list = [user_input]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(user_input_list, 0)

    return user_input_list[0]


def main(args):
    """
    Main chat loop that handles conversation state, tool calls, and model generation.

    This function:
    1. Initializes the selected inference backend (torch, triton, or vllm)
    2. Loads the Harmony encoding for structured message handling
    3. Sets up the system message with configuration and available tools
    4. Enters an interactive loop where:
       - User messages are collected
       - Model generates responses with Harmony parsing
       - Tool calls are intercepted and executed
       - Results are fed back to the model
       - Final responses are displayed

    Args:
        args: Command-line arguments containing:
            - checkpoint: Path to model checkpoint directory
            - backend: Inference backend (torch/triton/vllm)
            - reasoning_effort: How much reasoning to use (low/medium/high)
            - browser: Enable web browsing tool
            - python: Enable Python execution tool
            - apply_patch: Enable code patching function
            - developer_message: Optional developer instructions
            - context: Maximum context length
            - raw: Whether to show raw tokens instead of parsed output
            - show_browser_results: Whether to display browser search results
    """
    # Initialize the appropriate inference backend based on user selection
    match args.backend:
        case "triton":
            # Triton backend: optimized single-GPU inference with custom CUDA kernels
            from gpt_oss.triton.model import TokenGenerator as TritonGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TritonGenerator(args.checkpoint, args.context, device)

        case "torch":
            # PyTorch backend: supports tensor parallelism for multi-GPU inference
            from gpt_oss.torch.model import TokenGenerator as TorchGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TorchGenerator(args.checkpoint, device)

        case "vllm":
            # vLLM backend: high-throughput inference with PagedAttention
            from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator
            generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=2)

        case _:
            raise ValueError(f"Invalid backend: {args.backend}")

    # Load the Harmony encoding which handles special tokens and message formatting
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Build the system message that configures the model's behavior
    # This includes model identity, reasoning effort, knowledge cutoff, etc.
    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])  # Controls CoT verbosity
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))  # Current date
    )

    # Enable web browsing tool if requested
    if args.browser:
        # Use You.com as the search backend
        backend = YouComBackend(
            source="web",
        )
        browser_tool = SimpleBrowserTool(backend=backend)
        # Add browser tool configuration to system message so model knows it's available
        system_message_content = system_message_content.with_tools(browser_tool.tool_config)

    # Enable Python execution tool if requested
    if args.python:
        python_tool = PythonTool()
        # Add Python tool configuration to system message
        system_message_content = system_message_content.with_tools(python_tool.tool_config)

    # Create the system message and initialize conversation history
    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    messages = [system_message]

    # Configure apply_patch function if enabled
    # This allows the model to modify files by generating unified diff patches
    if args.apply_patch:
        # Load instructions for the apply_patch function from markdown file
        apply_patch_instructions = Path(apply_patch.__file__).parent / "apply_patch.md"
        developer_message = ""
        if args.developer_message:
            # Prepend any custom developer message to the patch instructions
            developer_message = args.developer_message + "\n"
        developer_message += apply_patch_instructions.read_text()

        # Create developer message with apply_patch function definition
        # This tells the model about the function signature and how to use it
        developer_message_content = (
            DeveloperContent.new()
            .with_instructions(developer_message)
            .with_function_tools([
                ToolDescription.new(
                    "apply_patch",
                    "Patch a file",
                    parameters={
                        "type": "string",
                        "description": "Formatted patch code",
                        "default": "*** Begin Patch\n*** End Patch\n",
                    }
                ),
            ])
        )
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    elif args.developer_message:
        # Add standalone developer message without any function tools
        developer_message_content = DeveloperContent.new().with_instructions(args.developer_message)
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    else:
        # No developer message needed
        developer_message_content = None

    # Display system message in either raw token format or human-readable format
    if args.raw:
        # Raw mode: show the actual tokens as the model sees them
        # This is useful for debugging the exact prompt format
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation(conversation)
        system_message = encoding.decode(tokens)
        print(system_message, flush=True, end="")

        # Pre-render user message delimiters for raw mode
        # We split the user message into start and end tokens so we can
        # print user input between them
        empty_user_message_tokens = encoding.render(Message.from_role_and_content(Role.USER, ""))
        user_message_start = encoding.decode(empty_user_message_tokens[:-1])
        user_message_end = encoding.decode(empty_user_message_tokens[-1:])
    else:
        # Human-readable mode: display system configuration in a formatted way
        print(termcolor.colored("System Message:", "cyan"), flush=True)
        print(termcolor.colored("Model Identity:", "cyan"), system_message_content.model_identity, flush=True)
        print(termcolor.colored("Reasoning Effort:", "cyan"), system_message_content.reasoning_effort, flush=True)
        print(termcolor.colored("Conversation Start Date:", "cyan"), system_message_content.conversation_start_date, flush=True)
        print(termcolor.colored("Knowledge Cutoff:", "cyan"), system_message_content.knowledge_cutoff, flush=True)
        print(termcolor.colored("Browser Tool:", "cyan"), "Enabled" if args.browser else "Disabled", flush=True)
        print(termcolor.colored("Python Tool:", "cyan"), "Enabled" if args.python else "Disabled", flush=True)
        print(termcolor.colored("Apply Patch Function:", "cyan"), "Enabled" if args.apply_patch else "Disabled", flush=True)
        if developer_message_content:
            print(termcolor.colored("Developer Message:", "yellow"), flush=True)
            print(developer_message_content.instructions, flush=True)

    # Main chat loop: alternates between user input and model responses
    # The loop continues indefinitely until interrupted (Ctrl+C)
    MESSAGE_PADDING = 12  # Space padding for aligned message labels
    while True:
        # Check the last message to determine the next action
        last_message = messages[-1]

        # If the last message has no recipient, we need user input
        # (recipient is None for regular messages, set for tool calls)
        if last_message.recipient is None:
            # Collect user input with appropriate formatting
            if args.raw:
                # Raw mode: print message delimiters around user input
                print(user_message_start, end="", flush=True)
                user_message = get_user_input()
                print(user_message_end, flush=True, end="")
            else:
                # Human-readable mode: print colored user label
                print(termcolor.colored("User:".ljust(MESSAGE_PADDING), "red"), flush=True)
                user_message = get_user_input()

            # Convert user input to a Harmony message and add to conversation
            user_message = Message.from_role_and_content(Role.USER, user_message)
            messages.append(user_message)
        else:
            # Last message is a tool/function call that needs to be executed
            # The recipient field indicates which tool to call

            # Execute browser/web search tool
            if last_message.recipient.startswith("browser."):
                assert args.browser, "Browser tool is not enabled"
                tool_name = "Search"

                # Define async wrapper to collect all messages from the tool
                # The browser tool yields multiple messages (search results, citations, etc.)
                async def run_tool():
                    results = []
                    async for msg in browser_tool.process(last_message):
                        results.append(msg)
                    return results

                # Execute the async tool and add all results to conversation
                result = asyncio.run(run_tool())
                messages += result

            # Execute Python code execution tool
            elif last_message.recipient.startswith("python"):
                assert args.python, "Python tool is not enabled"
                tool_name = "Python"

                # Define async wrapper to collect all messages from the tool
                # The Python tool runs code in a Docker container and returns output
                async def run_tool():
                    results = []
                    async for msg in python_tool.process(last_message):
                        results.append(msg)
                    return results

                # Execute the async tool and add all results to conversation
                result = asyncio.run(run_tool())
                messages += result

            # Execute apply_patch function to modify files
            elif last_message.recipient == "functions.apply_patch":
                assert args.apply_patch, "Apply patch tool is not enabled"
                tool_name = "Apply Patch"
                text = last_message.content[0].text
                tool_output = None

                # Handle JSON-wrapped patches (some models wrap the patch in JSON)
                if text.startswith("{"):
                    import json
                    try:
                        # Extract patch text from JSON object
                        some_dict = json.loads(text)
                        _, text = some_dict.popitem()
                    except Exception as e:
                        tool_output = f"Error parsing JSON: {e}"

                # Apply the patch to the target file
                if tool_output is None:
                    try:
                        tool_output = apply_patch.apply_patch(text)
                    except Exception as e:
                        tool_output = f"Error applying patch: {e}"

                # Create tool response message with the patch result
                # The message is addressed back to the assistant to continue the conversation
                message = (
                    Message(
                        author=Author.new(Role.TOOL, last_message.recipient),
                        content=[TextContent(text=tool_output)]
                    )
                    .with_recipient("assistant")
                )
                # Preserve the channel if the tool call had one (e.g., CoT channel)
                if last_message.channel:
                    message = message.with_channel(last_message.channel)

                result = [message]
                messages += result
            else:
                # Unknown tool or function - this shouldn't happen
                raise ValueError(f"Unknown tool or function call: {last_message.recipient}")
            # Display the tool execution result to the user
            if args.raw:
                # Raw mode: render tool result as tokens
                rendered_result = encoding.render_conversation(Conversation.from_messages(result))
                print(encoding.decode(rendered_result), flush=True, end="")
            else:
                # Human-readable mode: show tool output with colored label
                print(termcolor.colored(f"{tool_name} output:".ljust(MESSAGE_PADDING), "magenta"), flush=True)
                if tool_name == "Search" and not args.show_browser_results:
                    # Hide search results by default (they can be verbose)
                    # The results are still fed to the model, just not shown to user
                    print("[Search results fed to the model]")
                else:
                    # Show the actual tool output
                    print(result[0].content[0].text)

        # Prepare the conversation for model completion
        # This renders all messages into tokens and sets up for assistant response
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        if args.raw:
            # In raw mode, print the assistant message header tokens
            # (the last two tokens are the role/author markers)
            print(encoding.decode(tokens[-2:]), flush=True, end="")

        # Initialize the Harmony streaming parser
        # This parses tokens as they arrive and extracts structured content
        # (channels, tool calls, text content, etc.)
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        field_created = False  # Track if we've printed a label for current output section
        current_output_text = ""  # Accumulated text for citation processing
        output_text_delta_buffer = ""  # Buffer for handling partial citations

        # Generate and stream the model's response token by token
        # Stop tokens ensure the model stops at appropriate boundaries (tool calls, message end, etc.)
        for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
            # Feed each token to the parser to extract structured content
            parser.process(predicted_token)

            if args.raw:
                # Raw mode: just print tokens as they arrive
                print(encoding.decode([predicted_token]), end="", flush=True)
                continue

            # Check if we're starting a new message field (channel change, new tool call, etc.)
            if parser.state == StreamState.EXPECT_START:
                print("")  # Add newline before new field
                field_created = False  # Reset flag for new field

            # Skip if no new content in this token
            if not parser.last_content_delta:
                continue

            # Print label for new field (Assistant response, CoT, or Tool call)
            if not field_created:
                field_created = True
                if parser.current_channel == "final":
                    # Main assistant response in the "final" channel
                    print(termcolor.colored("Assistant:", "green"), flush=True)
                elif parser.current_recipient is not None:
                    # Tool call - show which tool is being invoked
                    print(termcolor.colored(f"Tool call to {parser.current_recipient}:", "cyan"), flush=True)
                else:
                    # Chain-of-thought reasoning in default channel
                    print(termcolor.colored("CoT:", "yellow"), flush=True)

            # Handle citation normalization for browser tool
            # Citations like [[1]] need to be properly formatted and we buffer
            # partial citations to avoid printing incomplete citation markers
            should_send_output_text_delta = True
            output_text_delta_buffer += parser.last_content_delta

            if args.browser:
                # Normalize citations in the accumulated text
                # has_partial_citations indicates if there's an incomplete citation marker
                updated_output_text, _annotations, has_partial_citations = browser_tool.normalize_citations(current_output_text + output_text_delta_buffer)
                output_text_delta_buffer = updated_output_text[len(current_output_text):]

                # If we have a partial citation (e.g., just "[[1"), buffer it
                # until we get the complete marker
                if has_partial_citations:
                    should_send_output_text_delta = False

            # Print the buffered text if it's safe to do so
            if should_send_output_text_delta:
                print(output_text_delta_buffer, end="", flush=True)
                current_output_text += output_text_delta_buffer
                output_text_delta_buffer = ""

        # Add all parsed messages from this generation to the conversation
        # The parser may have extracted multiple messages (e.g., CoT + final answer + tool calls)
        messages += parser.messages


if __name__ == "__main__":
    # Command-line argument parser for configuring the chat interface
    parser = argparse.ArgumentParser(
        description="Chat example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required: Path to the model checkpoint directory
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
    )

    # Reasoning effort: controls how much chain-of-thought the model uses
    # Higher effort = more reasoning tokens = better quality but slower
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        metavar="REASONING_EFFORT",
        type=str,
        default="low",
        choices=["high", "medium", "low"],
        help="Reasoning effort",
    )

    # Enable apply_patch function for code modification
    parser.add_argument(
        "-a",
        "--apply-patch",
        action="store_true",
        help="Make apply_patch function available to the model",
    )

    # Enable web browser/search tool
    parser.add_argument(
        "-b",
        "--browser",
        default=False,
        action="store_true",
        help="Use browser tool",
    )

    # Show detailed browser search results (off by default to reduce clutter)
    parser.add_argument(
        "--show-browser-results",
        default=False,
        action="store_true",
        help="Show browser results",
    )

    # Enable Python code execution tool (runs in Docker container)
    parser.add_argument(
        "-p",
        "--python",
        default=False,
        action="store_true",
        help="Use python tool",
    )

    # Custom developer message for additional instructions
    parser.add_argument(
        "--developer-message",
        default="",
        help="Developer message",
    )

    # Maximum context length (only used by triton backend)
    parser.add_argument(
        "-c",
        "--context",
        metavar="CONTEXT",
        type=int,
        default=8192,
        help="Max context length",
    )

    # Raw mode: show actual tokens instead of parsed/formatted output
    # Useful for debugging the exact prompt format and token structure
    parser.add_argument(
        "--raw",
        default=False,
        action="store_true",
        help="Raw mode (does not render Harmony encoding)",
    )

    # Backend selection: triton (single GPU), torch (multi-GPU), or vllm (high throughput)
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        choices=["triton", "torch", "vllm"],
        help="Inference backend",
    )

    args = parser.parse_args()

    # Set up readline history for better interactive experience
    # Only in single-GPU mode (multi-GPU uses distributed input handling)
    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        histfile = os.path.join(os.path.expanduser("~"), ".chat")
        try:
            # Load previous chat history
            readline.read_history_file(histfile)
            readline.set_history_length(10000)
        except FileNotFoundError:
            # No history file yet, that's fine
            pass

        # Save history when program exits
        atexit.register(readline.write_history_file, histfile)

    # Start the main chat loop
    main(args)
