"""
GPT-OSS Integration with Anthropic Agents SDK and MCP

This example demonstrates how to use gpt-oss with the Anthropic Agents SDK,
including integration with Model Context Protocol (MCP) servers for extended
functionality like filesystem operations.

Dependencies:
    - openai: OpenAI Python client (for API compatibility)
    - anthropic-agents: Anthropic's Agents SDK
    - Node.js with npx: Required for MCP filesystem server

Setup:
    1. Install Python dependencies:
       pip install openai anthropic-agents

    2. Install Node.js and npx:
       npm install -g npx

    3. Start gpt-oss server locally:
       The server should be running at http://localhost:11434/v1
       (This example uses Ollama-style endpoint)

    4. Run this example:
       python example.py

Features:
    - Async agent execution with streaming
    - Custom function tools (e.g., get_weather)
    - MCP server integration for filesystem operations
    - Real-time event streaming and processing

Integration with gpt-oss:
    This example connects to gpt-oss via OpenAI-compatible API, allowing the
    Agents SDK to use gpt-oss models while maintaining compatibility with the
    broader agent ecosystem.

MCP (Model Context Protocol):
    MCP servers provide standardized tools for models. This example uses the
    @modelcontextprotocol/server-filesystem package to give the agent file
    system access in a controlled manner.
"""

import asyncio
from pathlib import Path
import shutil

from openai import AsyncOpenAI
from agents import (
    Agent,
    ItemHelpers,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    function_tool,
)
from agents.mcp import MCPServerStdio


async def prompt_user(question: str) -> str:
    """
    Asynchronously prompt user for input.

    This function wraps the synchronous input() function to make it work
    with async code without blocking the event loop.

    Args:
        question (str): The prompt to display to the user

    Returns:
        str: The user's input
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, question)


async def main():
    """
    Main function that sets up and runs the agent with gpt-oss and MCP tools.

    This function:
    1. Configures the OpenAI client to connect to gpt-oss
    2. Sets up an MCP filesystem server for file operations
    3. Creates an agent with custom tools and instructions
    4. Runs the agent with user input and streams the results
    """

    # Set up OpenAI client for local gpt-oss server
    # Using Ollama-style endpoint format (common for local LLM servers)
    openai_client = AsyncOpenAI(
        api_key="local",  # Placeholder key for local server
        base_url="http://localhost:11434/v1",  # gpt-oss server endpoint
    )

    # Get current working directory to provide filesystem access
    samples_dir = str(Path.cwd())

    # Create MCP server for filesystem operations
    # This uses the official MCP filesystem server via npx
    # The server will provide tools for reading/writing files in samples_dir
    mcp_server = MCPServerStdio(
        name="Filesystem MCP Server, via npx",
        params={
            "command": "npx",  # Node package executor
            "args": [
                "-y",  # Auto-confirm package installation
                "@modelcontextprotocol/server-filesystem",  # MCP filesystem package
                samples_dir,  # Root directory for filesystem access
            ],
        },
    )

    # Connect to MCP server (starts the subprocess)
    await mcp_server.connect()

    # Configure the Agents SDK global settings
    set_tracing_disabled(True)  # Disable tracing for cleaner output
    set_default_openai_client(openai_client)  # Use our gpt-oss client
    set_default_openai_api("chat_completions")  # Use chat completions API format

    # Define a custom weather tool using the @function_tool decorator
    # This tool will be available to the agent during execution
    @function_tool
    async def get_weather(location: str) -> str:
        """
        Get the current weather for a location.

        This is a mock implementation that always returns sunny weather.
        In a real application, this would call a weather API.

        Args:
            location (str): The city/location to get weather for

        Returns:
            str: Weather description
        """
        return f"The weather in {location} is sunny."

    # Create the agent with combined tools from custom functions and MCP
    agent = Agent(
        name="My Agent",
        instructions="You are a helpful assistant.",
        tools=[get_weather],  # Custom function tools
        model="gpt-oss:20b-test",  # gpt-oss model identifier
        mcp_servers=[mcp_server],  # MCP servers provide additional tools
    )

    # Get user input from the terminal
    user_input = await prompt_user("> ")

    # Run the agent with streaming enabled
    # This allows us to see tool calls and responses as they happen
    result = Runner.run_streamed(agent, user_input)

    # Process streaming events from the agent execution
    # Different event types represent different stages of agent processing
    async for event in result.stream_events():
        # Raw response events contain low-level API data - skip for cleaner output
        if event.type == "raw_response_event":
            continue

        # Agent updated events occur when agent state changes
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")

        # Run item events represent discrete actions/outputs during execution
        elif event.type == "run_item_stream_event":
            # Tool call started
            if event.item.type == "tool_call_item":
                print("-- Tool was called")

            # Tool call completed with output
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")

            # Agent generated a message output
            elif event.item.type == "message_output_item":
                print(
                    f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                )

            # Other event types (can be extended as needed)
            else:
                pass

    print("=== Run complete ===")


if __name__ == "__main__":
    # Verify that npx is installed before running
    # npx is required for the MCP filesystem server
    if not shutil.which("npx"):
        raise RuntimeError(
            "npx is not installed. Please install it with `npm install -g npx`."
        )

    # Run the async main function
    asyncio.run(main())
