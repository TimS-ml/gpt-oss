"""
GPT-OSS System Prompt Builder with MCP Server Integration

This script demonstrates how to dynamically build system prompts for gpt-oss by
fetching tool definitions from MCP servers. It connects to running MCP servers,
retrieves their tool schemas, and constructs a complete system prompt using the
OpenAI Harmony encoding format.

Dependencies:
    - gpt_oss: GPT-OSS tokenizer
    - openai_harmony: Harmony protocol for system prompt encoding
    - mcp: Model Context Protocol client library

Setup:
    1. Install dependencies:
       pip install gpt-oss openai-harmony mcp

    2. Start MCP servers (must be running before this script):
       - Browser server: python browser_server.py (port 8001)
       - Python server: python python_server.py (port 8000)

    3. Run this script:
       python build-system-prompt.py

    The script will output the complete system prompt that gpt-oss should use
    when these tools are available.

Use Case:
    This is useful for:
    - Generating system prompts dynamically based on available tools
    - Understanding the prompt structure for gpt-oss with MCP tools
    - Testing and validating tool integration
    - Documentation and debugging

Output:
    The script prints the complete system prompt in the format expected by
    gpt-oss, including tool definitions, instructions, and conversation metadata.
"""

import datetime
import asyncio

from gpt_oss.tokenizer import get_tokenizer

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolNamespaceConfig,
    ToolDescription,
    load_harmony_encoding,
)

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import ListToolsResult


async def list_server_and_tools(server_url: str):
    """
    Connect to an MCP server and retrieve its metadata and tools.

    This function establishes an SSE (Server-Sent Events) connection to an MCP
    server, initializes the session, and retrieves the list of available tools
    with their schemas.

    Args:
        server_url (str): The SSE endpoint URL of the MCP server

    Returns:
        tuple: (initialize_response, list_tools_response)
            - initialize_response: Server metadata (name, version, etc.)
            - list_tools_response: List of tools with their schemas

    Example:
        init_resp, tools_resp = await list_server_and_tools("http://localhost:8001/sse")
    """
    async with sse_client(url=server_url) as streams, ClientSession(
            *streams) as session:
        # Initialize the MCP session
        initialize_response = await session.initialize()

        # Get list of tools provided by this server
        list_tools_response = await session.list_tools()

        return initialize_response, list_tools_response


def trim_schema(schema: dict) -> dict:
    """
    Convert MCP-generated JSON Schema to Harmony's variant.

    MCP servers return JSON schemas in standard format, but Harmony (the protocol
    used by gpt-oss) expects a slightly different schema format. This function:
    1. Removes unnecessary "title" fields
    2. Removes null default values
    3. Converts "anyOf" type unions to array format
    4. Removes "null" from type unions (Harmony ignores them)
    5. Recursively processes nested properties

    Args:
        schema (dict): JSON Schema in MCP format

    Returns:
        dict: JSON Schema in Harmony format

    Example:
        # MCP format
        {"anyOf": [{"type": "string"}, {"type": "null"}]}
        # Becomes Harmony format
        {"type": ["string"]}
    """
    # Remove title field (not needed in Harmony)
    if "title" in schema:
        del schema["title"]

    # Remove null default values
    if "default" in schema and schema["default"] is None:
        del schema["default"]

    # Convert anyOf type unions to array format
    if "anyOf" in schema:
        # Turn "anyOf": [{"type": "type-1"}, {"type": "type-2"}] into "type": ["type-1", "type-2"]
        # if there's more than 1 types, also remove "null" type as Harmony will just ignore it
        types = [
            type_dict["type"] for type_dict in schema["anyOf"]
            if type_dict["type"] != 'null'
        ]
        schema["type"] = types
        del schema["anyOf"]

    # Recursively process nested properties
    if "properties" in schema:
        schema["properties"] = {
            k: trim_schema(v)
            for k, v in schema["properties"].items()
        }

    return schema


def post_process_tools_description(
        list_tools_result: ListToolsResult) -> ListToolsResult:
    """
    Post-process MCP tool descriptions for Harmony compatibility.

    This function:
    1. Converts each tool's input schema from MCP to Harmony format
    2. Filters out tools that shouldn't be included in the prompt (based on annotations)

    Some tools (like the Python tool) are simple text-in/text-out and don't need
    their schema exposed in the prompt. The "include_in_prompt" annotation controls this.

    Args:
        list_tools_result (ListToolsResult): Raw tool list from MCP server

    Returns:
        ListToolsResult: Processed tool list ready for Harmony encoding
    """
    # Convert all tool schemas from MCP to Harmony format
    for tool in list_tools_result.tools:
        tool.inputSchema = trim_schema(tool.inputSchema)

    # Filter out tools that shouldn't be in the prompt
    # Some tools schema don't need to be part of the prompt (e.g. simple text in text out for Python)
    list_tools_result.tools = [
        tool for tool in list_tools_result.tools
        if getattr(tool.annotations, "include_in_prompt", True)
    ]

    return list_tools_result

# ============================================================================
# Main Script: Build System Prompt from MCP Servers
# ============================================================================

# Get the tokenizer for rendering the final prompt
tokenizer = get_tokenizer()

# List of MCP server SSE endpoints to fetch tools from
tools_urls = [
    "http://localhost:8001/sse",  # browser server
    "http://localhost:8000/sse",  # python server
]

# Collect tool descriptions from all servers
harmony_tool_descriptions = []

for tools_url in tools_urls:
    # Connect to MCP server and get tools
    initialize_response, list_tools_response = asyncio.run(
        list_server_and_tools(tools_url))

    # Convert tool schemas to Harmony format
    list_tools_response = post_process_tools_description(list_tools_response)

    # Create a Harmony ToolNamespaceConfig for this server's tools
    tool_from_mcp = ToolNamespaceConfig(
        name=initialize_response.serverInfo.name,  # e.g., "browser", "python"
        description=initialize_response.instructions,  # Server instructions
        tools=[
            ToolDescription.new(name=tool.name,
                                description=tool.description,
                                parameters=tool.inputSchema)
            for tool in list_tools_response.tools
        ])

    harmony_tool_descriptions.append(tool_from_mcp)

# Load the Harmony encoding for gpt-oss
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Build system message content with metadata
system_message_content = (SystemContent.new()
    .with_reasoning_effort(ReasoningEffort.LOW)  # Set default reasoning effort
    .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d")))  # Add current date

# Add all tool descriptions to the system message
for tool_description in harmony_tool_descriptions:
    system_message_content = system_message_content.with_tools(
        tool_description)

# Create the system message
system_message = Message.from_role_and_content(Role.SYSTEM,
                                               system_message_content)

# Create an empty developer message (can be populated with custom instructions)
developer_message_content = DeveloperContent.new().with_instructions("")
developer_message = Message.from_role_and_content(Role.DEVELOPER,
                                                  developer_message_content)

# Combine all messages into a conversation
messages = [system_message, developer_message]

# Render the conversation using Harmony encoding
conversation = Conversation.from_messages(messages)
tokens = encoding.render_conversation(conversation)

# Decode tokens back to text to see the final system prompt
system_message = tokenizer.decode(tokens)

# Print the complete system prompt
print(system_message)
