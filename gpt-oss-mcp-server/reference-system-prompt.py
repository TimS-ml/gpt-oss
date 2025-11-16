"""
GPT-OSS Reference System Prompt Builder with Native Tools

This script demonstrates how to build system prompts for gpt-oss using native
gpt-oss tools (not MCP servers). It directly instantiates SimpleBrowserTool and
PythonTool to generate a system prompt with their capabilities.

This is a reference implementation showing the difference between:
1. Using MCP servers (build-system-prompt.py) - dynamic, network-based
2. Using native tools (this file) - direct, in-process

Dependencies:
    - gpt_oss: GPT-OSS tools and tokenizer
    - openai_harmony: Harmony protocol for system prompt encoding

Setup:
    1. Install dependencies:
       pip install gpt-oss openai-harmony

    2. Set environment variables for tool backends:
       export YOUCOM_API_KEY=your_key  # For You.com search backend

    3. Run this script:
       python reference-system-prompt.py

    The script will output the complete system prompt that gpt-oss should use
    when these native tools are integrated.

Use Case:
    This is useful for:
    - Understanding how native tools are integrated into system prompts
    - Comparing MCP vs. native tool integration
    - Building standalone gpt-oss deployments without MCP
    - Testing tool configurations before deploying MCP servers

Difference from build-system-prompt.py:
    - No network calls or server connections required
    - Tools are instantiated directly in the process
    - Simpler setup but less flexible than MCP approach
    - Good for single-process deployments

Output:
    The script prints the complete system prompt in the format expected by
    gpt-oss, including native tool definitions and instructions.
"""

import datetime

from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import YouComBackend
from gpt_oss.tools.python_docker.docker_tool import PythonTool
from gpt_oss.tokenizer import tokenizer

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)

# ============================================================================
# Main Script: Build System Prompt with Native Tools
# ============================================================================

# Load the Harmony encoding for gpt-oss
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Build system message content with metadata
system_message_content = (SystemContent.new()
    .with_reasoning_effort(ReasoningEffort.LOW)  # Set default reasoning effort
    .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d")))  # Add current date

# Initialize browser tool with You.com backend
# This provides web search and browsing capabilities
backend = YouComBackend(source="web")
browser_tool = SimpleBrowserTool(backend=backend)

# Add browser tool configuration to system message
# The tool_config property contains the Harmony-formatted tool description
system_message_content = system_message_content.with_tools(
    browser_tool.tool_config)

# Initialize Python execution tool
# This provides sandboxed Python code execution in Docker
python_tool = PythonTool()

# Add Python tool configuration to system message
system_message_content = system_message_content.with_tools(
    python_tool.tool_config)

# Create the system message with all tool configurations
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
