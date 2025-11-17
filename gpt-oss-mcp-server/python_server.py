"""
GPT-OSS MCP Server for Python Code Execution

This module provides a Model Context Protocol (MCP) server that enables Python code
execution in a sandboxed Docker container. It's designed for use with gpt-oss to
provide secure, isolated code execution capabilities.

Dependencies:
    - mcp: Model Context Protocol library
    - gpt_oss: GPT-OSS tools for Python execution
    - openai_harmony: Harmony protocol for message formatting

Setup:
    1. Install dependencies:
       pip install mcp gpt-oss openai-harmony

    2. Ensure Docker is installed and running on your system

    3. Run the server:
       python python_server.py
       (or integrate with gpt-oss via MCP client)

Features:
    - Executes Python code in stateless Docker containers
    - Secure sandboxed execution environment
    - Returns stdout from code execution
    - Designed for internal reasoning, not user-visible outputs

Integration with gpt-oss:
    This MCP server exposes a 'python' tool that gpt-oss can use to execute
    Python code during its reasoning process. The code execution happens in
    isolation and results are returned to the model for further processing.

Security:
    - All code runs in Docker containers
    - Containers are stateless (destroyed after execution)
    - No persistent file system access
    - Network access controlled by Docker configuration
"""

from mcp.server.fastmcp import FastMCP
from gpt_oss.tools.python_docker.docker_tool import PythonTool
from openai_harmony import Message, TextContent, Author, Role

# Initialize MCP server with Python execution capabilities
# The instructions inform the model how to use this tool properly
mcp = FastMCP(
    name="python",
    instructions=r"""
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you.
""".strip(),
)


@mcp.tool(
    name="python",
    title="Execute Python code",
    description="""
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you.
    """,
    annotations={
        # Harmony format don't want this schema to be part of it because it's simple text in text out
        # This annotation prevents the tool schema from being included in the prompt
        "include_in_prompt": False,
    })
async def python(code: str) -> str:
    """
    Execute Python code in a sandboxed Docker container.

    This tool function:
    1. Creates a PythonTool instance for Docker-based execution
    2. Wraps the code in a Harmony Message format
    3. Processes the code through the Docker container
    4. Collects all output messages
    5. Returns the combined stdout as a string

    Args:
        code (str): Python code to execute

    Returns:
        str: Standard output from the code execution, or error messages if execution failed

    Example:
        code = "print('Hello, world!')"
        result = await python(code)
        # result: "Hello, world!"
    """
    # Initialize the Python Docker tool
    tool = PythonTool()

    # Collect all messages from the tool execution
    messages = []

    # Process the code in Docker container
    # The tool.process method yields messages as the code executes
    async for message in tool.process(
            Message(author=Author(role=Role.TOOL, name="python"),
                    content=[TextContent(text=code)])):
        messages.append(message)

    # Combine all message outputs into a single string
    return "\n".join([message.content[0].text for message in messages])
