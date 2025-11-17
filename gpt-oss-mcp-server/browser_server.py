"""
GPT-OSS MCP Server for Web Browser Search

This module provides a Model Context Protocol (MCP) server that enables web browsing
and search capabilities for gpt-oss. It supports multiple search backends (Exa, You.com)
and provides tools for searching, opening links, and finding content on web pages.

Dependencies:
    - mcp: Model Context Protocol library
    - gpt_oss: GPT-OSS tools for browser functionality
    - Exa API key or You.com API key (depending on backend choice)

Setup:
    1. Install dependencies:
       pip install mcp gpt-oss

    2. Set environment variable for search backend:
       export BROWSER_BACKEND=exa  # or "youcom"

    3. Configure API key for your chosen backend:
       export EXA_API_KEY=your_key  # for Exa backend
       export YOUCOM_API_KEY=your_key  # for You.com backend

    4. Run the server:
       python browser_server.py
       (Server runs on port 8001 by default)

Features:
    - Multi-session browser management (separate browser per client)
    - Search: Find information with top-N results
    - Open: Navigate to links or URLs with optional location scrolling
    - Find: Search for patterns within the current page
    - Citation support with cursor-based references

Integration with gpt-oss:
    This MCP server exposes browser tools that gpt-oss can use to search the web,
    navigate to pages, and extract information. The server maintains separate
    browser sessions for each client to prevent cross-client interference.

Backends:
    - Exa: AI-powered search engine optimized for semantic queries
    - You.com: Privacy-focused search engine with comprehensive results
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Union, Optional

from mcp.server.fastmcp import Context, FastMCP
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import YouComBackend, ExaBackend

@dataclass
class AppContext:
    """
    Application context for managing browser sessions.

    This class maintains a dictionary of browser instances, one per client session.
    Each browser has its own state (current page, cursor position, etc.) to enable
    multi-client usage without interference.
    """
    browsers: dict[str, SimpleBrowserTool] = field(default_factory=dict)

    def create_or_get_browser(self, session_id: str) -> SimpleBrowserTool:
        """
        Get existing browser for a session or create a new one.

        The browser backend is selected based on the BROWSER_BACKEND environment
        variable. Each client gets its own isolated browser instance.

        Args:
            session_id (str): Unique identifier for the client session

        Returns:
            SimpleBrowserTool: Browser instance for this session

        Raises:
            ValueError: If BROWSER_BACKEND is set to an unsupported value
        """
        if session_id not in self.browsers:
            # Select backend based on environment variable
            tool_backend = os.getenv("BROWSER_BACKEND", "exa")

            if tool_backend == "youcom":
                backend = YouComBackend(source="web")
            elif tool_backend == "exa":
                backend = ExaBackend(source="web")
            else:
                raise ValueError(f"Invalid tool backend: {tool_backend}")

            # Create new browser instance for this session
            self.browsers[session_id] = SimpleBrowserTool(backend=backend)

        return self.browsers[session_id]

    def remove_browser(self, session_id: str) -> None:
        """
        Remove browser instance for a session (cleanup).

        Args:
            session_id (str): Session to clean up
        """
        self.browsers.pop(session_id, None)


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Lifecycle manager for the MCP server.

    This async context manager creates and provides the AppContext for the
    duration of the server's lifetime. The context manages browser sessions
    across all connected clients.

    Args:
        _server: FastMCP server instance (unused)

    Yields:
        AppContext: The application context with browser session management
    """
    yield AppContext()


# Initialize MCP server with browser capabilities
# The instructions teach the model how to properly cite sources from browser output
mcp = FastMCP(
    name="browser",
    instructions=r"""
Tool for browsing.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Cite information from the tool using the following format:
`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
Do not quote more than 10 words directly from the tool output.
sources=web
""".strip(),
    lifespan=app_lifespan,  # Provide application context lifecycle
    port=8001,  # Server listens on port 8001
)


@mcp.tool(
    name="search",
    title="Search for information",
    description=
    "Searches for information related to `query` and displays `topn` results.",
)
async def search(ctx: Context,
                 query: str,
                 topn: int = 10,
                 source: Optional[str] = None) -> str:
    """
    Search the web for information related to a query.

    This tool performs a web search using the configured backend (Exa or You.com)
    and returns the top N results with citations. Each result is assigned a cursor
    that can be used with the 'open' tool to navigate to that result.

    Args:
        ctx (Context): MCP context containing client session info
        query (str): Search query string
        topn (int, optional): Number of results to return. Defaults to 10.
        source (Optional[str], optional): Source filter (e.g., "web"). Defaults to None.

    Returns:
        str: Formatted search results with cursors for citation

    Example:
        query = "Python async programming"
        results = await search(ctx, query, topn=5)
        # Results will include cursors like [0], [1], etc. that can be opened
    """
    # Get or create browser instance for this client
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)

    # Collect all search result messages
    messages = []
    async for message in browser.search(query=query, topn=topn, source=source):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)

    # Combine all messages into a single response
    return "\n".join(messages)


@mcp.tool(
    name="open",
    title="Open a link or page",
    description="""
Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.
Valid link ids are displayed with the formatting: `【{id}†.*】`.
If `cursor` is not provided, the most recent page is implied.
If `id` is a string, it is treated as a fully qualified URL associated with `source`.
If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
Use this function without `id` to scroll to a new location of an opened page.
""".strip(),
)
async def open_link(ctx: Context,
                    id: Union[int, str] = -1,
                    cursor: int = -1,
                    loc: int = -1,
                    num_lines: int = -1,
                    view_source: bool = False,
                    source: Optional[str] = None) -> str:
    """
    Open a link from search results or navigate to a specific URL.

    This tool can:
    1. Open a link by its ID from previous search results
    2. Navigate to a direct URL (when id is a string)
    3. Scroll to a specific line within an already-opened page
    4. View the HTML source of a page (with view_source=True)

    Args:
        ctx (Context): MCP context containing client session info
        id (Union[int, str], optional): Link ID from search results or direct URL. Defaults to -1.
        cursor (int, optional): Which search result page to get the link from. Defaults to -1 (most recent).
        loc (int, optional): Line number to scroll to. Defaults to -1 (auto or top).
        num_lines (int, optional): Number of lines to display. Defaults to -1 (auto).
        view_source (bool, optional): Whether to view HTML source. Defaults to False.
        source (Optional[str], optional): Source hint for URL resolution. Defaults to None.

    Returns:
        str: Page content with line numbers and cursor for citation

    Examples:
        # Open first search result
        await open_link(ctx, id=0)

        # Open a direct URL
        await open_link(ctx, id="https://example.com")

        # Scroll to line 50 of current page
        await open_link(ctx, loc=50)
    """
    # Get or create browser instance for this client
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)

    # Collect all page content messages
    messages = []
    async for message in browser.open(id=id,
                                      cursor=cursor,
                                      loc=loc,
                                      num_lines=num_lines,
                                      view_source=view_source,
                                      source=source):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)

    # Combine all messages into a single response
    return "\n".join(messages)


@mcp.tool(
    name="find",
    title="Find pattern in page",
    description=
    "Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
)
async def find_pattern(ctx: Context, pattern: str, cursor: int = -1) -> str:
    """
    Find exact matches of a text pattern within the current page.

    This tool searches for an exact string match within the currently opened page
    (or a specific page identified by cursor) and returns all matching locations
    with line numbers for citation.

    Args:
        ctx (Context): MCP context containing client session info
        pattern (str): Exact text pattern to search for (case-sensitive)
        cursor (int, optional): Which page to search in. Defaults to -1 (current page).

    Returns:
        str: List of matches with line numbers and surrounding context

    Example:
        # Search for "async def" in the current page
        matches = await find_pattern(ctx, "async def")
        # Returns lines containing the pattern with citation cursors
    """
    # Get or create browser instance for this client
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)

    # Collect all match result messages
    messages = []
    async for message in browser.find(pattern=pattern, cursor=cursor):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)

    # Combine all messages into a single response
    return "\n".join(messages)
