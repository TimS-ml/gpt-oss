"""
Simple Browser Tool Module for GPT-OSS

This module provides web browsing and search capabilities to the model, enabling it to:
- Search the web using providers like Exa or You.com
- Open and navigate web pages
- Find text within pages
- Extract and cite information from web content

Architecture:
-------------
The browser tool consists of three main components:

1. SimpleBrowserTool (simple_browser_tool.py):
   - Main tool implementation
   - Provides search(), open(), find() functions to the model
   - Maintains browsing history and state
   - Handles citation formatting with special markers like 【0†source】

2. Backend (backend.py):
   - Abstraction for different search/fetch providers
   - ExaBackend: Uses Exa Search API
   - YouComBackend: Uses You.com Search API
   - Handles HTML fetching and API communication

3. PageContents (page_contents.py):
   - Converts HTML to model-readable text
   - Processes links with numbered markers
   - Extracts metadata and snippets
   - Handles images, math, and special formatting

Integration with Model:
-----------------------
The model learns to:
- Call browser.search(query="...") to search the web
- Call browser.open(id=N) to open links from search results
- Call browser.find(pattern="...") to search within a page
- Cite sources using the format: 【cursor†L{start}-L{end}】

Citation Format:
----------------
The tool uses a special bracket notation for citations:
- 【0†Title†domain.com】 - Link in page content
- 【6†L9-L11】 - Citation pointing to lines 9-11 of page at cursor 6
- The cursor is an index into the browsing history

This format allows the model to:
- Attribute information to specific sources
- Point to specific sections of web pages
- Enable users to verify claims

Example Usage:
--------------
1. Model: browser.search(query="Python asyncio tutorial")
   Tool: Returns search results with numbered links

2. Model: browser.open(id=0)
   Tool: Opens first search result, displays content with line numbers

3. Model: "Asyncio provides async/await syntax【0†L15-L20】"
   (The citation refers to lines 15-20 of the opened page)

Safety:
-------
- All HTML content is sanitized before being sent to the model
- No JavaScript execution (static HTML only)
- Rate limiting and timeouts to prevent abuse
- Domain validation to prevent SSRF attacks
"""

from .simple_browser_tool import SimpleBrowserTool
from .backend import ExaBackend, YouComBackend

__all__ = [
    "SimpleBrowserTool",
    "ExaBackend",
    "YouComBackend",
]
