"""
Backend Abstraction for Simple Browser Tool

This module defines the backend interface for web search and HTML fetching.
Backends are pluggable components that handle the actual HTTP requests to
search engines and web pages.

Architecture:
-------------
The Backend abstract class defines two core operations:
1. search() - Query a search engine and return results
2. fetch() - Retrieve and process a web page

Concrete implementations:
- ExaBackend: Uses Exa Search API (https://exa.ai)
- YouComBackend: Uses You.com Search API (https://you.com)

Both backends:
- Make HTTP requests to their respective APIs
- Convert API responses to PageContents objects
- Handle errors and retries
- Process HTML into model-readable format

API Keys:
---------
Backends require API keys set via environment variables:
- ExaBackend: EXA_API_KEY
- YouComBackend: YDC_API_KEY

Error Handling:
---------------
- BackendError is raised for API failures, timeouts, etc.
- Retry logic with exponential backoff for transient errors
- Proper error messages returned to the model

Integration:
------------
SimpleBrowserTool receives a Backend instance at initialization and uses it
for all web operations. This allows easy switching between providers or
adding new ones.
"""

import functools
import asyncio
import logging
import os
from abc import abstractmethod
from typing import Callable, ParamSpec, TypeVar
from urllib.parse import quote

import chz
from aiohttp import ClientSession, ClientTimeout
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .page_contents import (
    Extract,
    FetchResult,
    PageContents,
    get_domain,
    process_html,
)

logger = logging.getLogger(__name__)

# Prefix for URLs requesting the raw HTML source view
VIEW_SOURCE_PREFIX = "view-source:"


class BackendError(Exception):
    """
    Raised when a backend operation fails.

    This includes:
    - API authentication failures
    - Network errors
    - Invalid responses
    - Rate limiting
    - Timeouts
    """
    pass


P = ParamSpec("P")
R = TypeVar("R")


def with_retries(
    func: Callable[P, R],
    num_retries: int,
    max_wait_time: float,
) -> Callable[P, R]:
    """
    Decorator to add retry logic with exponential backoff.

    This wraps a function to automatically retry on exceptions with
    exponentially increasing wait times between attempts.

    Args:
        func: The function to wrap
        num_retries: Maximum number of retry attempts
        max_wait_time: Maximum seconds to wait between retries

    Returns:
        Wrapped function with retry logic (or original if num_retries=0)

    Retry Strategy:
        - Wait time starts at 2 seconds
        - Increases exponentially (multiplier=1)
        - Caps at max_wait_time
        - Logs before sleep and after completion
        - Retries on any Exception
    """
    if num_retries > 0:
        retry_decorator = retry(
            stop=stop_after_attempt(num_retries),
            wait=wait_exponential(
                multiplier=1,
                min=2,
                max=max_wait_time,
            ),
            before_sleep=before_sleep_log(logger, logging.INFO),
            after=after_log(logger, logging.INFO),
            retry=retry_if_exception_type(Exception),
        )
        return retry_decorator(func)
    else:
        return func


def maybe_truncate(text: str, num_chars: int = 1024) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if truncated.

    Used to limit error message lengths when reporting back to the model.

    Args:
        text: Text to potentially truncate
        num_chars: Maximum length (default 1024)

    Returns:
        Original text if short enough, otherwise truncated with "..." appended
    """
    if len(text) > num_chars:
        text = text[: (num_chars - 3)] + "..."
    return text


@chz.chz(typecheck=True)
class Backend:
    """
    Abstract base class for web search and fetching backends.

    Backends implement the actual HTTP communication with search engines
    and web servers. Subclasses must implement search() and fetch().

    Attributes:
        source: Human-readable description of the backend (e.g., "Exa Search API")
    """
    source: str = chz.field(doc="Description of the backend source")

    @abstractmethod
    async def search(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> PageContents:
        """
        Perform a web search and return results as a PageContents object.

        Args:
            query: Search query string
            topn: Maximum number of results to return
            session: aiohttp ClientSession for making requests

        Returns:
            PageContents representing the search results page.
            The page contains numbered links that can be opened with browser.open(id=N).

        Raises:
            BackendError: If the search fails
        """
        pass

    @abstractmethod
    async def fetch(self, url: str, session: ClientSession) -> PageContents:
        """
        Fetch and process a web page.

        Args:
            url: URL to fetch (may have VIEW_SOURCE_PREFIX for raw HTML)
            session: aiohttp ClientSession for making requests

        Returns:
            PageContents with processed HTML converted to model-readable text

        Raises:
            BackendError: If the fetch fails
        """
        pass

    async def _post(self, session: ClientSession, endpoint: str, payload: dict) -> dict:
        headers = {"x-api-key": self._get_api_key()}
        async with session.post(f"{self.BASE_URL}{endpoint}", json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"{self.__class__.__name__} error {resp.status}: {await resp.text()}"
                )
            return await resp.json()

    async def _get(self, session: ClientSession, endpoint: str, params: dict) -> dict:
        headers = {"x-api-key": self._get_api_key()}
        async with session.get(f"{self.BASE_URL}{endpoint}", params=params, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"{self.__class__.__name__} error {resp.status}: {await resp.text()}"
                )
            return await resp.json()


@chz.chz(typecheck=True)
class ExaBackend(Backend):
    """
    Backend implementation using the Exa Search API.

    Exa (https://exa.ai) is a search engine optimized for AI applications.
    It provides:
    - Semantic search (understands intent beyond keywords)
    - Clean, structured results
    - Built-in content extraction
    - Summary generation

    Configuration:
        Set EXA_API_KEY environment variable or pass api_key parameter

    API Endpoints:
        - POST /search: Search for web pages
        - POST /contents: Fetch page content with HTML

    Features:
        - Returns summaries with search results
        - Fetches HTML with tags included for better processing
        - Automatic domain extraction
        - Handles view-source: URLs for raw HTML inspection
    """

    source: str = chz.field(doc="Description of the backend source")
    api_key: str | None = chz.field(
        doc="Exa API key. Uses EXA_API_KEY environment variable if not provided.",
        default=None,
    )

    BASE_URL: str = "https://api.exa.ai"

    def _get_api_key(self) -> str:
        """
        Retrieve the Exa API key from instance variable or environment.

        Returns:
            The API key

        Raises:
            BackendError: If no API key is configured
        """
        key = self.api_key or os.environ.get("EXA_API_KEY")
        if not key:
            raise BackendError("Exa API key not provided")
        return key


    async def search(
        self, query: str, topn: int, session: ClientSession
    ) -> PageContents:
        """
        Search using Exa API and return results as a PageContents object.

        Makes a POST request to /search with the query and converts the
        results into an HTML page with numbered links that the model can open.

        Args:
            query: Search query
            topn: Maximum number of results
            session: aiohttp session

        Returns:
            PageContents with search results as an HTML list of links with summaries
        """
        data = await self._post(
            session,
            "/search",
            {"query": query, "numResults": topn, "contents": {"text": True, "summary": True}},
        )
        # make a simple HTML page to work with browser format
        titles_and_urls = [
            (result["title"], result["url"], result["summary"])
            for result in data["results"]
        ]
        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in titles_and_urls])}
</ul>
</body></html>
"""

        return process_html(
            html=html_page,
            url="",
            title=query,
            display_urls=True,
            session=session,
        )

    async def fetch(self, url: str, session: ClientSession) -> PageContents:
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX) :]
        data = await self._post(
            session,
            "/contents",
            {"urls": [url], "text": { "includeHtmlTags": True }},
        )
        results = data.get("results", [])
        if not results:
            raise BackendError(f"No contents returned for {url}")
        return process_html(
            html=results[0].get("text", ""),
            url=url,
            title=results[0].get("title", ""),
            display_urls=True,
            session=session,
        )

@chz.chz(typecheck=True)
class YouComBackend(Backend):
    """
    Backend implementation using the You.com Search API.

    You.com (https://you.com) provides a search API with:
    - Web search results
    - News search results
    - Snippets and descriptions
    - Live web crawling for content fetching

    Configuration:
        Set YDC_API_KEY environment variable

    API Endpoints:
        - GET /v1/search: Search for web pages and news
        - POST /v1/contents: Fetch page content with live crawling

    Features:
        - Combines web and news results
        - Live HTML crawling for up-to-date content
        - Structured snippets in search results
        - Handles view-source: URLs for raw HTML
    """

    source: str = chz.field(doc="Description of the backend source")

    BASE_URL: str = "https://api.ydc-index.io"

    def _get_api_key(self) -> str:
        """
        Retrieve the You.com API key from environment.

        Returns:
            The API key

        Raises:
            BackendError: If YDC_API_KEY is not set
        """
        key = os.environ.get("YDC_API_KEY")
        if not key:
            raise BackendError("You.com API key not provided")
        return key


    async def search(
        self, query: str, topn: int, session: ClientSession
    ) -> PageContents:
        """
        Search using You.com API and return results as a PageContents object.

        Makes a GET request to /v1/search and combines web and news results
        into an HTML page with numbered links.

        Args:
            query: Search query
            topn: Maximum number of results
            session: aiohttp session

        Returns:
            PageContents with search results (web + news) as an HTML list
        """
        data = await self._get(
            session,
            "/v1/search",
            {"query": query, "count": topn},
        )
        # make a simple HTML page to work with browser format
        web_titles_and_urls, news_titles_and_urls = [], []
        if "web" in data["results"]:
            web_titles_and_urls = [
                (result["title"], result["url"], result["snippets"])
                for result in data["results"]["web"]
            ]
        if "news" in data["results"]:
            news_titles_and_urls = [
                (result["title"], result["url"], result["description"])
                for result in data["results"]["news"]
            ]
        titles_and_urls = web_titles_and_urls + news_titles_and_urls
        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in titles_and_urls])}
</ul>
</body></html>
"""

        return process_html(
            html=html_page,
            url="",
            title=query,
            display_urls=True,
            session=session,
        )

    async def fetch(self, url: str, session: ClientSession) -> PageContents:
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX) :]
        data = await self._post(
            session,
            "/v1/contents",
            {"urls": [url], "livecrawl_formats": "html"},
        )
        if not data:
            raise BackendError(f"No contents returned for {url}")
        if "html" not in data[0]:
            raise BackendError(f"No HTML returned for {url}")
        return process_html(
            html=data[0].get("html", ""),
            url=url,
            title=data[0].get("title", ""),
            display_urls=True,
            session=session,
        )

