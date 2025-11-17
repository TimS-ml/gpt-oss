"""
Page Contents Processing for Simple Browser Tool

This module handles the conversion of raw HTML into model-readable text format.
It's responsible for extracting content, processing links, handling images,
and creating the special citation format that the model uses.

Key Transformations:
--------------------
1. HTML → Clean Text:
   - Removes scripts, styles, and other non-content elements
   - Converts HTML to markdown-like plaintext
   - Preserves structure while being token-efficient

2. Link Processing:
   - Extracts all <a href="..."> tags
   - Replaces them with numbered markers: 【0†Link Text†domain.com】
   - Creates a mapping from numbers to URLs
   - Handles relative URLs by converting to absolute

3. Image Handling:
   - Replaces <img> tags with placeholders: [Image 0: alt text]
   - Preserves alt text for context

4. Special Character Handling:
   - Replaces certain characters that might confuse the model
   - Removes zero-width spaces and other invisible characters
   - Handles math symbols and subscripts/superscripts

Citation Format:
----------------
Links in HTML are converted to a special bracket notation that enables
citations. For example:

    <a href="https://example.com/page">Read more</a>

Becomes:

    【0†Read more†example.com】

Where:
- 0 is the link ID
- "Read more" is the visible text
- "example.com" is the domain (only shown for external links)

This format allows the model to cite sources in its responses using the same
notation, creating traceable attributions.

Classes:
--------
- Extract: Represents a search result snippet or quotable text section
- FetchResult: Result of fetching a URL (success/failure with metadata)
- PageContents: Final processed representation of a web page

Functions:
----------
- process_html(): Main entry point for HTML → PageContents conversion
- html_to_text(): Converts HTML to clean plaintext
- _clean_links(): Processes all links and generates the citation format
- replace_images(): Handles image tag replacements
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
from urllib.parse import urljoin, urlparse

import aiohttp
import html2text
import lxml
import lxml.etree
import lxml.html
import pydantic

import tiktoken

logger = logging.getLogger(__name__)


HTML_SUP_RE = re.compile(r"<sup( [^>]*)?>([\w\-]+)</sup>")
HTML_SUB_RE = re.compile(r"<sub( [^>]*)?>([\w\-]+)</sub>")
HTML_TAGS_SEQ_RE = re.compile(r"(?<=\w)((<[^>]*>)+)(?=\w)")
WHITESPACE_ANCHOR_RE = re.compile(r"(【\@[^】]+】)(\s+)")
EMPTY_LINE_RE = re.compile(r"^\s+$", flags=re.MULTILINE)
EXTRA_NEWLINE_RE = re.compile(r"\n(\s*\n)+")


class Extract(pydantic.BaseModel):
    """
    Represents a snippet of text from a web page (e.g., search result or find result).

    Extracts are used when displaying search results or find-in-page results.
    They contain a small piece of text along with metadata about where it came from.

    Attributes:
        url: The URL where this text was found
        text: The extracted text snippet
        title: A title for this extract (e.g., "#0" for find results)
        line_idx: Optional line number where this text appears in the page
    """
    url: str
    text: str
    title: str
    line_idx: int | None = None


class FetchResult(pydantic.BaseModel):
    """
    Result of attempting to fetch a web page.

    This is an intermediate representation that captures success/failure
    along with the retrieved content or error information.

    Attributes:
        url: The URL that was fetched
        success: Whether the fetch succeeded
        title: Page title (if available)
        error_type: Type of error if fetch failed
        error_message: Error details if fetch failed
        html: Raw HTML if successful
        raw_content: Raw bytes if content isn't HTML
        plaintext: Extracted text content
    """
    url: str
    success: bool
    title: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    html: str | None = None
    raw_content: bytes | None = None
    plaintext: str | None = None


class PageContents(pydantic.BaseModel):
    """
    Processed representation of a web page ready for the model.

    This is the final output of HTML processing, containing clean text
    with numbered link markers and metadata.

    Attributes:
        url: The page URL
        text: Processed text content with link markers
        title: Page title
        urls: Mapping from link IDs (as strings) to URLs
        snippets: Optional mapping from IDs to Extract objects (for search/find results)
        error_message: Error information if page couldn't be fully processed

    The text field contains special markers like 【0†Link Text†domain.com】 that
    reference entries in the urls dict. This enables citation tracking.
    """
    url: str
    text: str
    title: str
    urls: dict[str, str]
    snippets: dict[str, Extract] | None = None
    error_message: str | None = None


@dataclasses.dataclass(frozen=True)
class Tokens:
    tokens: list[int]
    tok2idx: list[int]  # Offsets = running sum of lengths.


def get_domain(url: str) -> str:
    """Extracts the domain from a URL."""
    if "http" not in url:
        # If `get_domain` is called on a domain, add a scheme so that the
        # original domain is returned instead of the empty string.
        url = "http://" + url
    return urlparse(url).netloc


def multiple_replace(text: str, replacements: dict[str, str]) -> str:
    """Performs multiple string replacements using regex pass."""
    regex = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))
    return regex.sub(lambda mo: replacements[mo.group(1)], text)


@functools.lru_cache(maxsize=1024)
def mark_lines(text: str) -> str:
    """Adds line numbers (ex: 'L0:') to the beginning of each line in a string."""
    # Split the string by newline characters
    lines = text.split("\n")

    # Add lines numbers to each line and join into a single string
    numbered_text = "\n".join([f"L{i}: {line}" for i, line in enumerate(lines)])
    return numbered_text


@functools.cache
def _tiktoken_vocabulary_lengths(enc_name: str) -> list[int]:
    """Gets the character lengths of all tokens in the specified TikToken vocabulary."""
    encoding = tiktoken.get_encoding(enc_name)
    return [len(encoding.decode([i])) for i in range(encoding.n_vocab)]


def warmup_caches(enc_names: list[str]) -> None:
    """Warm up the cache by computing token length lists for the given TikToken encodings."""
    for _ in map(_tiktoken_vocabulary_lengths, enc_names):
        pass


def _replace_special_chars(text: str) -> str:
    """Replaces specific special characters with visually similar alternatives."""
    replacements = {
        "【": "〖",
        "】": "〗",
        "◼": "◾",
        # "━": "─",
        "\u200b": "",  # zero width space
        # Note: not replacing †
    }
    return multiple_replace(text, replacements)


def merge_whitespace(text: str) -> str:
    """Replace newlines with spaces and merge consecutive whitespace into a single space."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def arxiv_to_ar5iv(url: str) -> str:
    """Converts an arxiv.org URL to its ar5iv.org equivalent."""
    return re.sub(r"arxiv.org", r"ar5iv.org", url)


def _clean_links(root: lxml.html.HtmlElement, cur_url: str) -> dict[str, str]:
    """Processes all anchor tags in the HTML, replaces them with a custom format and returns an ID-to-URL mapping."""
    cur_domain = get_domain(cur_url)
    urls: dict[str, str] = {}
    urls_rev: dict[str, str] = {}
    for a in root.findall(".//a[@href]"):
        assert a.getparent() is not None
        link = a.attrib["href"]
        if link.startswith(("mailto:", "javascript:")):
            continue
        text = _get_text(a).replace("†", "‡")
        if not re.sub(r"【\@([^】]+)】", "", text):  # Probably an image
            continue
        if link.startswith("#"):
            replace_node_with_text(a, text)
            continue
        try:
            link = urljoin(cur_url, link)  # works with both absolute and relative links
            domain = get_domain(link)
        except Exception:
            domain = ""
        if not domain:
            logger.debug("SKIPPING LINK WITH URL %s", link)
            continue
        link = arxiv_to_ar5iv(link)
        if (link_id := urls_rev.get(link)) is None:
            link_id = f"{len(urls)}"
            urls[link_id] = link
            urls_rev[link] = link_id
        if domain == cur_domain:
            replacement = f"【{link_id}†{text}】"
        else:
            replacement = f"【{link_id}†{text}†{domain}】"
        replace_node_with_text(a, replacement)
    return urls


def _get_text(node: lxml.html.HtmlElement) -> str:
    """Extracts all text from an HTML element and merges it into a whitespace-normalized string."""
    return merge_whitespace(" ".join(node.itertext()))


def _remove_node(node: lxml.html.HtmlElement) -> None:
    """Removes a node from its parent in the lxml tree."""
    node.getparent().remove(node)


def _escape_md(text: str) -> str:
    return text


def _escape_md_section(text: str, snob: bool = False) -> str:
    return text


def html_to_text(html: str) -> str:
    """Converts an HTML string to clean plaintext."""
    html = re.sub(HTML_SUP_RE, r"^{\2}", html)
    html = re.sub(HTML_SUB_RE, r"_{\2}", html)
    # add spaces between tags such as table cells
    html = re.sub(HTML_TAGS_SEQ_RE, r" \1", html)
    # we don't need to escape markdown, so monkey-patch the logic
    orig_escape_md = html2text.utils.escape_md
    orig_escape_md_section = html2text.utils.escape_md_section
    html2text.utils.escape_md = _escape_md
    html2text.utils.escape_md_section = _escape_md_section
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0  # no wrapping
    h.ignore_tables = True
    h.unicode_snob = True
    h.ignore_emphasis = True
    result = h.handle(html).strip()
    html2text.utils.escape_md = orig_escape_md
    html2text.utils.escape_md_section = orig_escape_md_section
    return result


def _remove_math(root: lxml.html.HtmlElement) -> None:
    """Removes all <math> elements from the lxml tree."""
    for node in root.findall(".//math"):
        _remove_node(node)


def remove_unicode_smp(text: str) -> str:
    """Removes Unicode characters in the Supplemental Multilingual Plane (SMP) from `text`.

    SMP characters are not supported by lxml.html processing.
    """
    smp_pattern = re.compile(r"[\U00010000-\U0001FFFF]", re.UNICODE)
    return smp_pattern.sub("", text)


def replace_node_with_text(node: lxml.html.HtmlElement, text: str) -> None:
    """Replaces an lxml node with a text string while preserving surrounding text."""
    previous = node.getprevious()
    parent = node.getparent()
    tail = node.tail or ""
    if previous is None:
        parent.text = (parent.text or "") + text + tail
    else:
        previous.tail = (previous.tail or "") + text + tail
    parent.remove(node)


def replace_images(
    root: lxml.html.HtmlElement,
    base_url: str,
    session: aiohttp.ClientSession | None,
) -> None:
    """Finds all image tags and replaces them with numbered placeholders (includes alt/title if available)."""
    cnt = 0
    for img_tag in root.findall(".//img"):
        image_name = img_tag.get("alt", img_tag.get("title"))
        if image_name:
            replacement = f"[Image {cnt}: {image_name}]"
        else:
            replacement = f"[Image {cnt}]"
        replace_node_with_text(img_tag, replacement)
        cnt += 1


def process_html(
    html: str,
    url: str,
    title: str | None,
    session: aiohttp.ClientSession | None = None,
    display_urls: bool = False,
) -> PageContents:
    """
    Convert raw HTML into a model-readable PageContents object.

    This is the main entry point for HTML processing. It orchestrates all the
    transformation steps to convert HTML into clean, citation-friendly text.

    Processing Pipeline:
    1. Remove problematic Unicode characters (Supplementary Multilingual Plane)
    2. Replace special characters that might confuse the model
    3. Parse HTML into an lxml tree
    4. Extract and process all links with citation markers
    5. Replace images with placeholders
    6. Remove math elements (not well supported)
    7. Convert to plaintext while preserving structure
    8. Clean up whitespace and formatting

    Args:
        html: Raw HTML string
        url: The page URL (used for resolving relative links)
        title: Optional explicit title (otherwise extracted from <title> tag)
        session: Optional aiohttp session (for potential future use)
        display_urls: Whether to show the URL at the top of the text

    Returns:
        PageContents object with:
        - Processed text with citation markers
        - Mapping of link IDs to URLs
        - Page title and metadata

    Example:
        >>> html = '<a href="/page">Click here</a>'
        >>> result = process_html(html, "https://example.com", None)
        >>> print(result.text)
        【0†Click here】
        >>> print(result.urls)
        {'0': 'https://example.com/page'}
    """
    # Remove problematic Unicode characters
    html = remove_unicode_smp(html)
    # Replace special characters that might interfere with citation markers
    html = _replace_special_chars(html)
    # Parse into lxml tree for manipulation
    root = lxml.html.fromstring(html)

    # Extract or construct the page title
    title_element = root.find(".//title")
    if title:
        final_title = title
    elif title_element is not None:
        final_title = title_element.text or ""
    elif url and (domain := get_domain(url)):
        final_title = domain  # Fallback to domain name
    else:
        final_title = ""

    # Process all links and create citation markers
    urls = _clean_links(root, url)

    # Replace images with placeholders
    replace_images(
        root=root,
        base_url=url,
        session=session,
    )

    # Remove math elements (they don't convert well to text)
    _remove_math(root)

    # Convert the modified HTML tree to clean plaintext
    clean_html = lxml.etree.tostring(root, encoding="UTF-8").decode()
    text = html_to_text(clean_html)

    # Post-processing: clean up whitespace and formatting
    text = re.sub(WHITESPACE_ANCHOR_RE, lambda m: m.group(2) + m.group(1), text)
    # ^^^ Move citation markers to the right through whitespace
    # This prevents markers from creating extra spaces

    text = re.sub(EMPTY_LINE_RE, "", text)
    # ^^^ Remove lines that are only whitespace

    text = re.sub(EXTRA_NEWLINE_RE, "\n\n", text)
    # ^^^ Collapse multiple newlines to at most two (one blank line)

    # Optionally add URL header
    top_parts = []
    if display_urls:
        top_parts.append(f"\nURL: {url}\n")
    # NOTE: Publication date is currently not extracted due to performance costs.

    return PageContents(
        url=url,
        text="".join(top_parts) + text,
        urls=urls,
        title=final_title,
    )
