"""
Type definitions for the Responses API.

This module contains all Pydantic models used by the Responses API, which provides
a standardized interface for conversational AI interactions. The API supports:
- Text generation with streaming
- Function calling
- Web search integration
- Code interpreter capabilities
- Reasoning/chain-of-thought outputs

The types follow a hierarchical structure:
- Request types: Define what clients send to the API
- Response types: Define what the API returns
- Item types: Individual components within responses (messages, function calls, etc.)
- Content types: Actual content within items (text, citations, code outputs, etc.)
"""

from typing import Any, Dict, Literal, Optional, Union

from openai_harmony import ReasoningEffort
from pydantic import BaseModel, ConfigDict

# Default configuration constants for the Responses API
MODEL_IDENTIFIER = "gpt-oss-120b"  # Default model name returned in responses
DEFAULT_TEMPERATURE = 0.0  # Greedy decoding by default (deterministic)
REASONING_EFFORT = ReasoningEffort.LOW  # Default reasoning effort level
DEFAULT_MAX_OUTPUT_TOKENS = 131072  # Maximum tokens the model can generate


class UrlCitation(BaseModel):
    """
    Represents a citation to a web URL within text content.

    Used by the browser/web search tool to link generated text back to source URLs.
    Citations include the character position in the text where they apply.
    """
    type: Literal["url_citation"]
    end_index: int  # Character position where the citation ends in the text
    start_index: int  # Character position where the citation starts in the text
    url: str  # The full URL being cited
    title: str  # Display title for the citation


class TextContentItem(BaseModel):
    """
    A piece of text content within a message.

    Can represent:
    - input_text: User-provided text
    - output_text: Model-generated text
    - text: Generic text content

    May include annotations like URL citations when web search is used.
    """
    type: Union[Literal["text"], Literal["input_text"], Literal["output_text"]]
    text: str  # The actual text content
    status: Optional[str] = "completed"  # Generation status
    annotations: Optional[list[UrlCitation]] = None  # Citations and other annotations


class SummaryTextContentItem(BaseModel):
    """
    A summary of reasoning content.

    This type exists for API compatibility and represents a condensed
    version of the model's reasoning process.
    """
    type: Literal["summary_text"]
    text: str  # Summary text


class ReasoningTextContentItem(BaseModel):
    """
    Chain-of-thought reasoning content from the model.

    When reasoning is enabled, the model generates internal thought processes
    before producing its final response. This type captures that reasoning.
    """
    type: Literal["reasoning_text"]
    text: str  # The reasoning/thinking text


class ReasoningItem(BaseModel):
    """
    A complete reasoning/chain-of-thought item in the response.

    Contains both the full reasoning content and an optional summary.
    This allows clients to show/hide the detailed reasoning as needed.
    """
    id: str = "rs_1234"  # Unique identifier for this reasoning item
    type: Literal["reasoning"]
    summary: list[SummaryTextContentItem]  # Condensed reasoning summary
    content: Optional[list[ReasoningTextContentItem]] = []  # Full reasoning steps


class Item(BaseModel):
    """
    A message in the conversation.

    Represents a single turn from a user, assistant, or system. Messages
    can contain multiple content items and may be in various states of completion
    when streaming.
    """
    id: Optional[str] = None  # Unique identifier for this message
    type: Optional[Literal["message"]] = "message"
    role: Literal["user", "assistant", "system"]  # Who sent this message
    content: Union[list[TextContentItem], str]  # Message content (structured or simple string)
    status: Union[Literal["in_progress", "completed", "incomplete"], None] = None  # Generation status


class FunctionCallItem(BaseModel):
    """
    Represents a function/tool call made by the assistant.

    When the model decides to use a tool, it generates a function call with
    arguments. The client is expected to execute the function and return
    the result via FunctionCallOutputItem.
    """
    type: Literal["function_call"]
    name: str  # Name of the function being called
    arguments: str  # JSON-formatted function arguments
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    id: str = "fc_1234"  # Unique identifier for this function call
    call_id: str = "call_1234"  # ID used to match with function output


class FunctionCallOutputItem(BaseModel):
    """
    The result of executing a function call.

    After the client executes a function, it sends back the output using this type.
    The call_id links it to the original FunctionCallItem.
    """
    type: Literal["function_call_output"]
    call_id: str = "call_1234"  # Must match the call_id from FunctionCallItem
    output: str  # The result of executing the function (usually JSON)


class WebSearchActionSearch(BaseModel):
    """
    A web search action - searching the web with a query.

    Part of the built-in browser/web search tool.
    """
    type: Literal["search"]
    query: Optional[str] = None  # Search query string


class WebSearchActionOpenPage(BaseModel):
    """
    A web search action - opening a specific URL.

    Part of the built-in browser/web search tool.
    """
    type: Literal["open_page"]
    url: Optional[str] = None  # URL to open


class WebSearchActionFind(BaseModel):
    """
    A web search action - finding text within a page.

    Part of the built-in browser/web search tool.
    """
    type: Literal["find"]
    pattern: Optional[str] = None  # Text pattern to search for
    url: Optional[str] = None  # URL of the page to search within


class WebSearchCallItem(BaseModel):
    """
    Represents a web search/browser tool invocation.

    The built-in browser tool allows the model to search the web, open pages,
    and find content. This item captures one such action and its results.
    """
    type: Literal["web_search_call"]
    id: str = "ws_1234"  # Unique identifier for this search call
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    action: Union[WebSearchActionSearch, WebSearchActionOpenPage, WebSearchActionFind]  # The specific action taken


class CodeInterpreterOutputLogs(BaseModel):
    """
    Text output (stdout/stderr) from code execution.

    Part of the code interpreter tool output.
    """
    type: Literal["logs"]
    logs: str  # The captured output text


class CodeInterpreterOutputImage(BaseModel):
    """
    An image generated by code execution.

    When executed code produces plots or images, they're captured here.
    """
    type: Literal["image"]
    url: str  # URL where the generated image can be accessed


class CodeInterpreterCallItem(BaseModel):
    """
    Represents a code interpreter/execution invocation.

    The built-in code interpreter allows the model to write and execute Python code
    in a sandboxed environment. This captures the code, execution status, and outputs.
    """
    type: Literal["code_interpreter_call"]
    id: str = "ci_1234"  # Unique identifier for this code execution
    status: Literal[
        "in_progress",  # Code execution is starting
        "completed",  # Code executed successfully
        "incomplete",  # Code execution was partial/truncated
        "interpreting",  # Code is currently running
        "failed",  # Code execution failed with an error
    ] = "completed"
    code: Optional[str] = None  # The Python code that was/will be executed
    container_id: Optional[str] = None  # ID of the Docker container (if applicable)
    outputs: Optional[
        list[Union[CodeInterpreterOutputLogs, CodeInterpreterOutputImage]]
    ] = None  # Results from code execution (logs, images, etc.)


class Error(BaseModel):
    """
    Error information when a request fails.

    Contains both a machine-readable error code and human-readable message.
    """
    code: str  # Error code (e.g., "invalid_function_call", "rate_limit_exceeded")
    message: str  # Detailed error message


class IncompleteDetails(BaseModel):
    """
    Details about why a response was incomplete.

    Provides context when a response doesn't fully complete (e.g., hit token limit).
    """
    reason: str  # Explanation of why the response is incomplete


class Usage(BaseModel):
    """
    Token usage statistics for a request.

    Tracks how many tokens were consumed by the request and response,
    useful for monitoring costs and quotas.
    """
    input_tokens: int  # Number of tokens in the input/prompt
    output_tokens: int  # Number of tokens generated in the response
    total_tokens: int  # Sum of input and output tokens


class FunctionToolDefinition(BaseModel):
    """
    Definition of a user-defined function/tool the model can call.

    Functions allow the model to interact with external APIs and services.
    The model generates function calls based on these definitions, and the
    client executes them and returns results.
    """
    type: Literal["function"]
    name: str  # Function name (used in function calls)
    parameters: dict  # JSON Schema describing function parameters
    strict: bool = False  # Whether to enforce strict schema validation (not yet supported)
    description: Optional[str] = ""  # Description helping the model understand when to use this function


class BrowserToolConfig(BaseModel):
    """
    Configuration for the built-in browser/web search tool.

    Enables the model to search the web, open pages, and extract information.
    Uses either "browser_search" or "web_search" type for compatibility.
    """
    model_config = ConfigDict(extra='allow')  # Allow additional fields for flexibility
    type: Literal["browser_search"] | Literal["web_search"]


class CodeInterpreterToolConfig(BaseModel):
    """
    Configuration for the built-in code interpreter tool.

    Enables the model to write and execute Python code in a sandboxed environment.
    """
    type: Literal["code_interpreter"]


class ReasoningConfig(BaseModel):
    """
    Configuration for chain-of-thought reasoning.

    Controls how much reasoning/thinking the model does before responding.
    Higher effort levels produce more thorough reasoning but take longer.
    """
    effort: Literal["low", "medium", "high"] = REASONING_EFFORT  # Reasoning effort level


class ResponsesRequest(BaseModel):
    """
    Main request object for the Responses API.

    This is what clients send to /v1/responses to generate completions.
    Supports both simple string inputs and complex multi-turn conversations
    with function calls, reasoning, and tool usage.

    Example simple request:
        {"input": "What is 2+2?", "model": "gpt-oss-120b"}

    Example with tools:
        {
            "input": "Search for recent AI news",
            "tools": [{"type": "web_search"}],
            "stream": true
        }
    """
    instructions: Optional[str] = None  # System-level instructions for the model
    max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS  # Maximum tokens to generate
    input: Union[
        str,  # Simple string input (single user message)
        list[  # Complex multi-turn conversation with various item types
            Union[
                Item,  # User/assistant messages
                ReasoningItem,  # Chain-of-thought reasoning
                FunctionCallItem,  # Tool/function invocations
                FunctionCallOutputItem,  # Results from function calls
                WebSearchCallItem,  # Web search actions
                CodeInterpreterCallItem,  # Code execution items
            ]
        ],
    ]
    model: Optional[str] = MODEL_IDENTIFIER  # Model to use for generation
    stream: Optional[bool] = False  # Whether to stream the response via SSE
    tools: Optional[  # Available tools the model can use
        list[
            Union[FunctionToolDefinition, BrowserToolConfig, CodeInterpreterToolConfig]
        ]
    ] = []
    reasoning: Optional[ReasoningConfig] = ReasoningConfig()  # Chain-of-thought configuration
    metadata: Optional[Dict[str, Any]] = {}  # Arbitrary metadata (can include __debug flag)
    tool_choice: Optional[Literal["auto", "none"]] = "auto"  # How the model should use tools
    parallel_tool_calls: Optional[bool] = False  # Whether to allow parallel tool invocations
    store: Optional[bool] = False  # Whether to store this conversation for continuation
    previous_response_id: Optional[str] = None  # ID of previous response to continue from
    temperature: Optional[float] = DEFAULT_TEMPERATURE  # Sampling temperature (0 = greedy)
    include: Optional[list[str]] = None  # Additional fields to include in response


class ResponseObject(BaseModel):
    """
    Response object returned by the Responses API.

    Contains the generated output, which may include:
    - Assistant messages
    - Chain-of-thought reasoning
    - Function calls
    - Web search results
    - Code execution results

    Can be in various states (in_progress, completed, failed) when streaming.
    """
    output: list[  # The generated output items
        Union[
            Item,  # Generated messages
            ReasoningItem,  # Chain-of-thought reasoning
            FunctionCallItem,  # Function calls made by the model
            FunctionCallOutputItem,  # Function results (echoed from input)
            WebSearchCallItem,  # Web search actions taken
            CodeInterpreterCallItem,  # Code that was executed
        ]
    ]
    created_at: int  # Unix timestamp when this response was created
    usage: Optional[Usage] = None  # Token usage statistics
    status: Literal["completed", "failed", "incomplete", "in_progress"] = "in_progress"  # Response status
    background: None = None  # Reserved for future use
    error: Optional[Error] = None  # Error details if status is "failed"
    incomplete_details: Optional[IncompleteDetails] = None  # Details if status is "incomplete"
    instructions: Optional[str] = None  # Echo of the instructions from request
    max_output_tokens: Optional[int] = None  # Echo of max_output_tokens from request
    max_tool_calls: Optional[int] = None  # Maximum tool calls allowed
    metadata: Optional[Dict[str, Any]] = {}  # Metadata (may include debug info)
    model: Optional[str] = MODEL_IDENTIFIER  # Model that generated this response
    parallel_tool_calls: Optional[bool] = False  # Echo of parallel_tool_calls from request
    previous_response_id: Optional[str] = None  # ID of previous response this continues from
    id: Optional[str] = "resp_1234"  # Unique identifier for this response
    object: Optional[str] = "response"  # Type discriminator
    text: Optional[Dict[str, Any]] = None  # Text format configuration
    tool_choice: Optional[str] = "auto"  # Echo of tool_choice from request
    top_p: Optional[int] = 1  # Top-p sampling parameter (not currently used)
