"""
Streaming event definitions for the Responses API.

When streaming is enabled (stream=true in the request), the API emits a series
of Server-Sent Events (SSE) that allow clients to receive incremental updates
as the response is generated.

Event Flow:
1. response.created - Initial response object created
2. response.in_progress - Generation has started
3. response.output_item.added - New output item started (message, reasoning, tool call)
4. response.content_part.added - New content part started within an item
5. response.output_text.delta - Incremental text chunks (multiple times)
6. response.output_text.annotation.added - Citations added (if web search used)
7. response.output_text.done - Text content complete
8. response.content_part.done - Content part complete
9. response.output_item.done - Output item complete
10. response.completed - Entire response complete

For reasoning: Similar flow but with reasoning_text.delta events
For tools: Specific events like web_search_call.in_progress, code_interpreter_call.interpreting, etc.
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel

from .types import (
    CodeInterpreterCallItem,
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs,
    FunctionCallItem,
    Item,
    ReasoningItem,
    ReasoningTextContentItem,
    ResponseObject,
    TextContentItem,
    UrlCitation,
    WebSearchCallItem,
)


class ResponseEvent(BaseModel):
    """
    Base class for all streaming events.

    Every event includes a sequence number to maintain ordering when
    events arrive asynchronously.
    """
    sequence_number: Optional[int] = 1  # Incrementing number to order events


class ResponseCreatedEvent(ResponseEvent):
    """
    Emitted when a response is first created.

    This is always the first event in a stream. Contains the initial
    (empty) response object with status "in_progress".
    """
    type: Literal["response.created"]
    response: ResponseObject  # Initial response object


class ResponseCompletedEvent(ResponseEvent):
    """
    Emitted when response generation is complete.

    This is always the last event in a successful stream. Contains the
    final complete response with status "completed" and full usage stats.
    """
    type: Literal["response.completed"]
    response: ResponseObject  # Final complete response object


class ResponseOutputTextDelta(ResponseEvent):
    """
    Incremental text chunk from the model's response.

    Emitted multiple times as the model generates text. Clients should
    append these deltas to build the complete message.
    """
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str = "item_1234"  # ID of the message item being generated
    output_index: int = 0  # Index in the output array
    content_index: int = 0  # Index within the item's content array
    delta: str = ""  # The text chunk being added
    logprobs: list = []  # Log probabilities (if requested, not currently supported)


class ResponseReasoningSummaryTextDelta(ResponseEvent):
    """
    Incremental summary text for reasoning content.

    Emitted when generating a summary of the model's reasoning.
    """
    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    item_id: str = "item_1234"  # ID of the reasoning item
    output_index: int = 0
    content_index: int = 0
    delta: str = ""  # Summary text chunk


class ResponseReasoningTextDelta(ResponseEvent):
    """
    Incremental reasoning/thinking text from the model.

    Emitted as the model generates its chain-of-thought reasoning.
    These thoughts are internal and precede the final response.
    """
    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"
    item_id: str = "item_1234"  # ID of the reasoning item
    output_index: int = 0
    content_index: int = 0
    delta: str = ""  # Reasoning text chunk


class ResponseReasoningTextDone(ResponseEvent):
    """
    Emitted when reasoning content is complete.

    Contains the full reasoning text that was generated.
    """
    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"
    item_id: str = "item_1234"  # ID of the reasoning item
    output_index: int = 0
    content_index: int = 0
    text: str = ""  # Complete reasoning text


class ResponseOutputItemAdded(ResponseEvent):
    """
    Emitted when a new output item starts.

    Indicates that the model is beginning to generate a new item
    (message, reasoning, function call, etc.).
    """
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int = 0  # Position in the output array
    item: Union[  # The item being added (initially incomplete)
        Item,
        ReasoningItem,
        FunctionCallItem,
        WebSearchCallItem,
        CodeInterpreterCallItem,
    ]


class ResponseOutputItemDone(ResponseEvent):
    """
    Emitted when an output item is complete.

    Contains the final complete version of the item.
    """
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int = 0  # Position in the output array
    item: Union[  # The completed item
        Item,
        ReasoningItem,
        FunctionCallItem,
        WebSearchCallItem,
        CodeInterpreterCallItem,
    ]


class ResponseInProgressEvent(ResponseEvent):
    """
    Emitted to indicate generation is actively in progress.

    Sent after the initial response.created event.
    """
    type: Literal["response.in_progress"]
    response: ResponseObject  # Current response state


class ResponseContentPartAdded(ResponseEvent):
    """
    Emitted when a new content part starts within an item.

    Messages and reasoning items can have multiple content parts.
    """
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str = "item_1234"  # ID of the parent item
    output_index: int = 0
    content_index: int = 0  # Index of this part within the item
    part: Union[TextContentItem, ReasoningTextContentItem]  # The content part


class ResponseOutputTextDone(ResponseEvent):
    """
    Emitted when text generation for a content part is complete.

    Contains the full generated text.
    """
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str = "item_1234"  # ID of the message item
    output_index: int = 0
    content_index: int = 0
    text: str = ""  # Complete generated text
    logprobs: list = []  # Log probabilities (if requested)


class ResponseContentPartDone(ResponseEvent):
    """
    Emitted when a content part is complete.

    Signals that all text and annotations for this part are finalized.
    """
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str = "item_1234"  # ID of the parent item
    output_index: int = 0
    content_index: int = 0
    part: Union[TextContentItem, ReasoningTextContentItem]  # The complete content part


class ResponseOutputTextAnnotationAdded(ResponseEvent):
    """
    Emitted when a citation/annotation is added to text.

    Used primarily by the web search tool to add URL citations
    to specific parts of the generated text.
    """
    type: Literal["response.output_text.annotation.added"] = (
        "response.output_text.annotation.added"
    )
    item_id: str = "item_1234"  # ID of the message item
    output_index: int = 0
    content_index: int = 0
    annotation_index: int = 0  # Index in the annotations array
    annotation: UrlCitation  # The citation being added


class ResponseWebSearchCallInProgress(ResponseEvent):
    """
    Emitted when a web search call starts.

    Indicates the model has invoked the browser tool and
    the search operation is beginning.
    """
    type: Literal["response.web_search_call.in_progress"] = (
        "response.web_search_call.in_progress"
    )
    output_index: int = 0
    item_id: str = "item_1234"  # ID of the web search call item


class ResponseWebSearchCallSearching(ResponseEvent):
    """
    Emitted when actively performing a web search.

    Indicates the search query is being executed against
    the search backend (Exa, YouCom, etc.).
    """
    type: Literal["response.web_search_call.searching"] = (
        "response.web_search_call.searching"
    )
    output_index: int = 0
    item_id: str = "item_1234"  # ID of the web search call item


class ResponseWebSearchCallCompleted(ResponseEvent):
    """
    Emitted when a web search call completes.

    The search results have been retrieved and added to the context.
    The model will continue generation with this new information.
    """
    type: Literal["response.web_search_call.completed"] = (
        "response.web_search_call.completed"
    )
    output_index: int = 0
    item_id: str = "item_1234"  # ID of the web search call item


class ResponseCodeInterpreterCallInProgress(ResponseEvent):
    """
    Emitted when a code interpreter call starts.

    Indicates the model has written code and code execution
    is about to begin.
    """
    type: Literal["response.code_interpreter_call.in_progress"] = (
        "response.code_interpreter_call.in_progress"
    )
    output_index: int = 0
    item_id: str = "item_1234"  # ID of the code interpreter call item


class ResponseCodeInterpreterCallInterpreting(ResponseEvent):
    """
    Emitted when code is actively executing.

    The Python code is running in the sandboxed environment.
    """
    type: Literal["response.code_interpreter_call.interpreting"] = (
        "response.code_interpreter_call.interpreting"
    )
    output_index: int = 0
    item_id: str = "item_1234"  # ID of the code interpreter call item


class ResponseCodeInterpreterCallCodeDelta(ResponseEvent):
    """
    Incremental code chunks or execution output.

    Emitted as code is being written or as execution output arrives.
    """
    type: Literal["response.code_interpreter_call_code.delta"] = (
        "response.code_interpreter_call_code.delta"
    )
    output_index: int = 0
    item_id: str = "item_1234"  # ID of the code interpreter call item
    delta: str = ""  # Code chunk or output delta
    code_output: Optional[  # Execution output (if any)
        Union[CodeInterpreterOutputLogs, CodeInterpreterOutputImage]
    ] = None


class ResponseCodeInterpreterCallCodeDone(ResponseEvent):
    """
    Emitted when code generation and execution are complete.

    Contains the full code that was executed and all outputs.
    """
    type: Literal["response.code_interpreter_call_code.done"] = (
        "response.code_interpreter_call_code.done"
    )
    output_index: int = 0
    item_id: str = "item_1234"  # ID of the code interpreter call item
    code: str = ""  # Complete code that was executed
    outputs: Optional[  # All execution outputs (logs, images, etc.)
        list[Union[CodeInterpreterOutputLogs, CodeInterpreterOutputImage]]
    ] = None


class ResponseCodeInterpreterCallCompleted(ResponseEvent):
    """
    Emitted when a code interpreter call is fully complete.

    Code has been executed, outputs captured, and the model
    will continue generation with the results.
    """
    type: Literal["response.code_interpreter_call.completed"] = (
        "response.code_interpreter_call.completed"
    )
    output_index: int = 0
    item_id: str = "item_1234"  # ID of the code interpreter call item
