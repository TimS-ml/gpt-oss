"""
Streamlit Chat Interface for GPT-OSS

This module provides an interactive web-based chat interface using Streamlit for
communicating with the gpt-oss model. It supports advanced features like custom
functions, browser search, code interpreter, reasoning modes, and real-time streaming.

Dependencies:
    - streamlit: Web framework for data apps
    - requests: HTTP library for API calls
    - json: JSON parsing and formatting

Setup:
    1. Install dependencies: pip install streamlit requests
    2. Start the gpt-oss server:
       - Large model: http://localhost:8000/v1/responses
       - Small model: http://localhost:8081/v1/responses
    3. Run this script: streamlit run streamlit_chat.py
    4. Open the provided URL in your browser (typically http://localhost:8501)

Features:
    - Model selection (large/small) with query parameter persistence
    - Custom system instructions
    - Reasoning effort control (low/medium/high)
    - Custom function calling with JSON schema
    - Built-in tools: browser search and code interpreter
    - Temperature and token limit controls
    - Debug mode with API response inspection
    - Real-time streaming responses with SSE
    - Interactive function output submission

Integration with gpt-oss:
    This interface communicates with the gpt-oss /v1/responses endpoint using
    streaming Server-Sent Events (SSE) for real-time updates. It supports the
    full range of gpt-oss features including reasoning, tools, and metadata.
"""

import json

import requests
import streamlit as st

# Default JSON schema for function parameters example
# Demonstrates how to define a function parameter schema
DEFAULT_FUNCTION_PROPERTIES = """
{
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
        }
    },
    "required": ["location"]
}
""".strip()

# ============================================================================
# Session State Initialization
# ============================================================================

# Initialize session state for chat messages
# Streamlit re-runs the script on each interaction, so we use session_state
# to persist data across runs
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 Chatbot")

# ============================================================================
# Model Selection Configuration
# ============================================================================

# Initialize model selection from query params or default to "small"
# Query params allow sharing URLs with specific configurations
if "model" not in st.session_state:
    if "model" in st.query_params:
        st.session_state.model = st.query_params["model"]
    else:
        st.session_state.model = "small"

# Model selection control in sidebar
options = ["large", "small"]
selection = st.sidebar.segmented_control(
    "Model", options, selection_mode="single", default=st.session_state.model
)

# Update query params to persist model selection in URL
st.query_params.update({"model": selection})

# ============================================================================
# Sidebar Configuration Controls
# ============================================================================

# System instructions (developer message)
instructions = st.sidebar.text_area(
    "Instructions",
    value="You are a helpful assistant that can answer questions and help with tasks.",
)

# Reasoning effort controls how much extended thinking the model does
effort = st.sidebar.radio(
    "Reasoning effort",
    ["low", "medium", "high"],
    index=1,  # Default to "medium"
)

st.sidebar.divider()

# Custom function calling configuration
st.sidebar.subheader("Functions")
use_functions = st.sidebar.toggle("Use functions", value=False)

st.sidebar.subheader("Built-in Tools")

# Built-in tools provided by gpt-oss
use_browser_search = st.sidebar.toggle("Use browser search", value=False)
use_code_interpreter = st.sidebar.toggle("Use code interpreter", value=False)

# Show function configuration controls only when functions are enabled
if use_functions:
    function_name = st.sidebar.text_input("Function name", value="get_weather")
    function_description = st.sidebar.text_area(
        "Function description", value="Get the weather for a given city"
    )
    function_parameters = st.sidebar.text_area(
        "Function parameters", value=DEFAULT_FUNCTION_PROPERTIES
    )
else:
    # Set to None when functions are disabled
    function_name = None
    function_description = None
    function_parameters = None

st.sidebar.divider()

# Sampling parameters
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.01
)
max_output_tokens = st.sidebar.slider(
    "Max output tokens", min_value=1, max_value=131072, value=30000, step=1000
)

st.sidebar.divider()

# Debug mode shows detailed API response information
debug_mode = st.sidebar.toggle("Debug mode", value=False)

# Display current message state in debug mode
if debug_mode:
    st.sidebar.divider()
    st.sidebar.code(json.dumps(st.session_state.messages, indent=2), "json")

# Flag to control whether to show the input field
render_input = True

# Select API endpoint based on model choice
# Small model runs on port 8081, large model on port 8000
URL = (
    "http://localhost:8081/v1/responses"
    if selection == options[1]
    else "http://localhost:8000/v1/responses"
)


# ============================================================================
# Helper Functions
# ============================================================================

def trigger_fake_tool(container):
    """
    Handle function call output submission.

    When the model makes a function call, this function is triggered to:
    1. Get the user-provided function output
    2. Add it to the message history as a function_call_output
    3. Continue the conversation by calling run() again

    Args:
        container: Streamlit container for rendering responses
    """
    function_output = st.session_state.get("function_output", "It's sunny!")
    last_call = st.session_state.messages[-1]

    # Only process if the last message was a function call
    if last_call.get("type") == "function_call":
        st.session_state.messages.append(
            {
                "type": "function_call_output",
                "call_id": last_call.get("call_id"),
                "output": function_output,
            }
        )
        # Continue the conversation with the function output
        run(container)


def run(container):
    """
    Main function to run the conversation and stream responses from gpt-oss.

    This function:
    1. Prepares the tools configuration (functions, browser, code interpreter)
    2. Makes a streaming POST request to the gpt-oss API
    3. Processes Server-Sent Events (SSE) to display responses in real-time
    4. Handles various output types: messages, reasoning, function calls,
       web searches, and code interpreter execution

    Args:
        container: Streamlit container for rendering responses
    """

    # Build tools array based on user configuration
    tools = []

    # Add custom function if enabled
    if use_functions:
        tools.append(
            {
                "type": "function",
                "name": function_name,
                "description": function_description,
                "parameters": json.loads(function_parameters),
            }
        )

    # Add browser search tool if enabled
    if use_browser_search:
        tools.append({"type": "browser_search"})

    # Add code interpreter tool if enabled
    if use_code_interpreter:
        tools.append({"type": "code_interpreter"})

    # Make streaming POST request to gpt-oss API
    response = requests.post(
        URL,
        json={
            "input": st.session_state.messages,  # Conversation history
            "stream": True,  # Enable streaming responses
            "instructions": instructions,  # System instructions
            "reasoning": {"effort": effort},  # Reasoning effort level
            "metadata": {"__debug": debug_mode},  # Debug metadata
            "tools": tools,  # Available tools for the model
            "temperature": temperature,  # Sampling temperature
            "max_output_tokens": max_output_tokens,  # Token limit
        },
        stream=True,
    )

    # State tracking for streaming response
    text_delta = ""  # Accumulated text for current item
    code_interpreter_sessions: dict[str, dict] = {}  # Track code interpreter state

    _current_output_index = 0  # Track which output we're processing

    # Process Server-Sent Events (SSE) stream
    for line in response.iter_lines(decode_unicode=True):
        # Skip empty lines or lines without the "data:" prefix
        if not line or not line.startswith("data:"):
            continue
        data_str = line[len("data:") :].strip()
        if not data_str:
            continue

        # Parse JSON data from SSE event
        try:
            data = json.loads(data_str)
        except Exception:
            continue

        event_type = data.get("type", "")
        output_index = data.get("output_index", 0)

        # New output item started (message, reasoning, function call, etc.)
        if event_type == "response.output_item.added":
            _current_output_index = output_index
            output_type = data.get("item", {}).get("type", "message")

            # Regular assistant message
            if output_type == "message":
                output = container.chat_message("assistant")
                placeholder = output.empty()

            # Reasoning/thinking output
            elif output_type == "reasoning":
                output = container.chat_message("reasoning", avatar="🤔")
                placeholder = output.empty()

            # Web search tool call
            elif output_type == "web_search_call":
                output = container.chat_message("web_search_call", avatar="🌐")
                output.code(
                    json.dumps(data.get("item", {}).get("action", {}), indent=4),
                    language="json",
                )
                placeholder = output.empty()

            # Code interpreter tool call
            elif output_type == "code_interpreter_call":
                item = data.get("item", {})
                item_id = item.get("id")

                # Create a message container for code interpreter output
                message_container = container.chat_message(
                    "code_interpreter_call", avatar="🧪"
                )
                status_placeholder = message_container.empty()
                code_placeholder = message_container.empty()
                outputs_container = message_container.container()

                # Display initial code if available
                code_text = item.get("code") or ""
                if code_text:
                    code_placeholder.code(code_text, language="python")

                # Store session info for updating later
                code_interpreter_sessions[item_id] = {
                    "status": status_placeholder,
                    "code": code_placeholder,
                    "outputs": outputs_container,
                    "code_text": code_text,
                    "rendered_outputs": False,
                }
                placeholder = status_placeholder

            text_delta = ""  # Reset text accumulator

        # Streaming reasoning text (extended thinking)
        elif event_type == "response.reasoning_text.delta":
            output.avatar = "🤔"
            text_delta += data.get("delta", "")
            placeholder.markdown(text_delta)

        # Streaming output text (the actual response)
        elif event_type == "response.output_text.delta":
            text_delta += data.get("delta", "")
            placeholder.markdown(text_delta)

        # Output item completed
        elif event_type == "response.output_item.done":
            item = data.get("item", {})

            # Display completed function call
            if item.get("type") == "function_call":
                with container.chat_message("function_call", avatar="🔨"):
                    st.markdown(f"Called `{item.get('name')}`")
                    st.caption("Arguments")
                    st.code(item.get("arguments", ""), language="json")

            # Mark web search as completed
            if item.get("type") == "web_search_call":
                placeholder.markdown("✅ Done")

            # Process completed code interpreter call with outputs
            if item.get("type") == "code_interpreter_call":
                item_id = item.get("id")
                session = code_interpreter_sessions.get(item_id)

                if session:
                    # Mark as completed
                    session["status"].markdown("✅ Done")

                    # Update with final code
                    final_code = item.get("code") or session["code_text"]
                    if final_code:
                        session["code"].code(final_code, language="python")
                        session["code_text"] = final_code

                    # Render outputs (logs, images, etc.)
                    outputs = item.get("outputs") or []
                    if outputs and not session["rendered_outputs"]:
                        with session["outputs"]:
                            st.markdown("**Outputs**")
                            for output_item in outputs:
                                output_type = output_item.get("type")
                                if output_type == "logs":
                                    st.code(
                                        output_item.get("logs", ""),
                                        language="text",
                                    )
                                elif output_type == "image":
                                    st.image(
                                        output_item.get("url", ""),
                                        caption="Code interpreter image",
                                    )
                        session["rendered_outputs"] = True
                    elif not outputs and not session["rendered_outputs"]:
                        with session["outputs"]:
                            st.caption("(No outputs)")
                        session["rendered_outputs"] = True
                else:
                    # Fallback if session not found
                    placeholder.markdown("✅ Done")

        # Code interpreter status events
        elif event_type == "response.code_interpreter_call.in_progress":
            item_id = data.get("item_id")
            session = code_interpreter_sessions.get(item_id)
            if session:
                session["status"].markdown("⏳ Running")
            else:
                try:
                    placeholder.markdown("⏳ Running")
                except Exception:
                    pass

        elif event_type == "response.code_interpreter_call.interpreting":
            item_id = data.get("item_id")
            session = code_interpreter_sessions.get(item_id)
            if session:
                session["status"].markdown("🧮 Interpreting")

        elif event_type == "response.code_interpreter_call.completed":
            item_id = data.get("item_id")
            session = code_interpreter_sessions.get(item_id)
            if session:
                session["status"].markdown("✅ Done")
            else:
                try:
                    placeholder.markdown("✅ Done")
                except Exception:
                    pass

        # Streaming code updates for code interpreter
        elif event_type == "response.code_interpreter_call_code.delta":
            item_id = data.get("item_id")
            session = code_interpreter_sessions.get(item_id)
            if session:
                session["code_text"] += data.get("delta", "")
                if session["code_text"].strip():
                    session["code"].code(session["code_text"], language="python")

        elif event_type == "response.code_interpreter_call_code.done":
            item_id = data.get("item_id")
            session = code_interpreter_sessions.get(item_id)
            if session:
                final_code = data.get("code") or session["code_text"]
                session["code_text"] = final_code
                if final_code:
                    session["code"].code(final_code, language="python")

        # Complete response received
        elif event_type == "response.completed":
            response = data.get("response", {})

            # Display debug information if debug mode is enabled
            if debug_mode:
                container.expander("Debug", expanded=False).code(
                    response.get("metadata", {}).get("__debug", ""), language="text"
                )

            # Add response outputs to message history
            st.session_state.messages.extend(response.get("output", []))

            # If the last message was a function call, show a form to submit output
            if st.session_state.messages[-1].get("type") == "function_call":
                with container.form("function_output_form"):
                    _function_output = st.text_input(
                        "Enter function output",
                        value=st.session_state.get("function_output", "It's sunny!"),
                        key="function_output",
                    )
                    st.form_submit_button(
                        "Submit function output",
                        on_click=trigger_fake_tool,
                        args=[container],
                    )



# ============================================================================
# Chat Display and Message Rendering
# ============================================================================

# Display all messages in the conversation history
for msg in st.session_state.messages:
    # Regular message (user or assistant)
    if msg.get("type") == "message":
        with st.chat_message(msg["role"]):
            for item in msg["content"]:
                # Display text content
                if (
                    item.get("type") == "text"
                    or item.get("type") == "output_text"
                    or item.get("type") == "input_text"
                ):
                    st.markdown(item["text"])

                    # Display annotations (citations, sources)
                    if item.get("annotations"):
                        annotation_lines = "\n".join(
                            f"- {annotation.get('url')}"
                            for annotation in item["annotations"]
                            if annotation.get("url")
                        )
                        st.caption(f"**Annotations:**\n{annotation_lines}")

    # Reasoning/thinking output
    elif msg.get("type") == "reasoning":
        with st.chat_message("reasoning", avatar="🤔"):
            for item in msg["content"]:
                if item.get("type") == "reasoning_text":
                    st.markdown(item["text"])

    # Function call
    elif msg.get("type") == "function_call":
        with st.chat_message("function_call", avatar="🔨"):
            st.markdown(f"Called `{msg.get('name')}`")
            st.caption("Arguments")
            st.code(msg.get("arguments", ""), language="json")

    # Function call output
    elif msg.get("type") == "function_call_output":
        with st.chat_message("function_call_output", avatar="✅"):
            st.caption("Output")
            st.code(msg.get("output", ""), language="text")

    # Web search call
    elif msg.get("type") == "web_search_call":
        with st.chat_message("web_search_call", avatar="🌐"):
            st.code(json.dumps(msg.get("action", {}), indent=4), language="json")
            st.markdown("✅ Done")

    # Code interpreter call (simplified display from history)
    elif msg.get("type") == "code_interpreter_call":
        with st.chat_message("code_interpreter_call", avatar="🧪"):
            st.markdown("✅ Done")


# ============================================================================
# User Input and Conversation Initiation
# ============================================================================

if render_input:
    # Chat input field at the bottom of the page
    if prompt := st.chat_input("Type a message..."):
        # Add user message to history
        st.session_state.messages.append(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the conversation and get assistant response
        run(st.container())
