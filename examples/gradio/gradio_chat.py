"""
Gradio Chat Interface for GPT-OSS

This module provides a web-based chat interface using Gradio for interacting with
the gpt-oss model. It supports custom functions, browser search, reasoning modes,
and real-time streaming responses.

Dependencies:
    - gradio: Web UI framework for ML demos
    - requests: HTTP library for API calls
    - json: JSON parsing and formatting

Setup:
    1. Install dependencies: pip install gradio requests
    2. Start the gpt-oss server:
       - Large model: http://localhost:8000/v1/responses
       - Small model: http://localhost:8081/v1/responses
    3. Run this script: python gradio_chat.py
    4. Open the provided URL in your browser (typically http://localhost:7860)

Features:
    - Model selection (large/small)
    - Custom system instructions
    - Reasoning effort control (low/medium/high)
    - Custom function calling with JSON schema
    - Browser search integration
    - Temperature and token limit controls
    - Debug mode for inspecting API responses
    - Real-time streaming of responses

Integration with gpt-oss:
    This interface communicates with the gpt-oss /v1/responses endpoint using the
    OpenAI-compatible format with extended features for reasoning and tools.
"""

import json
import requests
import gradio as gr

# Default JSON schema for function parameters example
# This demonstrates how to define a function that takes a location parameter
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

def chat_with_model(message, history, model_choice, instructions, effort, use_functions,
                   function_name, function_description, function_parameters,
                   use_browser_search, temperature, max_output_tokens, debug_mode):
    """
    Main chat function that handles user messages and streams responses from gpt-oss.

    This function:
    1. Converts Gradio chat history to gpt-oss message format
    2. Configures tools (functions, browser search) based on user settings
    3. Makes a streaming POST request to the gpt-oss API
    4. Processes Server-Sent Events (SSE) to display responses in real-time
    5. Handles reasoning, text output, function calls, and web searches

    Args:
        message (str): The current user message
        history (list): Gradio chat history as [[user_msg, bot_msg], ...]
        model_choice (str): "large" or "small" model selection
        instructions (str): System instructions for the model
        effort (str): Reasoning effort level ("low", "medium", "high")
        use_functions (bool): Whether to enable custom function calling
        function_name (str): Name of the custom function
        function_description (str): Description of what the function does
        function_parameters (str): JSON schema for function parameters
        use_browser_search (bool): Whether to enable browser search tool
        temperature (float): Sampling temperature (0.0-1.0)
        max_output_tokens (int): Maximum tokens in response
        debug_mode (bool): Whether to show debug information

    Returns:
        tuple: (updated_history, empty_string) - Updates chat display and clears input
    """

    if not message.strip():
        return history, ""

    # Append user message and empty assistant placeholder (idiomatic Gradio pattern)
    history = history + [[message, ""]]

    # Build messages list from history (excluding the empty assistant placeholder)
    messages = []

    # Convert history to messages format (excluding the last empty assistant message)
    for user_msg, assistant_msg in history[:-1]:
        if user_msg:
            messages.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_msg}]
            })
        if assistant_msg:
            messages.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_msg}]
            })

    # Add current user message
    messages.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": message}]
    })
    
    # Prepare tools array based on user configuration
    tools = []

    # Add custom function if enabled
    if use_functions:
        try:
            tools.append({
                "type": "function",
                "name": function_name,
                "description": function_description,
                "parameters": json.loads(function_parameters),
            })
        except json.JSONDecodeError:
            # Silently skip if JSON parameters are invalid
            pass

    # Add browser search tool if enabled
    if use_browser_search:
        tools.append({"type": "browser_search"})

    # Select API endpoint based on model choice
    # Small model runs on port 8081, large model on port 8000
    options = ["large", "small"]
    URL = ("http://localhost:8081/v1/responses" if model_choice == options[1]
           else "http://localhost:8000/v1/responses")
    
    try:
        # Make streaming POST request to gpt-oss API
        response = requests.post(
            URL,
            json={
                "input": messages,  # Conversation history
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
        full_content = ""  # Accumulated response text
        text_delta = ""  # Current delta text
        current_output_index = 0  # Track which output we're processing
        in_reasoning = False  # Track if we're in reasoning mode

        # Process Server-Sent Events (SSE) stream
        for line in response.iter_lines(decode_unicode=True):
            # Skip empty lines or lines without the "data:" prefix
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue

            # Parse JSON data from SSE event
            try:
                data = json.loads(data_str)
            except Exception:
                continue

            event_type = data.get("type", "")
            output_index = data.get("output_index", 0)

            # New output item started (reasoning, message, function call, etc.)
            if event_type == "response.output_item.added":
                current_output_index = output_index
                output_type = data.get("item", {}).get("type", "message")
                text_delta = ""
                
                if output_type == "reasoning":
                    if not in_reasoning:
                        full_content += "🤔 **Thinking...**\n"
                        in_reasoning = True
                elif output_type == "message":
                    if in_reasoning:
                        full_content += "\n\n"
                        in_reasoning = False
                
            # Streaming reasoning text (extended thinking)
            elif event_type == "response.reasoning_text.delta":
                delta = data.get("delta", "")
                full_content += delta

                # Update last assistant message with new content (idiomatic Gradio pattern)
                history[-1][1] = full_content
                yield history, ""

            # Streaming output text (the actual response)
            elif event_type == "response.output_text.delta":
                delta = data.get("delta", "")
                full_content += delta

                # Update last assistant message with new content (idiomatic Gradio pattern)
                history[-1][1] = full_content
                yield history, ""

            # Output item completed (function call, web search, etc.)
            elif event_type == "response.output_item.done":
                item = data.get("item", {})

                # Display function call information
                if item.get("type") == "function_call":
                    function_call_text = f"\n\n🔨 Called `{item.get('name')}`\n**Arguments**\n```json\n{item.get('arguments', '')}\n```"
                    full_content += function_call_text

                    # Update last assistant message (idiomatic Gradio pattern)
                    history[-1][1] = full_content
                    yield history, ""

                # Display web search call information
                elif item.get("type") == "web_search_call":
                    web_search_text = f"\n\n🌐 **Web Search**\n```json\n{json.dumps(item.get('action', {}), indent=2)}\n```\n✅ Done"
                    full_content += web_search_text

                    # Update last assistant message (idiomatic Gradio pattern)
                    history[-1][1] = full_content
                    yield history, ""

            # Complete response received
            elif event_type == "response.completed":
                response_data = data.get("response", {})

                # Append debug information if debug mode is enabled
                if debug_mode:
                    debug_info = response_data.get("metadata", {}).get("__debug", "")
                    if debug_info:
                        full_content += f"\n\n**Debug**\n```\n{debug_info}\n```"

                        # Update last assistant message (idiomatic Gradio pattern)
                        history[-1][1] = full_content
                        yield history, ""
                break

        # Return final history and empty string to clear textbox
        return history, ""

    except Exception as e:
        # Display error message in chat
        error_message = f"❌ Error: {str(e)}"
        history[-1][1] = error_message
        return history, ""



# ============================================================================
# Gradio UI Definition
# ============================================================================

# Create the Gradio interface with a Blocks layout
# Blocks allows for more flexible, custom layouts compared to Interface
with gr.Blocks(title="💬 Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot")

    # Main layout: chat on left (75%), controls on right (25%)
    with gr.Row():
        # Left column: Chat interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)

            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)

            clear_btn = gr.Button("Clear Chat")

        # Right column: Configuration controls
        with gr.Column(scale=1):
            # Model selection
            model_choice = gr.Radio(["large", "small"], value="small", label="Model")

            # System instructions (developer message)
            instructions = gr.Textbox(
                label="Instructions",
                value="You are a helpful assistant that can answer questions and help with tasks.",
                lines=3
            )

            # Reasoning effort controls how much extended thinking the model does
            effort = gr.Radio(["low", "medium", "high"], value="medium", label="Reasoning effort")

            # Custom function calling configuration
            gr.Markdown("#### Functions")
            use_functions = gr.Checkbox(label="Use functions", value=False)

            # Function details (shown only when functions are enabled)
            with gr.Column(visible=False) as function_group:
                function_name = gr.Textbox(label="Function name", value="get_weather")
                function_description = gr.Textbox(
                    label="Function description", 
                    value="Get the weather for a given city"
                )
                function_parameters = gr.Textbox(
                    label="Function parameters", 
                    value=DEFAULT_FUNCTION_PROPERTIES,
                    lines=6
                )
            
            # Built-in tools provided by gpt-oss
            gr.Markdown("#### Built-in Tools")
            use_browser_search = gr.Checkbox(label="Use browser search", value=False)

            # Sampling parameters
            temperature = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Temperature")
            max_output_tokens = gr.Slider(1000, 20000, value=1024, step=100, label="Max output tokens")

            # Debug mode shows detailed API response information
            debug_mode = gr.Checkbox(label="Debug mode", value=False)

    # ========================================================================
    # Event handlers
    # ========================================================================

    def toggle_function_group(use_funcs):
        """Show/hide function configuration when checkbox is toggled"""
        return gr.update(visible=use_funcs)
    
    # Connect checkbox to function group visibility
    use_functions.change(toggle_function_group, use_functions, function_group)

    # Collect all inputs for the chat function
    inputs = [msg, chatbot, model_choice, instructions, effort, use_functions,
              function_name, function_description, function_parameters,
              use_browser_search, temperature, max_output_tokens, debug_mode]

    # Connect events to chat function
    msg.submit(chat_with_model, inputs, [chatbot, msg])  # Enter key in textbox
    send_btn.click(chat_with_model, inputs, [chatbot, msg])  # Send button click
    clear_btn.click(lambda: [], outputs=chatbot)  # Clear chat history


if __name__ == "__main__":
    # Launch Gradio web interface
    # Default URL: http://localhost:7860
    # Set share=True to create a public URL
    demo.launch()