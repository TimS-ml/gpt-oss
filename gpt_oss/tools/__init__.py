"""
Tool System Module for GPT-OSS

This module serves as the entry point for the tool system in GPT-OSS, which provides
the model with capabilities to interact with external systems and execute code.

Tools Overview:
--------------
Tools are modular components that extend the model's capabilities beyond text generation.
They allow the model to:
- Execute Python code (via Docker or local environments)
- Browse the web and search for information
- Apply code patches to files

All tools inherit from the base Tool class (defined in tool.py) and follow the
OpenAI Harmony message format for communication with the model.

Integration with the Model:
---------------------------
Tools are invoked when the model sends a message with a specific recipient field
that matches a tool's name. The tool processes the message and returns response
messages that are added to the conversation history.

This architecture enables:
- Agentic workflows where the model can call tools iteratively
- Stateful interactions (e.g., maintaining browser history)
- Safe execution of potentially dangerous operations (e.g., code in containers)

For implementation details, see the individual tool modules:
- tool.py: Base Tool abstract class
- apply_patch.py: Code patching utilities
- python_docker/: Python execution in Docker containers
- simple_browser/: Web browsing and search capabilities
"""
