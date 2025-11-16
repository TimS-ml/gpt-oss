"""
Responses API server entry point.

This script starts the Responses API server with a specified inference backend.
It handles:
- Command-line argument parsing
- Backend selection and initialization
- FastAPI application creation
- Server startup with uvicorn

Usage:
    # Start with Triton backend (default on Linux)
    python -m gpt_oss.responses_api.serve --checkpoint /path/to/model --port 8000

    # Start with Metal backend (default on macOS)
    python -m gpt_oss.responses_api.serve --checkpoint /path/to/model --inference-backend metal

    # Start with vLLM backend
    python -m gpt_oss.responses_api.serve --checkpoint /path/to/model --inference-backend vllm

    # For multi-GPU with Triton, use torchrun:
    torchrun --nproc-per-node=4 serve.py --checkpoint /path/to/model --inference-backend triton

Available backends:
- triton: Custom Triton kernel implementation (fast, requires CUDA)
- metal: Metal Performance Shaders (macOS only)
- vllm: vLLM library (versatile, supports many models)
- transformers: HuggingFace Transformers (widely compatible)
- ollama: Ollama service (requires Ollama running on localhost:11434)
- stub: Fake backend for testing (returns pre-defined tokens)
"""

import argparse

import uvicorn
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)

from .api_server import create_api_server

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Responses API server")
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the model checkpoint directory",
        default="~/model",
        required=False,
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--inference-backend",
        metavar="BACKEND",
        type=str,
        help="Inference backend to use (triton, metal, vllm, transformers, ollama, stub)",
        # Default to Metal on macOS, Triton on other platforms (Linux/CUDA)
        default="metal" if __import__("platform").system() == "Darwin" else "triton",
    )
    args = parser.parse_args()

    # Dynamically import the setup_model function from the selected backend
    # Each backend module exports a setup_model() function that returns
    # an infer_next_token callable with signature: (tokens, temperature, new_request) -> int
    if args.inference_backend == "triton":
        from .inference.triton import setup_model
    elif args.inference_backend == "stub":
        from .inference.stub import setup_model
    elif args.inference_backend == "metal":
        from .inference.metal import setup_model
    elif args.inference_backend == "ollama":
        from .inference.ollama import setup_model
    elif args.inference_backend == "vllm":
        from .inference.vllm import setup_model
    elif args.inference_backend == "transformers":
        from .inference.transformers import setup_model
    else:
        raise ValueError(f"Invalid inference backend: {args.inference_backend}")

    # Load the Harmony tokenizer/encoder
    # This handles the special message formatting and control tokens
    # used by the gpt-oss model
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Initialize the model and get the inference function
    infer_next_token = setup_model(args.checkpoint)

    # Create the FastAPI application with the inference function and encoding
    app = create_api_server(infer_next_token, encoding)

    # Start the server with uvicorn
    # The server will listen on the specified port and handle requests at /v1/responses
    uvicorn.run(app, port=args.port)
