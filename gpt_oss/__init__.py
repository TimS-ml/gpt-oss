"""
GPT-OSS: Open Source GPT Model Inference Library

This package provides multi-backend inference capabilities for OpenAI's open-weight
gpt-oss models (gpt-oss-120b and gpt-oss-20b). It supports PyTorch, Triton, Metal,
vLLM, and other backends for flexible deployment across different hardware platforms.

Key Features:
- Multi-backend support (PyTorch, Triton, Metal, vLLM, Ollama, Transformers)
- OpenAI Responses API compatible server implementation
- Tool integration (web browsing, code execution, patching)
- Model evaluation framework
- Production-ready streaming and chat interfaces

Models:
- gpt-oss-120b: 117B total parameters, 5.1B active parameters
- gpt-oss-20b: 21B total parameters, 3.6B active parameters

Both models use Mixture of Experts (MoE) architecture with MXFP4 quantization.
"""
