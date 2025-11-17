"""
Metal Inference Module for Apple Silicon

This module provides Python bindings to the gpt-oss Metal inference backend,
which is optimized for Apple Silicon GPUs (M1, M2, M3 series chips).

The Metal backend leverages Apple's Metal Performance Shaders (MPS) framework
for high-performance GPU acceleration on macOS. Key features include:
- Native Metal compute shaders for transformer operations
- Optimized memory management for Apple's unified memory architecture
- Low-precision inference (FP16, MXFP4) for faster inference
- Custom kernels for attention, MoE routing, and matrix operations

The compiled C++ extension (_metal) is dynamically loaded and its public
symbols are exposed at the module level for convenient access.

Usage:
    from gpt_oss.metal import Context, Model
    model = Model("path/to/model")
    context = Context(model)
    context.append("Hello world")
    token = context.sample()
"""

from importlib import import_module as _im

# Dynamically load the compiled Metal extension (gpt_oss.metal._metal)
# This extension is implemented in C++ and provides GPU-accelerated inference
# using Apple's Metal framework for optimal performance on Apple Silicon
_ext = _im(f"{__name__}._metal")

# Export all public symbols from the extension to the module namespace
# This makes classes like Model, Context, etc. directly accessible
# Only non-private symbols (not starting with '_') are exported
globals().update({k: v for k, v in _ext.__dict__.items() if not k.startswith("_")})

# Clean up temporary import references to keep module namespace clean
del _im, _ext
