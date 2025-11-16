"""
Triton-optimized implementations for GPT model components.

This package contains high-performance Triton kernel implementations for:
- Flash Attention with support for learned sinks and banded attention
- Mixture of Experts (MoE) layers with MX4 quantization
- Complete transformer model using Triton-accelerated operations

Triton is used throughout this package to achieve CUDA-level performance
while maintaining Python-like programmability. The implementations provide
significant speedups over standard PyTorch operations, especially for
attention mechanisms and sparse expert routing in MoE layers.
"""
