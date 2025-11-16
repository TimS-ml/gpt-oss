"""
Mixture of Experts (MoE) implementation with MX4 quantization and Triton kernels.

This module implements a high-performance MoE layer that combines:
1. MX4 (Microscaling 4-bit) quantization for memory-efficient weight storage
2. Sparse expert routing with top-k selection
3. SwiGLU activation function (Swish-Gated Linear Unit)
4. Triton-optimized matrix multiplications using the OGS (Optimized GEMM for Sparse) kernel

Key optimizations:
- MX4 quantization reduces memory footprint by 4x while maintaining accuracy
- Sparse routing only activates k experts per token (typically 4 out of 128)
- Fused activation eliminates intermediate materialization of large tensors
- Triton kernels provide CUDA-level performance for all operations

The MoE layer follows this computation flow:
1. Gate network selects top-k experts per token
2. Tokens are routed to their selected experts
3. Each expert applies: x -> SwiGLU(W1 @ x) -> W2 @ x
4. Expert outputs are combined with learned routing weights
"""

import torch
from torch.profiler import record_function

import triton_kernels
import triton_kernels.swiglu
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import routing
from triton_kernels.tensor import convert_layout
from triton_kernels.tensor_details.layout import StridedLayout, HopperMXScaleLayout, HopperMXValueLayout
from triton_kernels.tensor import wrap_torch_tensor, FP4


def quantize_mx4(w):
    """
    Quantize a weight tensor to MX4 (Microscaling 4-bit floating point) format.

    MX4 is a block-based quantization format developed by Microsoft and NVIDIA that:
    - Stores values in 4-bit format (16 possible values per element)
    - Uses a shared scaling factor per block (typically 32 elements)
    - Maintains good accuracy despite aggressive quantization
    - Is optimized for NVIDIA Hopper GPU architecture

    The quantization process:
    1. Partition tensor into blocks along specified axis
    2. Compute a scaling factor per block (stored in bfloat16)
    3. Quantize each element to 4 bits relative to its block's scale
    4. Convert to Hopper-optimized memory layout for efficient computation

    Benefits:
    - 4x memory reduction compared to bfloat16
    - 2-4x speedup in matrix multiplication on Hopper GPUs
    - Minimal accuracy loss (<1% degradation in most models)

    Args:
        w: Weight tensor to quantize [any shape]
            Will be converted to bfloat16 if not already

    Returns:
        Tuple of (quantized_weights, scale_factors):
            quantized_weights: 4-bit quantized values in Hopper MX layout
            scale_factors: Per-block scaling factors in bfloat16
    """
    # Downcast to MX4 format: quantize to 4 bits with per-block scales
    # axis=1 means scales are shared along dimension 1 (typically output dimension)
    w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)

    # Convert to Hopper-optimized memory layout for efficient matrix multiplication
    # HopperMXValueLayout arranges 4-bit values for optimal tensor core throughput
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), HopperMXValueLayout, mx_axis=1)

    # Convert scales to standard strided layout for easy access during computation
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)

    return w, w_scale


def swiglu(x, alpha: float = 1.702, limit: float = 7.0, interleaved: bool = True):
    """
    SwiGLU activation function (Swish-Gated Linear Unit).

    SwiGLU is a gated activation function introduced in the paper "GLU Variants Improve Transformer"
    (https://arxiv.org/abs/2002.05202). It combines Swish activation with gating, providing
    better performance than ReLU or GELU in many transformer models.

    The computation is: SwiGLU(x) = Swish(W_gate @ x) * (W_linear @ x)
    where Swish(x) = x * sigmoid(alpha * x)

    This implementation includes:
    - Clamping to prevent numerical overflow (important for bfloat16)
    - Offset of +1 to the linear path for better gradient flow
    - Support for both interleaved and chunked weight layouts

    Args:
        x: Input tensor [..., 2*hidden_dim]
            Expected to contain both gate and linear projections
        alpha: Swish activation temperature (default: 1.702)
            Higher values make Swish more like ReLU, lower values smoother
        limit: Clipping threshold to prevent overflow (default: 7.0)
            Important for bfloat16 to avoid inf/nan
        interleaved: Whether weights are interleaved (default: True)
            If True: [gate_0, linear_0, gate_1, linear_1, ...]
            If False: [gate_0, gate_1, ...], [linear_0, linear_1, ...]

    Returns:
        Activated output tensor [..., hidden_dim]
            Output dimension is half of input (gate and linear are combined)

    Note:
        This is a standalone implementation for testing. In production, the fused
        Triton kernel is used to avoid materializing intermediate tensors.
    """
    if interleaved:
        # Split interleaved layout: even indices are gate, odd are linear
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
    else:
        # Split chunked layout: first half is gate, second half is linear
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)

    # Clamp to prevent overflow in bfloat16 (exp(7*1.702) is near max)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)

    # Apply Swish activation: x * sigmoid(alpha * x)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)

    # Combine with linear path: gate * (linear + 1)
    # The +1 offset helps with gradient flow and stability
    return out_glu * (x_linear + 1)


def moe(x, wg, w1, w1_mx, w2, w2_mx, bg, b1, b2, experts_per_token=4, num_experts=128, swiglu_limit=7.0, fused_act=True, interleaved=True):
    """
    Mixture of Experts (MoE) layer with sparse routing and MX4 quantization.

    This implements a high-performance sparse MoE layer where:
    1. A gate network selects top-k experts for each token
    2. Tokens are routed to their selected experts using gather/scatter operations
    3. Each expert computes: output = W2 @ SwiGLU(W1 @ input + b1) + b2
    4. Expert outputs are weighted by routing scores and combined

    The implementation uses several advanced optimizations:
    - MX4 quantization for expert weights (4-bit storage, bfloat16 computation)
    - Sparse routing: only k out of n experts are activated per token
    - Fused operations: matmul+activation combined in single Triton kernel
    - OGS (Optimized GEMM for Sparse) kernels for efficient sparse matrix operations

    Triton optimizations:
    - matmul_ogs: Custom Triton kernel for matrix multiplication with sparse routing
    - Supports MX4 quantized weights natively (no explicit dequantization)
    - Fused activation avoids materializing intermediate [tokens, 2*intermediate_dim] tensor
    - All operations are fused to minimize memory bandwidth

    Args:
        x: Input activations [tokens, hidden_dim]
        wg: Gate network weights [hidden_dim, num_experts] in bfloat16
        w1: First expert layer weights (quantized) [num_experts, hidden_dim, 2*intermediate_dim]
        w1_mx: Scaling factors for w1 MX4 quantization
        w2: Second expert layer weights (quantized) [num_experts, intermediate_dim, hidden_dim]
        w2_mx: Scaling factors for w2 MX4 quantization
        bg: Gate network bias [num_experts]
        b1: First expert layer bias [num_experts, 2*intermediate_dim]
        b2: Second expert layer bias [num_experts, hidden_dim]
        experts_per_token: Number of experts to activate per token (default: 4)
            Higher values increase compute but can improve quality
        num_experts: Total number of experts in the layer (default: 128)
        swiglu_limit: Clipping threshold for SwiGLU activation (default: 7.0)
        fused_act: Whether to fuse matmul+activation (default: True)
            Fused mode is faster but requires interleaved weights
        interleaved: Weight layout for W1 (default: True)
            True: [gate, linear] interleaved, False: [gates], [linears] chunked

    Returns:
        Output activations [tokens, hidden_dim]
            Sparse combination of expert outputs weighted by routing scores

    Performance characteristics:
    - Memory: O(tokens * hidden_dim + k * tokens * intermediate_dim)
        Only active expert outputs are materialized (k << num_experts)
    - Compute: O(tokens * k * (hidden_dim * intermediate_dim))
        Much less than dense: tokens * num_experts * (hidden_dim * intermediate_dim)
    - Bandwidth: Reduced by MX4 quantization (4x less weight traffic)
    """
    # Handle empty input (edge case during dynamic batching)
    if x.numel() == 0:
        return x

    # Configure precision for quantized and non-quantized operations
    # InFlexData indicates weights are in MX4 format and will be dequantized on-the-fly
    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=InFlexData()))  # Gate is not quantized

    # Step 1: Compute routing logits using gate network
    # Output: [tokens, num_experts] - raw scores for each expert
    with record_function("wg"):
        logits = matmul_ogs(x, wg, bg, precision_config=pcg)

    # Step 2: Route tokens to top-k experts
    # This computes softmax, selects top-k experts, and generates routing metadata
    # rdata: routing data including normalized gate weights
    # gather_indx: indices for gathering tokens to send to each expert
    # scatter_indx: indices for scattering expert outputs back to token positions
    with record_function("routing"):
        rdata, gather_indx, scatter_indx = routing(logits, experts_per_token, simulated_ep=1)

    # Step 3: First expert layer with SwiGLU activation
    if fused_act:
        # Fused path: matmul and activation combined in single Triton kernel
        # This is faster as it avoids writing/reading the intermediate tensor
        # Saves bandwidth: no materialization of [tokens, 2*intermediate_dim] tensor
        assert interleaved, "Fused activation requires interleaved weights"
        with record_function("w1+swiglu"):
            # Create fused activation specification for Triton kernel
            act = FusedActivation(
                FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
                (1.702, swiglu_limit),
                2  # Output is half the matmul output size (gate and linear combined)
            )
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
    else:
        # Separate path: matmul then activation (useful for debugging)
        # Less efficient due to intermediate materialization
        with record_function("w1"):
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        with record_function("swiglu"):
            x = swiglu(x, limit=swiglu_limit, interleaved=interleaved)

    # Step 4: Second expert layer (projection back to hidden dimension)
    # scatter_indx combines expert outputs back to original token positions
    # gammas applies the learned routing weights to weight expert contributions
    with record_function("w2"):
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2, gammas=rdata.gate_scal)

    return x
