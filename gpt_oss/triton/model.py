"""
Complete Transformer model implementation using Triton-optimized kernels.

This module implements a full transformer-based language model with the following features:
- Rotary Position Embeddings (RoPE) with YaRN scaling for long context
- Grouped Query Attention (GQA) with Flash Attention implementation
- Mixture of Experts (MoE) layers with MX4 quantization
- Sliding window attention with learned sinks
- KV cache for efficient autoregressive generation
- CUDA graph capture for minimal overhead during sampling

Architecture highlights:
- Attention: Custom Triton Flash Attention kernel with learned sinks and banded attention
- MoE FFN: Sparse expert routing with top-k selection and MX4 quantized weights
- Position encoding: RoPE with YaRN NTK-aware interpolation for context extension
- Normalization: RMSNorm (Root Mean Square Layer Normalization)

Performance optimizations:
- Triton kernels for attention and MoE operations
- MX4 quantization (4-bit) for expert weights (4x memory reduction)
- KV cache to avoid recomputing attention for past tokens
- CUDA graphs for single-token generation (eliminates kernel launch overhead)
- Grouped query attention to reduce KV cache size

The model is designed for efficient inference on NVIDIA GPUs, particularly
Hopper architecture which has native support for MX4 operations.
"""

import json
import math
import os

import torch
from torch.profiler import record_function

from gpt_oss.torch.model import ModelConfig, RMSNorm
from gpt_oss.torch.weights import Checkpoint
from gpt_oss.triton.attention import attention, attention_ref
from gpt_oss.triton.moe import quantize_mx4, moe


class RotaryEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) with YaRN scaling for extended context.

    RoPE encodes position information by rotating query and key representations
    in a manner that naturally captures relative positions. This implementation
    includes YaRN (Yet another RoPE extensioN method) which enables extending
    the context length beyond what the model was trained on.

    Key features:
    - Rotary embeddings for efficient relative position encoding
    - YaRN scaling with NTK-aware interpolation for context extension
    - Precomputed sin/cos lookup tables for efficiency
    - Support for very long contexts (up to 131k tokens)

    References:
    - RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - YaRN: "YaRN: Efficient Context Window Extension of Large Language Models"
      https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        max_context_length: int = 131072,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize RoPE with YaRN scaling parameters.

        Args:
            head_dim: Dimension of each attention head (must be even)
            base: Base frequency for rotary embeddings (typically 10000 or 500000)
            dtype: Data type for sin/cos tables (typically float32 for precision)
            initial_context_length: Original training context length (default: 4096)
                Used to determine interpolation vs extrapolation regions in YaRN
            max_context_length: Maximum supported context length (default: 131072)
                Pre-allocates sin/cos tables up to this length
            scaling_factor: Overall scaling factor for context extension (default: 1.0)
                Values > 1.0 enable longer contexts via frequency interpolation
            ntk_alpha: NTK-aware scaling lower bound (default: 1.0)
                Controls high-frequency component scaling
            ntk_beta: NTK-aware scaling upper bound (default: 32.0)
                Controls low-frequency component scaling
            device: Device for storing sin/cos tables
        """
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.max_context_length = max_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device
        self.cos, self.sin = self._compute_cos_sin(0, self.max_context_length)

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """
        Compute YaRN concentration factor and inverse frequencies for RoPE.

        YaRN enables extending context length by combining three techniques:
        1. Frequency interpolation: scales down frequencies to fit longer sequences
        2. NTK-aware interpolation: smoothly transitions between interpolation and extrapolation
        3. Attention temperature scaling: adjusts attention concentration

        The method partitions frequency dimensions into three regions:
        - Low frequencies (long wavelengths): use extrapolation (NTK scaling)
        - Mid frequencies: smooth ramp between extrapolation and interpolation
        - High frequencies (short wavelengths): use interpolation (direct scaling)

        This prevents high-frequency aliasing while maintaining low-frequency expressiveness.

        Returns:
            Tuple of (concentration, inv_freq):
                concentration: Temperature scaling factor for attention (scalar)
                inv_freq: Inverse frequencies for each dimension [head_dim//2]

        Reference:
            YaRN paper: https://arxiv.org/abs/2309.00071
        """
        # Compute base frequencies for each dimension (every other dimension gets rotated)
        # freq[i] = base^(2i/head_dim) for i in [0, head_dim/2)
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )

        if self.scaling_factor > 1.0:
            # YaRN scaling is enabled for context extension

            # Concentration factor: scales attention temperature
            # Prevents attention from becoming too diffuse with longer contexts
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )

            d_half = self.head_dim / 2

            # NTK-aware interpolation: compute dimension boundaries
            # Low-frequency dimensions (low < d) use extrapolation (NTK scaling)
            # High-frequency dimensions (d > high) use interpolation (direct scaling)
            # Middle dimensions (low < d < high) smoothly transition between the two
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            # Interpolation: scale frequencies down by scaling_factor
            # Used for high-frequency components (short wavelengths)
            interpolation = 1.0 / (self.scaling_factor * freq)

            # Extrapolation: use original frequencies (NTK scaling happens via position scaling)
            # Used for low-frequency components (long wavelengths)
            extrapolation = 1.0 / freq

            # Create smooth ramp between low and high frequency boundaries
            # mask = 1 for low frequencies (extrapolation), 0 for high frequencies (interpolation)
            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)  # Clamp to [0, 1] and invert

            # Blend between extrapolation and interpolation based on frequency
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            # No scaling: standard RoPE
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, start: int, num_tokens: int):
        """
        Precompute sin and cos values for rotary embeddings.

        Args:
            start: Starting position index
            num_tokens: Number of positions to compute

        Returns:
            Tuple of (cos, sin) tensors [num_tokens, head_dim//2]
                These are used to rotate query and key representations
        """
        concentration, inv_freq = self._compute_concentration_and_inv_freq()

        # Create position indices
        t = torch.arange(start, start + num_tokens, dtype=torch.float32, device=self.device)

        # Compute angle for each (position, frequency) pair
        # Shape: [num_tokens, head_dim//2]
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # Compute cos and sin, scaled by concentration factor
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    @record_function("rotate")
    def _rotate(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embedding rotation to input tensor.

        The rotation is applied pairwise to dimensions: (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)
        This encodes position information in a way that naturally captures relative positions.

        Args:
            x: Input tensor [batch, seq_len, heads, head_dim]
            cos: Cosine values [seq_len, head_dim//2]
            sin: Sine values [seq_len, head_dim//2]

        Returns:
            Rotated tensor [batch, seq_len, heads, head_dim]
        """
        # Broadcast cos/sin to match input shape
        cos = cos[None, :, None, :].to(x.dtype)
        sin = sin[None, :, None, :].to(x.dtype)

        # Split into pairs of dimensions for rotation
        x1, x2 = torch.chunk(x, 2, dim=-1)

        # Apply 2D rotation matrix: [[cos, -sin], [sin, cos]]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        # Concatenate rotated pairs back together
        return torch.cat((o1, o2), dim=-1)

    @record_function("rope")
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            query: Query tensor [batch, num_tokens, num_heads, head_dim]
            key: Key tensor [batch, num_tokens, num_kv_heads, head_dim]
            offset: Starting position in sequence (for KV cache) [1]
                When using KV cache, offset tracks where we are in the sequence

        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as inputs
        """
        batch_size, num_tokens, num_heads, head_dim = query.shape
        batch_size, num_tokens, num_key_value_heads, head_dim = key.shape

        # Compute position indices for this batch of tokens
        idx = torch.arange(num_tokens, device=query.device, dtype=torch.long) + offset

        # Wrap around to stay within precomputed table (for very long sequences)
        idx = idx % self.max_context_length

        # Lookup precomputed sin/cos values for these positions
        cos = self.cos.index_select(0, idx)
        sin = self.sin.index_select(0, idx)

        # Apply rotation to both query and key
        query = self._rotate(query, cos, sin)
        key = self._rotate(key, cos, sin)
        return query, key


class Cache:
    """
    Key-Value cache for efficient autoregressive generation.

    The KV cache stores previously computed key and value tensors to avoid
    recomputing them during autoregressive generation. This is a critical
    optimization that reduces the computational cost from O(N^2) to O(N)
    as sequence length grows.

    Features:
    - Preallocated buffers for maximum efficiency
    - Support for grouped query attention (GQA) with separate KV heads
    - Batch dimension support for parallel generation
    - In-place updates to minimize memory allocations

    Memory layout: [batch_size, max_seq_len, num_kv_heads, head_dim]
    """
    def __init__(self, batch_size, n_ctx, n_kv_heads, d_head=64, device: torch.device | None = None):
        """
        Initialize KV cache with preallocated buffers.

        Args:
            batch_size: Number of sequences to cache simultaneously
            n_ctx: Maximum context length (cache capacity)
            n_kv_heads: Number of key/value heads (for GQA)
            d_head: Dimension of each attention head (default: 64)
            device: Device for cache tensors
        """
        # Preallocate key and value buffers
        self.k = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)

        # Track current position in the cache (how many tokens have been added)
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self):
        """Reset cache to empty state (zeros out all entries and resets offset)."""
        self.k.zero_()
        self.v.zero_()
        self.offset.zero_()

    def repeat_interleave(self, n):
        """
        Repeat each cache entry n times along the batch dimension.

        Used for beam search or other scenarios where each sequence
        needs to be duplicated.

        Args:
            n: Number of times to repeat each batch entry
        """
        self.k = self.k.repeat_interleave(n, dim=0)
        self.v = self.v.repeat_interleave(n, dim=0)

    def truncate(self, n_ctx):
        """
        Truncate the cache to the first n_ctx tokens.

        Zeros out everything after position n_ctx and sets offset to n_ctx.
        Useful for resetting to a previous state or implementing sliding windows.

        Args:
            n_ctx: Number of tokens to keep (rest are zeroed)

        Returns:
            Tuple of (k, v) tensors after truncation
        """
        batch_size, _, n_kv_heads, d_head = self.k.shape
        assert batch_size == self.v.shape[0]
        assert n_ctx <= self.k.shape[1]

        # Zero out tokens beyond n_ctx
        self.k[:, n_ctx:, :, :].zero_()
        self.v[:, n_ctx:, :, :].zero_()

        # Update offset to reflect truncation
        self.offset.fill_(n_ctx)
        return self.k, self.v

    def extend(self, k, v):
        """
        Add new key/value pairs to the cache at the current offset.

        This is the core operation for autoregressive generation:
        each new token's keys and values are added to the cache.

        Args:
            k: New keys to add [batch, num_new_tokens, n_kv_heads, d_head]
            v: New values to add [batch, num_new_tokens, n_kv_heads, d_head]

        Returns:
            Tuple of (k, v) tensors with full cache contents
        """
        batch_size, n_ctx, *_rest = k.shape
        assert batch_size == self.k.shape[0]

        # Compute destination indices in cache (starting from current offset)
        indices = torch.arange(0, n_ctx, device=k.device, dtype=torch.long) + self.offset

        # Copy new keys and values into cache at computed indices
        # index_copy_ does in-place copy without allocating new memory
        self.k.index_copy_(1, indices, k)
        self.v.index_copy_(1, indices, v)

        # Advance offset by number of tokens added
        self.offset.add_(n_ctx)

        return self.k, self.v


class AttentionBlock(torch.nn.Module):
    """
    Attention layer with Flash Attention, learned sinks, and sliding window.

    This implements a complete attention block including:
    - RMSNorm pre-normalization
    - Linear projections for Q, K, V
    - Rotary position embeddings (RoPE)
    - Flash Attention with learned sinks and optional sliding window
    - Output projection
    - Residual connection

    Optimizations:
    - Uses custom Triton Flash Attention kernel for efficiency
    - Supports grouped query attention (GQA) to reduce KV cache size
    - Learned sinks provide additional capacity for attention
    - Sliding window attention (alternating layers) for local patterns

    The attention mechanism uses causal masking for autoregressive generation.
    """
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        """
        Initialize attention block.

        Args:
            config: Model configuration
            layer_idx: Layer index (used to alternate sliding window on/off)
            device: Device for parameters
        """
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Alternating sliding window pattern: even layers use sliding window, odd layers use full attention
        # This allows the model to combine local and global attention patterns
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.layer_idx = layer_idx

        # Learned attention sinks: per-head biases that act as additional attention weights
        # These provide a stable "sink" for attention, improving training stability
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )

        # Pre-normalization with RMSNorm
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Combined QKV projection for efficiency
        # For GQA: Q has more heads than K/V
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )

        # Output projection
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )

        # Attention scaling factor (1/sqrt(d_k) for stability)
        self.sm_scale = 1 / math.sqrt(config.head_dim)

        # Rotary position embeddings with YaRN scaling
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    @record_function("attn")
    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        """
        Forward pass for attention block.

        Args:
            x: Input activations [batch, seq_len, hidden_dim]
            cache: Optional KV cache for autoregressive generation

        Returns:
            Output activations [batch, seq_len, hidden_dim]
                Includes residual connection from input
        """
        batch_size, n_ctx, dim = x.shape

        # Apply pre-normalization
        t = self.norm(x)

        # Compute Q, K, V projections in one go for efficiency
        with record_function("qkv"):
            qkv = self.qkv(t)

            # Split into Q, K, V components
            # Q gets more heads (num_attention_heads) for grouped query attention
            # K and V share fewer heads (num_key_value_heads)
            qkv_parts = (
                self.num_attention_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim
            )
            q, k, v = torch.split(qkv, qkv_parts, dim=-1)
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # Reshape into multi-head format
        q = q.view(batch_size, n_ctx, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, n_ctx, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, n_ctx, self.num_key_value_heads, self.head_dim)

        # Apply rotary embeddings and optionally update KV cache
        if cache is not None:
            # Using KV cache for autoregressive generation
            offset = cache.offset.clone()
            q, k = self.rope(q, k, offset=offset)
            k, v = cache.extend(k, v)  # Add new K/V to cache, get full cache back
        else:
            # No cache (prefill phase or training)
            offset = torch.zeros((1,), dtype=torch.long, device=x.device)
            q, k = self.rope(q, k, offset=offset)

        # Reshape Q for grouped query attention
        # Convert from [batch, seq, num_q_heads, head_dim] to
        # [batch, seq, num_kv_heads, repeat_factor, head_dim]
        # where repeat_factor = num_q_heads / num_kv_heads
        q = q.view(
            batch_size,
            n_ctx,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )

        # Apply attention using Triton kernel or reference implementation
        with record_function("attn_kernel"):
            if n_ctx == 1:
                # Single token: use reference implementation (Triton has overhead for small sizes)
                t = attention_ref(
                    q,
                    k,
                    v,
                    self.sinks,
                    self.sm_scale,
                    self.sliding_window,
                    offset,
                )
            else:
                # Multiple tokens: use optimized Triton Flash Attention kernel
                t = attention(
                    q,
                    k,
                    v,
                    self.sinks,
                    self.sm_scale,
                    self.sliding_window,
                    offset,
                )

                # For small sequences, verify Triton kernel matches reference
                # This is a development-time check for correctness
                if n_ctx < 64:
                    t1 = attention_ref(
                        q,
                        k,
                        v,
                        self.sinks,
                        self.sm_scale,
                        self.sliding_window,
                        offset,
                    )
                    torch.testing.assert_close(t, t1)
                    t = t1

        # Output projection
        with record_function("c_proj"):
            t = self.out(t)

        # Residual connection
        t = x + t
        return t


class MLPBlock(torch.nn.Module):
    """
    Mixture of Experts (MoE) feed-forward block with MX4 quantization.

    This block replaces the standard dense FFN with a sparse MoE layer:
    - Gate network routes each token to top-k experts
    - Each expert is a small FFN: W2 @ SwiGLU(W1 @ x)
    - Expert weights are quantized to MX4 (4-bit) for memory efficiency
    - Only k out of n total experts are activated per token

    Benefits of MoE:
    - Sparse computation: O(k) instead of O(n) experts per token
    - Parameter efficiency: large model capacity with controlled compute
    - Specialization: experts can specialize on different patterns

    Optimizations:
    - Triton kernels for efficient sparse routing and matmuls
    - MX4 quantization reduces memory by 4x
    - Fused SwiGLU activation eliminates intermediate tensors
    """
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        """
        Initialize MoE block with quantized expert weights.

        Args:
            config: Model configuration
            layer_idx: Layer index
            device: Device for parameters
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit

        # Pre-normalization
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Gate network for expert routing (not quantized for better routing accuracy)
        self.gate = torch.nn.ParameterDict({
            "weight": torch.nn.Parameter(
                torch.empty(
                    (config.hidden_size, config.num_experts),
                    device=device,
                    dtype=torch.bfloat16,
                )
            ),
            "bias": torch.nn.Parameter(
                torch.empty(
                    (config.num_experts,),
                    device=device,
                    dtype=torch.bfloat16,
                )
            ),
        })

        # First expert layer weights (MX4 quantized)
        # Output dim is 2x for SwiGLU (gate + linear paths)
        self.mlp1_weight_tensor, self.mlp1_weight_mx = quantize_mx4(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size * 2,
                ),
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        self.mlp1_weight = torch.nn.Parameter(self.mlp1_weight_tensor.storage.data, requires_grad=False)
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=device,
                dtype=torch.bfloat16,
            )
        )

        # Second expert layer weights (MX4 quantized)
        self.mlp2_weight_tensor, self.mlp2_weight_mx = quantize_mx4(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            ),
        )
        self.mlp2_weight = torch.nn.Parameter(self.mlp2_weight_tensor.storage.data, requires_grad=False)
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    @record_function("mlp")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MoE block.

        Args:
            x: Input activations [batch, seq_len, hidden_dim]

        Returns:
            Output activations [batch, seq_len, hidden_dim]
                Includes residual connection from input
        """
        batch_size, n_ctx, dim = x.shape

        # Apply pre-normalization
        t = self.norm(x)

        # Flatten batch and sequence dimensions for MoE routing
        # MoE operates on [num_tokens, hidden_dim]
        t = t.view(batch_size * n_ctx, dim)

        # Apply MoE layer with Triton kernels
        t = moe(
            t,
            self.gate["weight"],
            self.mlp1_weight_tensor, self.mlp1_weight_mx,  # W1 with MX4 scales
            self.mlp2_weight_tensor, self.mlp2_weight_mx,  # W2 with MX4 scales
            self.gate["bias"].float(),
            self.mlp1_bias.float(),
            self.mlp2_bias.float(),
            experts_per_token=self.experts_per_token,
            num_experts=self.num_experts,
            swiglu_limit=self.swiglu_limit,
        )

        # Reshape back to [batch, seq_len, hidden_dim]
        t = t.view(batch_size, n_ctx, dim)

        # Residual connection
        return x + t


class TransformerBlock(torch.nn.Module):
    """
    Single transformer layer combining attention and MoE blocks.

    Each layer consists of:
    1. Attention block with Flash Attention and learned sinks
    2. MoE block with sparse expert routing

    Both blocks use pre-normalization and residual connections.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        """
        Initialize transformer block.

        Args:
            config: Model configuration
            layer_idx: Layer index (used for alternating sliding window)
            device: Device for parameters
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, layer_idx, device)

    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        """
        Forward pass through attention and MLP.

        Args:
            x: Input activations [batch, seq_len, hidden_dim]
            cache: Optional KV cache for autoregressive generation

        Returns:
            Output activations [batch, seq_len, hidden_dim]
        """
        x = self.attn(x, cache=cache)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    """
    Complete transformer model with embeddings and language modeling head.

    This is the top-level model class that combines:
    - Token embeddings
    - Stack of transformer blocks (attention + MoE)
    - Final layer normalization
    - Unembedding to vocabulary logits

    The model supports:
    - Autoregressive generation with KV cache
    - Long context via RoPE with YaRN scaling
    - Efficient inference via Triton kernels and MX4 quantization
    - Loading from checkpoints with automatic weight quantization
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        """
        Initialize transformer model.

        Args:
            config: Model configuration
            device: Device for parameters
        """
        super().__init__()
        self.config = config

        # Token embeddings
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )

        # Stack of transformer blocks
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final layer normalization
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Language modeling head (vocabulary projection)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input token IDs [batch, seq_len]
            caches: Optional list of KV caches (one per layer)

        Returns:
            Logits over vocabulary [batch, seq_len, vocab_size]
        """
        # Default to no caching if not provided
        caches = caches or [None] * len(self.block)

        # Embed tokens
        with record_function("embedding"):
            x = self.embedding(x)

        # Pass through all transformer blocks
        for block, cache in zip(self.block, caches):
            with record_function("block"):
                x = block(x, cache=cache)

        # Final normalization
        with record_function("norm_f"):
            x = self.norm(x)

        # Project to vocabulary logits
        with record_function("unembedding"):
            x = self.unembedding(x)

        return x.float()

    @staticmethod
    def from_checkpoint(
        path: str, config: ModelConfig | None = None, device: str | torch.device = "cuda",
    ) -> "Transformer":
        """
        Load model from checkpoint with automatic MX4 quantization.

        This loads pretrained weights and quantizes expert weights to MX4 format
        for efficient inference. The quantization happens during loading to avoid
        storing both full-precision and quantized weights in memory.

        Args:
            path: Path to checkpoint directory
            config: Optional model config (loaded from checkpoint if not provided)
            device: Device to load model onto

        Returns:
            Loaded and quantized model ready for inference
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # Load config from checkpoint if not provided
        if config is None:
            config_path = os.path.join(path, "config.json")
            with open(config_path, "r") as f:
                json_config = json.load(f)
                config = ModelConfig(**json_config)

        # Create model with randomly initialized weights
        model = Transformer(config=config, device=device)
        model.eval()

        # Open checkpoint for loading
        checkpoint = Checkpoint(path, device)

        # Load and quantize weights
        for name, param in model.named_parameters():
            # Clear cache to avoid OOM when loading large models
            torch.cuda.empty_cache()

            # Load parameter from checkpoint
            loaded_tensor = checkpoint.get(name)

            # Handle MLP1 weights (expert first layer)
            if "mlp1" in name:
                if "weight" in name:
                    # Quantize to MX4 (transpose first for correct axis)
                    loaded_tensor, scales = quantize_mx4(loaded_tensor.mT.contiguous())

                    # Store MX4 scales in the model
                    _, block_index, _, _ = name.split(".")
                    model.block[int(block_index)].mlp.mlp1_weight_mx = scales

                    # Copy quantized weights to parameter
                    param.data.copy_(loaded_tensor.storage.data)
                else:
                    # Bias: copy directly without quantization
                    param.data.copy_(loaded_tensor)

            # Handle MLP2 weights (expert second layer)
            elif "mlp2_weight" in name:
                # Quantize to MX4
                loaded_tensor, scales = quantize_mx4(loaded_tensor.mT.contiguous())

                # Store MX4 scales
                _, block_index, _, _ = name.split(".")
                model.block[int(block_index)].mlp.mlp2_weight_mx = scales

                # Copy quantized weights
                param.data.copy_(loaded_tensor.storage.data)

            # Handle gate weights (expert routing)
            elif "gate" in name and loaded_tensor.ndim == 2:
                # Transpose gate weights to match expected layout
                loaded_tensor = loaded_tensor.mT.contiguous()
                param.data.copy_(loaded_tensor)

            # All other parameters: copy directly
            else:
                param.data.copy_(loaded_tensor)

        # Clear cache after loading to free memory
        torch.cuda.empty_cache()
        return model


class TokenGenerator:
    """
    Efficient token-by-token generator with CUDA graph optimization.

    This class provides optimized autoregressive generation by:
    - Maintaining KV cache to avoid recomputing past tokens
    - Using CUDA graphs to eliminate kernel launch overhead
    - Supporting temperature-based sampling

    CUDA graph optimization:
    - After warmup, the single-token forward pass is captured in a graph
    - Replaying the graph is much faster than launching kernels individually
    - This is crucial for low-latency generation (important for single-token steps)

    Performance: Achieves near-optimal throughput for autoregressive generation.
    """
    @torch.inference_mode()
    def __init__(self, checkpoint: str, context: int, device: torch.device):
        """
        Initialize generator with CUDA graph capture.

        Args:
            checkpoint: Path to model checkpoint
            context: Maximum context length (KV cache size)
            device: Device for inference
        """
        self.device = device

        # Load model with MX4 quantization
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

        # Create KV caches for all layers
        self.caches = [
            Cache(1, context, self.model.config.num_key_value_heads, device=self.device)
            for _ in range(len(self.model.block))
        ]

        # Preallocate input tensor for CUDA graph
        self.input_token = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Warmup: run one forward pass to initialize everything
        self.model(self.input_token[None, :], caches=self.caches)

        # Capture CUDA graph for single-token generation
        # This records all kernel launches and replays them efficiently
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            # This forward pass is recorded; subsequent replays will be much faster
            self.logits = self.model(self.input_token[None, :], caches=self.caches)[0]

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int] | None = None,
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        """
        Generate tokens autoregressively from a prompt.

        This method implements efficient token generation using:
        - KV cache to avoid recomputing past tokens
        - CUDA graphs for minimal overhead per token
        - Temperature-based sampling

        Args:
            prompt_tokens: Input prompt as list of token IDs
            stop_tokens: Optional list of token IDs that end generation
            temperature: Sampling temperature (0.0 = greedy, >1.0 = more random)
            max_tokens: Maximum tokens to generate (0 = unlimited)
            return_logprobs: Whether to return log probabilities

        Yields:
            If return_logprobs=False: next_token_id (int)
            If return_logprobs=True: (next_token_id, logprob) tuple
        """
        # Initialize stop tokens
        stop_tokens = stop_tokens or []

        # Reset all KV caches for new generation
        for cache in self.caches:
            cache.reset()

        # Convert prompt to tensor
        prompt_tokens = torch.as_tensor(prompt_tokens, dtype=torch.int32, device=self.device)

        # Prefill: process all prompt tokens except the last one
        # This populates the KV cache with prompt context
        # We exclude the last token to maintain consistency with the generation loop
        self.model(prompt_tokens[None, :-1], self.caches)

        # Initialize with last prompt token
        predicted_token = prompt_tokens[-1]
        num_generated_tokens = 0

        # Generation loop
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            # Update input token for CUDA graph
            # The graph uses this preallocated tensor
            self.input_token[0] = predicted_token

            # Replay CUDA graph: runs entire forward pass with minimal overhead
            # This is much faster than calling model() due to eliminated launch overhead
            self.graph.replay()

            # Sample next token
            if temperature == 0.0:
                # Greedy sampling: always pick most likely token
                predicted_token = torch.argmax(self.logits[-1, :], dim=-1).item()
            else:
                # Temperature-based sampling
                # Lower temperature = more peaked distribution (more conservative)
                # Higher temperature = flatter distribution (more diverse)
                probs = torch.softmax(self.logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs[-1, :], num_samples=1).item()

            num_generated_tokens += 1

            # Yield result
            if return_logprobs:
                # Compute log probability of selected token
                logprobs = torch.log_softmax(self.logits[-1, :], dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            # Check for stop tokens
            if predicted_token in stop_tokens:
                break
