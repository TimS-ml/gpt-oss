"""
Transformer Model Architecture

This module implements a transformer-based language model with the following features:
- Multi-head grouped-query attention with sliding window and sink tokens
- Mixture of Experts (MoE) with sparse routing in MLP layers
- RoPE (Rotary Position Embeddings) with YaRN scaling for long context
- SwiGLU activation function in the MLP
- RMSNorm for layer normalization
- Support for distributed inference across multiple GPUs

The architecture is designed for efficient inference on large language models,
with optimizations for memory usage and computational efficiency.
"""

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from gpt_oss.torch.weights import Checkpoint


@dataclass
class ModelConfig:
    """
    Configuration class for the Transformer model.

    This dataclass stores all hyperparameters needed to construct the model.
    Default values are set for a specific model variant but can be overridden
    when loading from a checkpoint.

    Attributes:
        num_hidden_layers (int): Number of transformer blocks (depth of the model)
        num_experts (int): Total number of experts in each MoE layer
        experts_per_token (int): Number of experts activated per token (sparse MoE)
        vocab_size (int): Size of the vocabulary (number of unique tokens)
        hidden_size (int): Dimension of hidden representations
        intermediate_size (int): Dimension of MLP intermediate layer (per expert)
        swiglu_limit (float): Clamping limit for SwiGLU activation to prevent overflow
        head_dim (int): Dimension of each attention head
        num_attention_heads (int): Total number of attention heads (queries)
        num_key_value_heads (int): Number of key/value heads (for grouped-query attention)
        sliding_window (int): Size of sliding attention window (for local attention)
        initial_context_length (int): Base context length for RoPE initialization
        rope_theta (float): Base frequency for RoPE (rotation angle)
        rope_scaling_factor (float): Factor for extending context beyond initial length
        rope_ntk_alpha (float): NTK-aware scaling parameter (low frequency bound)
        rope_ntk_beta (float): NTK-aware scaling parameter (high frequency bound)

    Note:
        - Grouped-query attention: num_attention_heads > num_key_value_heads reduces KV cache size
        - Sliding window attention: Only applied to every other layer to balance locality and global context
        - YaRN scaling: Enables the model to handle sequences longer than initial_context_length
    """
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is a simpler and more efficient alternative to LayerNorm that
    normalizes using only the root mean square (no mean subtraction).
    It has been shown to work well in transformer models while being
    computationally cheaper than standard LayerNorm.

    Formula: x_normalized = x / rms(x) * scale
    where rms(x) = sqrt(mean(x^2) + eps)

    Attributes:
        num_features (int): Size of the feature dimension to normalize
        eps (float): Small constant for numerical stability
        scale (nn.Parameter): Learnable per-feature scaling parameter
    """

    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        """
        Initialize RMSNorm layer.

        Args:
            num_features (int): Dimension of the input features
            eps (float): Small value added to denominator for numerical stability
            device (torch.device): Device to place the parameters on
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # Scale parameter is kept in float32 for numerical stability
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., num_features)

        Returns:
            torch.Tensor: Normalized tensor with same shape as input

        Note:
            Computation is done in float32 for stability, then cast back to input dtype
        """
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype  # Save original dtype and upcast to float32
        # Compute RMS: sqrt(mean(x^2))
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        # Apply learnable scale and cast back to original dtype
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding (RoPE) to input tensor.

    RoPE encodes position information by rotating pairs of features in the
    embedding space. This allows the model to capture relative positions
    through the dot product of rotated vectors.

    The rotation is applied using the 2D rotation matrix:
    [[cos, -sin],
     [sin,  cos]]

    Args:
        x (torch.Tensor): Input tensor to rotate, shape (..., dim)
        cos (torch.Tensor): Cosine values for rotation angles
        sin (torch.Tensor): Sine values for rotation angles

    Returns:
        torch.Tensor: Rotated tensor with same shape as input

    Note:
        The input is split into pairs (x1, x2) and rotated using:
        x1' = x1 * cos - x2 * sin
        x2' = x2 * cos + x1 * sin
    """
    # Add dimension and match dtype
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    # Split features into pairs for rotation
    x1, x2 = torch.chunk(x, 2, dim=-1)

    # Apply 2D rotation formula
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    # Concatenate rotated pairs back together
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) with YaRN scaling for long context.

    RoPE encodes absolute position information in a way that naturally captures
    relative positions. YaRN (Yet another RoPE extensioN) extends RoPE to handle
    sequences longer than the training context by:
    1. Interpolating low frequencies (which encode fine-grained positions)
    2. Extrapolating high frequencies (which encode coarse positions)
    3. Applying attention temperature scaling (concentration)

    This allows the model to generalize to longer sequences than it was trained on.

    Reference: YaRN paper - https://arxiv.org/abs/2309.00071

    Attributes:
        head_dim (int): Dimension of each attention head
        base (int): Base frequency for RoPE (theta parameter)
        dtype (torch.dtype): Data type for computations
        initial_context_length (int): Context length model was trained on
        scaling_factor (float): How much to extend context (e.g., 32x = 128K tokens)
        ntk_alpha (float): Low frequency bound for NTK-aware scaling
        ntk_beta (float): High frequency bound for NTK-aware scaling
        device (torch.device): Device for tensor operations
    """

    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize Rotary Embedding with YaRN scaling.

        Args:
            head_dim (int): Dimension of attention heads (must be even)
            base (int): Base frequency (theta) for rotation
            dtype (torch.dtype): Data type for cos/sin computations
            initial_context_length (int): Training context length
            scaling_factor (float): Context extension factor (1.0 = no scaling)
            ntk_alpha (float): Alpha parameter for NTK-aware interpolation
            ntk_beta (float): Beta parameter for NTK-aware interpolation
            device (torch.device): Device to place tensors on
        """
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> tuple[float, torch.Tensor]:
        """
        Compute YaRN concentration and inverse frequencies.

        YaRN scales RoPE to support longer contexts through:
        1. Concentration: Temperature scaling for attention (reduces entropy)
        2. NTK-by-parts: Interpolation for low freq, extrapolation for high freq

        Returns:
            tuple[float, torch.Tensor]: (concentration factor, inverse frequencies)

        Reference: YaRN paper - https://arxiv.org/abs/2309.00071
        """
        # Compute base frequencies: base^(d/dim) for d in [0, 2, 4, ..., head_dim-2]
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )

        if self.scaling_factor > 1.0:
            # YaRN concentration: reduces attention entropy for long contexts
            # Helps prevent attention from becoming too diffuse
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )

            d_half = self.head_dim / 2

            # NTK-by-parts: compute boundaries for interpolation vs extrapolation
            # Low frequencies encode fine-grained positions -> interpolate
            # High frequencies encode coarse positions -> extrapolate
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
            assert 0 < low < high < d_half - 1, "Invalid NTK parameters"

            # Compute inverse frequencies for both strategies
            interpolation = 1.0 / (self.scaling_factor * freq)  # Scale down freq
            extrapolation = 1.0 / freq  # Keep freq unchanged

            # Create smooth transition between interpolation and extrapolation
            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)  # 1 for low freqs, 0 for high freqs

            # Blend interpolation (low freq) and extrapolation (high freq)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            # No scaling needed: standard RoPE
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine and sine values for RoPE rotation.

        Args:
            num_tokens (int): Number of tokens in the sequence

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (cos values, sin values) for each position
        """
        concentration, inv_freq = self._compute_concentration_and_inv_freq()

        # Create position indices [0, 1, 2, ..., num_tokens-1]
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)

        # Compute rotation angles: position * inverse_frequency
        # Shape: [num_tokens, head_dim/2]
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # Compute cos and sin, scaled by concentration
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            query (torch.Tensor): Query tensor, shape [num_tokens, ...]
            key (torch.Tensor): Key tensor, shape [num_tokens, ...]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Rotated (query, key) tensors

        Note:
            Only query and key are rotated (not value), as position information
            is encoded through their dot product in attention.
        """
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        # Reshape query for rotation, apply, then restore original shape
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        # Reshape key for rotation, apply, then restore original shape
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)

        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    """
    Scaled Dot-Product Attention with optional sliding window and sink tokens.

    This implements multi-head attention with grouped queries (GQA), where
    multiple query heads share the same key/value heads for efficiency.
    It also supports:
    - Causal masking (for autoregressive generation)
    - Sliding window attention (for local context)
    - Sink tokens (learnable attention sinks for stability)

    Args:
        Q (torch.Tensor): Query tensor, shape [n_tokens, n_heads, q_mult, d_head]
                         q_mult is the number of query heads per KV head (GQA)
        K (torch.Tensor): Key tensor, shape [n_tokens, n_heads, d_head]
        V (torch.Tensor): Value tensor, shape [n_tokens, n_heads, d_head]
        S (torch.Tensor): Sink tokens (learnable attention logits), shape [n_heads, q_mult] or compatible
        sm_scale (float): Scaling factor for attention scores (typically 1/sqrt(d_head))
        sliding_window (int): Size of sliding attention window (0 = full attention)

    Returns:
        torch.Tensor: Attention output, shape [n_tokens, n_heads * q_mult * d_head]

    Note:
        - Grouped-query attention: Each KV head is shared by q_mult query heads
        - Sink tokens: Prevent attention collapse by providing stable attention targets
        - Sliding window: Limits attention to nearby tokens for efficiency
    """
    # sliding_window == 0 means no sliding window (full attention)
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)

    # Expand K and V to match query multiplicity for grouped-query attention
    # Each KV head is replicated q_mult times to match the query heads
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)

    # Reshape and expand sink tokens to match attention dimensions
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)

    # Create causal mask: prevent attending to future tokens (upper triangular)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)

    # Add sliding window mask if enabled: prevent attending to tokens too far in the past
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )

    # Compute attention scores: Q @ K^T
    # einsum: qhmd,khmd->hmqk computes dot product for each head
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)

    # Scale attention scores and apply causal mask
    QK *= sm_scale
    QK += mask[None, None, :, :]

    # Concatenate sink tokens to attention scores
    # This gives the model stable attention targets
    QK = torch.cat([QK, S], dim=-1)

    # Compute attention weights via softmax
    W = torch.softmax(QK, dim=-1)

    # Remove sink token weights (we only needed them for softmax normalization)
    W = W[..., :-1]

    # Compute attention output: weighted sum of values
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)

    # Reshape to combine all heads into feature dimension
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    """
    Multi-head grouped-query attention block with RoPE and sliding window.

    This implements a complete attention layer including:
    - Pre-normalization with RMSNorm
    - Grouped-query attention (GQA) for efficiency
    - Rotary position embeddings (RoPE) with YaRN scaling
    - Optional sliding window attention
    - Learnable sink tokens for attention stability
    - Residual connection

    The grouped-query attention reduces memory usage by having multiple query
    heads share the same key/value heads, which is especially important for
    inference with large context lengths.

    Attributes:
        head_dim (int): Dimension of each attention head
        num_attention_heads (int): Total number of query heads
        num_key_value_heads (int): Number of key/value heads (GQA)
        sliding_window (int): Window size for local attention (0 = full attention)
        sinks (nn.Parameter): Learnable attention sink tokens
        norm (RMSNorm): Pre-attention normalization layer
        qkv (nn.Linear): Projection to query, key, and value
        out (nn.Linear): Output projection
        sm_scale (float): Attention score scaling factor
        rope (RotaryEmbedding): Rotary position embedding module
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
            config (ModelConfig): Model configuration
            layer_idx (int): Index of this layer (used for sliding window pattern)
            device (torch.device): Device to place parameters on

        Note:
            Sliding window is only applied to even-numbered layers to balance
            local and global attention throughout the model.
        """
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Alternate sliding window: even layers have local attention, odd layers have full
        # This balances computational efficiency with long-range dependencies
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        # Sink tokens: learnable attention logits that prevent attention collapse
        # One per attention head (after GQA expansion)
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )

        # Pre-attention normalization
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Combined QKV projection
        # Output size: num_q_heads * head_dim + 2 * num_kv_heads * head_dim
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )

        # Output projection (maps attention output back to hidden size)
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )

        # Attention scaling factor: 1/sqrt(head_dim)
        # Prevents softmax saturation for large head dimensions
        self.sm_scale = 1 / math.sqrt(config.head_dim)

        # Rotary position embeddings with YaRN scaling for long context
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor, shape [num_tokens, hidden_size]

        Returns:
            torch.Tensor: Output tensor with same shape as input

        Process:
            1. Normalize input
            2. Project to Q, K, V
            3. Reshape for multi-head attention
            4. Apply RoPE to Q and K
            5. Compute scaled dot-product attention
            6. Project output and add residual
        """
        # Pre-normalization
        t = self.norm(x)

        # Project to combined QKV
        qkv = self.qkv(t)

        # Split into Q, K, V
        # Layout: [Q_heads * head_dim | K_heads * head_dim | V_heads * head_dim]
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        # Reshape for grouped-query attention
        # Q: [n_tokens, n_kv_heads, q_mult, head_dim] where q_mult = n_q_heads / n_kv_heads
        # K, V: [n_tokens, n_kv_heads, head_dim]
        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        # Apply rotary position embeddings to Q and K
        q, k = self.rope(q, k)

        # Compute attention with optional sliding window and sink tokens
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)

        # Project back to hidden dimension
        t = self.out(t)

        # Residual connection
        t = x + t

        return t


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    """
    SwiGLU activation function with clamping.

    SwiGLU (Swish-Gated Linear Unit) is a variant of GLU that uses the Swish
    activation function. It's been shown to improve performance in transformer
    models compared to standard activations like ReLU or GELU.

    Formula: SwiGLU(x) = Swish(x_glu) * (x_linear + 1)
    where Swish(x) = x * sigmoid(alpha * x)

    The input tensor is expected to have interleaved glu/linear values:
    [glu_0, linear_0, glu_1, linear_1, ...]

    Args:
        x (torch.Tensor): Input tensor with interleaved glu/linear values
        alpha (float): Scaling factor for sigmoid in Swish activation (default: 1.702)
        limit (float): Clamping limit to prevent numerical overflow (default: 7.0)

    Returns:
        torch.Tensor: Activated output, half the size of input in last dimension

    Note:
        - The +1 bias on the linear component is a learned optimization
        - Clamping prevents overflow in sigmoid and subsequent operations
        - Values are arranged for SwiGLU: even indices are gated, odd are linear
    """
    # Split interleaved values: even indices for GLU, odd for linear
    x_glu, x_linear = x[..., ::2], x[..., 1::2]

    # Clamp input values to prevent numerical overflow
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)

    # Compute Swish activation: x * sigmoid(alpha * x)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)

    # Multiply gated output by (linear + 1)
    # The +1 bias improves gradient flow and model performance
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    """
    Mixture of Experts (MoE) MLP block with sparse routing.

    This implements a sparse MoE layer where each token is processed by a
    subset of experts rather than all experts. This allows the model to have
    a large total capacity while keeping computational cost manageable.

    Architecture:
    1. Router (gate): Selects top-k experts for each token
    2. MLP1: First linear layer with SwiGLU activation (per expert)
    3. MLP2: Second linear layer (per expert)
    4. Combine: Weighted sum of expert outputs

    The MLP weights are sharded across multiple GPUs for distributed inference,
    with all-reduce synchronization after MLP2.

    Attributes:
        num_experts (int): Total number of experts in the layer
        experts_per_token (int): Number of experts activated per token (k in top-k)
        swiglu_limit (float): Clamping limit for SwiGLU activation
        world_size (int): Number of GPUs for distributed inference
        norm (RMSNorm): Pre-MLP normalization layer
        gate (nn.Linear): Router network for expert selection
        mlp1_weight (nn.Parameter): First MLP weights for all experts (sharded)
        mlp1_bias (nn.Parameter): First MLP biases for all experts (sharded)
        mlp2_weight (nn.Parameter): Second MLP weights for all experts (sharded)
        mlp2_bias (nn.Parameter): Second MLP biases for all experts (not sharded)
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        """
        Initialize MoE MLP block.

        Args:
            config (ModelConfig): Model configuration
            device (torch.device): Device to place parameters on

        Note:
            - Weights are sharded across world_size GPUs
            - MLP1 output is 2x intermediate_size to support SwiGLU (glu + linear)
            - intermediate_size must be divisible by world_size
        """
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Pre-MLP normalization
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Router network: maps hidden states to expert scores
        # Output: [batch, num_experts] logits
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )

        # Ensure intermediate size is evenly divisible across GPUs
        assert config.intermediate_size % self.world_size == 0

        # MLP1: hidden -> intermediate (sharded across GPUs)
        # Output is 2x for SwiGLU (glu component + linear component)
        # Shape: [num_experts, intermediate_size*2/world_size, hidden_size]
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

        # MLP2: intermediate -> hidden (sharded across GPUs)
        # Shape: [num_experts, hidden_size, intermediate_size/world_size]
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )

        # MLP2 bias is NOT sharded (same across all GPUs)
        # Shape: [num_experts, hidden_size]
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse MoE MLP to input tensor.

        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, hidden_size]

        Returns:
            torch.Tensor: Output tensor with same shape as input

        Process:
            1. Normalize input
            2. Route to top-k experts per token
            3. Apply expert MLPs in parallel
            4. Combine expert outputs with learned weights
            5. Add residual connection

        Note:
            - Only k experts are computed per token (sparse routing)
            - Weights are sharded across GPUs, synchronized with all_reduce
            - Expert outputs are combined using softmax-normalized routing weights
        """
        # Pre-normalization
        t = self.norm(x)

        # Compute routing scores for all experts
        g = self.gate(t)  # Shape: [batch, num_experts]

        # Select top-k experts per token (sparse routing)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)

        # Normalize expert weights using softmax (over selected experts only)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)

        # Get indices of selected experts
        expert_indices = experts.indices  # Shape: [batch, k]

        # MLP Layer 1: hidden -> intermediate with SwiGLU
        # Gather weights for selected experts
        mlp1_weight = self.mlp1_weight[expert_indices, ...]  # Shape: [batch, k, intermediate*2, hidden]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]      # Shape: [batch, k, intermediate*2]

        # Apply first MLP layer: weight @ input + bias
        # einsum "beck,bk->bec": b=batch, e=experts, c=channels, k=input_dim
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias

        # Apply SwiGLU activation
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP Layer 2: intermediate -> hidden
        # Gather weights for selected experts
        mlp2_weight = self.mlp2_weight[expert_indices, ...]  # Shape: [batch, k, hidden, intermediate]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]      # Shape: [batch, k, hidden]

        # Apply second MLP layer: weight @ activation
        # einsum "beck,bek->bec": matrix multiply for each batch/expert
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)

        # Synchronize across GPUs if running distributed
        # Each GPU has a shard of the intermediate dimension, so we sum the results
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        # Add bias (after all_reduce since bias is not sharded)
        t += mlp2_bias

        # Combine expert outputs using routing weights
        # einsum "bec,be->bc": weighted sum over experts dimension
        t = torch.einsum("bec,be->bc", t, expert_weights)

        # Residual connection
        return x + t


class TransformerBlock(torch.nn.Module):
    """
    Single transformer block combining attention and MLP.

    This implements one layer of the transformer, consisting of:
    1. Multi-head attention with RoPE and optional sliding window
    2. Mixture of Experts MLP with sparse routing

    Both components use pre-normalization and residual connections.

    Attributes:
        layer_idx (int): Index of this layer in the model
        attn (AttentionBlock): Attention sub-layer
        mlp (MLPBlock): MoE MLP sub-layer
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
            config (ModelConfig): Model configuration
            layer_idx (int): Layer index (used for sliding window alternation)
            device (torch.device): Device to place parameters on
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block to input.

        Args:
            x (torch.Tensor): Input tensor, shape [num_tokens, hidden_size]

        Returns:
            torch.Tensor: Output tensor with same shape as input

        Note:
            Both attention and MLP include their own residual connections
        """
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    """
    Complete transformer language model.

    This implements a decoder-only transformer for autoregressive language modeling.
    The architecture includes:
    - Token embedding layer
    - Stack of transformer blocks (attention + MoE MLP)
    - Final normalization
    - Output projection to vocabulary (unembedding)

    The model supports distributed inference with tensor parallelism across
    multiple GPUs for the MoE layers.

    Attributes:
        embedding (nn.Embedding): Input token embedding
        block (nn.ModuleList): Stack of transformer blocks
        norm (RMSNorm): Final normalization before output
        unembedding (nn.Linear): Output projection to vocabulary logits
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        """
        Initialize transformer model.

        Args:
            config (ModelConfig): Model configuration
            device (torch.device): Device to place parameters on
        """
        super().__init__()

        # Token embedding: maps token IDs to hidden representations
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

        # Final normalization before output projection
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Output projection: maps hidden states to vocabulary logits
        # No bias term (common in modern LLMs)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            x (torch.Tensor): Input token IDs, shape [num_tokens]

        Returns:
            torch.Tensor: Vocabulary logits, shape [num_tokens, vocab_size]

        Note:
            This performs a single forward pass. For generation, use TokenGenerator
            which handles autoregressive sampling.
        """
        # Embed input tokens
        x = self.embedding(x)

        # Apply transformer blocks sequentially
        for block in self.block:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary logits
        x = self.unembedding(x)

        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        """
        Load a trained model from a checkpoint directory.

        This static method constructs a Transformer model and loads its weights
        from a checkpoint. It handles:
        - Reading model configuration from config.json
        - Creating the model architecture
        - Loading and dequantizing weights (MXFP4 -> bfloat16)
        - Sharding MoE weights across multiple GPUs for distributed inference

        Args:
            path (str): Path to checkpoint directory containing config.json and .safetensors files
            device (str | torch.device): Device to load the model on (default: "cuda")

        Returns:
            Transformer: Loaded model in evaluation mode

        Note:
            - MoE weights are automatically sharded across available GPUs
            - The intermediate dimension must be divisible by the number of GPUs
            - For efficiency, sharding happens after MXFP4 dequantization
        """
        # Convert device string to device object if needed
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # Load model configuration from checkpoint
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        # Create model architecture
        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()  # Set to evaluation mode (disables dropout, etc.)

        # Determine distributed setup for weight sharding
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate_size = config.intermediate_size // world_size

        # Initialize checkpoint loader
        checkpoint = Checkpoint(path, device)

        # Load all parameters
        for name, param in model.named_parameters():
            # Load tensor (handles both regular and MXFP4 formats)
            loaded_tensor = checkpoint.get(name)

            # Shard MoE weights across GPUs
            # Note: It would be more efficient to shard before dequantizing MXFP4,
            # but we do it after for code simplicity
            if "mlp1" in name:  # Both mlp1_weight and mlp1_bias
                # MLP1 output dimension is 2x intermediate_size (for SwiGLU)
                # Shard along the intermediate dimension
                loaded_tensor = loaded_tensor[
                    :,
                    my_rank * 2
                    * per_rank_intermediate_size : (my_rank + 1) * 2
                    * per_rank_intermediate_size,
                    ...,
                ]
            elif "mlp2_weight" in name:  # Only mlp2_weight (not bias)
                # Shard along the intermediate dimension (input to MLP2)
                loaded_tensor = loaded_tensor[
                    ...,
                    my_rank
                    * per_rank_intermediate_size : (my_rank + 1)
                    * per_rank_intermediate_size,
                ]

            # Copy loaded weights into model parameters
            try:
                param.data.copy_(loaded_tensor)
            except Exception as e:
                # Print debug info if shape mismatch occurs
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        return model


class TokenGenerator:
    """
    Autoregressive token generator for text generation.

    This class provides a high-level interface for generating text with the
    transformer model. It handles:
    - Loading the model from a checkpoint
    - Autoregressive generation (one token at a time)
    - Temperature-based sampling
    - Greedy decoding (temperature=0)
    - Stop token detection
    - Optional log probability computation

    The generator yields tokens one at a time, making it suitable for
    streaming applications.

    Attributes:
        device (torch.device): Device where the model is loaded
        model (Transformer): The loaded transformer model
    """

    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        """
        Initialize the token generator by loading a model from checkpoint.

        Args:
            checkpoint (str): Path to checkpoint directory
            device (torch.device): Device to load the model on

        Note:
            The @torch.inference_mode() decorator disables gradient computation
            for better performance during inference.
        """
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        """
        Generate tokens autoregressively from a prompt.

        This generator yields tokens one at a time, allowing for streaming
        generation. It continues until hitting a stop token or reaching max_tokens.

        Args:
            prompt_tokens (list[int]): Initial token IDs to condition generation
            stop_tokens (list[int]): Token IDs that signal end of generation (e.g., EOS)
            temperature (float): Sampling temperature (default: 1.0)
                                - 0.0: greedy decoding (always pick highest probability)
                                - 1.0: sample from the model's distribution
                                - >1.0: more random (flatter distribution)
                                - <1.0: more confident (sharper distribution)
            max_tokens (int): Maximum number of tokens to generate (0 = unlimited)
            return_logprobs (bool): If True, yield (token, logprob) tuples instead of just tokens

        Yields:
            int | tuple[int, float]: Generated token ID, or (token_id, logprob) if return_logprobs=True

        Example:
            >>> generator = TokenGenerator("path/to/checkpoint", device)
            >>> for token in generator.generate([1, 2, 3], stop_tokens=[0], temperature=0.7):
            ...     print(token)

        Note:
            - The entire prompt is re-processed at each step (no KV cache)
            - This is suitable for small-scale inference but not optimal for production
            - Temperature=0 is deterministic, temperature>0 is stochastic
        """
        # Start with the prompt tokens
        tokens = list(prompt_tokens)
        num_generated_tokens = 0

        # Continue generating until we hit a stopping condition
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            # Run the model on all tokens so far (autoregressive)
            # We only need the logits for the last token
            logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]

            # Sample or select the next token based on temperature
            if temperature == 0.0:
                # Greedy decoding: always pick the most likely token
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                # Sampling: apply temperature and sample from distribution
                # Lower temperature -> more confident, higher -> more random
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            # Add the predicted token to our sequence
            tokens.append(predicted_token)
            num_generated_tokens += 1

            # Yield the token (optionally with its log probability)
            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            # Check if we've hit a stop token (e.g., end of sentence)
            if predicted_token in stop_tokens:
                break
