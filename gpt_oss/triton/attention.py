"""FlashAttention w/support for learned sinks and banded attention.

This is an expanded version of the Flash Attention v2 implementation (see https://tridao.me/publications/flash2/flash2.pdf)
which can be found at https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.

This version has been extended to support banded attention and learned attention sinks.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor



@triton.jit
def _attn_fwd(
    Q,  # Query tensor descriptor [batch, heads, seq_len, head_dim]
    K,  # Key tensor descriptor [batch, heads, seq_len, head_dim]
    V,  # Value tensor descriptor [batch, heads, seq_len, head_dim]
    Sinks,  # Learned attention sinks (biases) per head [num_heads]
    sm_scale,  # Softmax scaling factor (typically 1/sqrt(head_dim))
    M,  # Output tensor for max values (used in backward pass) [batch*heads, seq_len]
    Out,  # Output tensor descriptor [batch, heads, seq_len, head_dim]
    Start_q,  # Starting position in the query sequence (for KV cache)
    Z,  # Batch size
    H,  # Number of attention heads
    N_Q_CTX,  # Query context length (padded to BLOCK_M multiple)
    N_KV_CTX,  # Key/Value context length
    HEAD_DIM: tl.constexpr,  # Dimension of each attention head (must be in {16,32,64,128,256})
    BLOCK_M: tl.constexpr,  # Block size for query dimension (tiling parameter)
    BLOCK_N: tl.constexpr,  # Block size for key dimension (tiling parameter)
    BANDWIDTH: tl.constexpr,  # Sliding window bandwidth (0 = full attention)
):
    """
    Triton kernel for Flash Attention forward pass with learned sinks and banded attention.

    This kernel implements the Flash Attention algorithm, which computes attention in a
    memory-efficient way by tiling the computation into blocks and fusing operations.
    The key optimization is computing softmax in an online manner without materializing
    the full attention matrix, which would require O(N^2) memory.

    Extensions beyond standard Flash Attention:
    1. Learned sinks: Learned bias terms per head that act as additional attention weights
    2. Banded/sliding window attention: Only attend to tokens within a sliding window

    Memory access pattern:
    - Queries are loaded once per block and reused across all key blocks (BLOCK_M x HEAD_DIM)
    - Keys and values are streamed through in BLOCK_N-sized chunks
    - This minimizes HBM (High Bandwidth Memory) accesses, keeping most data in SRAM

    Performance considerations:
    - BLOCK_M and BLOCK_N control the tiling granularity and SRAM usage
    - BLOCK_N must be <= HEAD_DIM for efficient matrix multiplication
    - Typical values: BLOCK_M=64, BLOCK_N=64 for good occupancy and memory efficiency
    """
    # Ensure BLOCK_N is small enough to fit efficiently in shared memory
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # Load the starting position of the query sequence (important for KV cache)
    start_q = tl.load(Start_q).to(tl.int32)

    # Compute program IDs for parallel execution across blocks
    # program_id(0) indexes the query block (M dimension)
    # program_id(1) indexes the batch*head combination
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H  # Batch index
    off_h = off_hz % H   # Head index

    # Load attention sink for this head
    # Sinks are learned biases that provide a fixed attention "reservoir"
    # They help stabilize attention distributions and can act as learned no-op tokens
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0

    # Initialize block offsets for tiling
    # offs_m: query token positions within this block [BLOCK_M]
    # offs_n: key token positions within current key block [BLOCK_N]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize running statistics for online softmax (Flash Attention trick)
    # m_i: running maximum of attention logits (for numerical stability)
    # l_i: running sum of exponentials (denominator of softmax)
    # acc: running sum of attention-weighted values (numerator)
    # Initialize m_i with sink value to incorporate learned sinks
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load scale factor for attention (typically 1/sqrt(d_k) for stability)
    qk_scale = sm_scale

    # Load the query block for this program instance
    # Shape: [BLOCK_M, HEAD_DIM] - this stays in registers/SRAM for reuse
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    # Determine the range of key blocks to process based on causal masking and sliding window
    if BANDWIDTH:
        # For sliding window attention: only attend to keys within BANDWIDTH tokens
        # lo: earliest key position = max(start_q, current_query_pos - BANDWIDTH)
        # hi: latest key position = current_query_pos (causal masking)
        lo, hi = tl.maximum(start_q, start_q + start_m * BLOCK_M - BANDWIDTH), start_q + (start_m + 1) * BLOCK_M
    else:
        # For full causal attention: attend to all previous keys
        lo, hi = start_q, start_q + (start_m + 1) * BLOCK_M

    # Main loop: iterate over key/value blocks (streaming from HBM)
    # This is the core of Flash Attention - we process K/V in chunks to save memory
    for start_n in range(lo, hi, BLOCK_N):
        # Ensure start_n is aligned to BLOCK_N for memory coalescing
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Compute causal mask: queries can only attend to earlier keys
        # Shape: [BLOCK_M, BLOCK_N] - True where attention should be masked out
        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            # Additional masking for sliding window: mask keys that are too old
            # This creates a banded attention pattern for local attention
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        # Load key block and transpose for matrix multiplication
        # K: [BLOCK_N, HEAD_DIM] -> K.T: [HEAD_DIM, BLOCK_N]
        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T

        # Compute attention scores: Q @ K.T
        # Shape: [BLOCK_M, BLOCK_N]
        # allow_tf32=False ensures full fp32 precision for numerical stability
        qk = tl.dot(q, k, allow_tf32=False)

        # Scale scores and apply mask (masked positions get very negative values)
        # -1.0e6 is large enough to make softmax output ~0 after exp
        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)

        # Online softmax update (Flash Attention core algorithm)
        # Compute new maximum for numerical stability
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        # Subtract max for numerical stability (standard softmax trick)
        qk -= m_ij[:, None]

        # Compute attention probabilities for current block
        p = tl.math.exp(qk)

        # Compute rescaling factor for previous accumulator
        # When we see a new max, we need to rescale previous softmax terms
        alpha = tl.math.exp(m_i - m_ij)

        # Sum of attention probabilities for this key block
        l_ij = tl.sum(p, 1)

        # Rescale previous accumulator based on new maximum
        # This is the key trick that makes Flash Attention work correctly
        acc = acc * alpha[:, None]

        # Load value block and convert to float32 for accumulation
        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = v.to(tl.float32)

        # Accumulate attention-weighted values: acc += P @ V
        # The third argument allows in-place accumulation
        acc = tl.dot(p, v, acc, allow_tf32=False)

        # Update running sum of exponentials (softmax denominator)
        l_i = l_i * alpha + l_ij

        # Update running maximum
        m_i = m_ij

    # Final normalization step: incorporate learned sinks
    # Sinks contribute to the softmax denominator like an additional attention term
    sink = tl.math.exp(sink - m_i)
    z = l_i + sink  # Total normalization factor (including sinks)

    # Normalize accumulated values by total softmax denominator
    acc = acc / z[:, None]

    # Store log-sum-exp for backward pass (used in gradient computation)
    m_i += tl.math.log(l_i)
    m_ptrs = M + off_hz * N_Q_CTX + offs_m
    tl.store(m_ptrs, m_i)

    # Convert output back to original dtype and store to HBM
    # Add batch and head dimensions back for proper indexing
    acc = acc.to(Out.dtype)[None, None, :, :]
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)


class _attention(torch.autograd.Function):
    """
    PyTorch autograd Function wrapper for the Triton Flash Attention kernel.

    This class provides the interface between PyTorch's autograd system and the
    custom Triton kernel. It handles:
    - Input tensor reshaping and padding for optimal Triton performance
    - Launching the Triton kernel with appropriate grid dimensions
    - Managing grouped query attention (GQA) by repeating KV heads
    - Output tensor reshaping back to expected format

    The forward pass uses the Triton kernel for efficient computation, while
    the backward pass is not implemented (forward-only mode).
    """

    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        """
        Forward pass for Flash Attention with learned sinks and sliding window.

        Args:
            ctx: Autograd context for saving tensors needed in backward pass
            q: Query tensor [batch, seq_len, n_kv_heads, repeat_kv, head_dim]
                Shape includes repeat_kv for grouped query attention (GQA)
            k: Key tensor [batch, kv_seq_len, n_kv_heads, head_dim]
            v: Value tensor [batch, kv_seq_len, n_kv_heads, head_dim]
            sinks: Learned attention sinks [num_heads] or None
                Learned biases that provide fixed attention weights
            sm_scale: Softmax scaling factor, typically 1/sqrt(head_dim)
            bandwidth: Sliding window bandwidth (0 or None for full attention)
                Controls local attention pattern
            start_q: Starting position in sequence [1] (scalar tensor)
                Used with KV cache to track current position

        Returns:
            Output tensor [batch, seq_len, num_heads * head_dim]
                Attention output reshaped to match input format

        Note:
            - Tensors are padded to multiples of BLOCK_M and BLOCK_N for efficiency
            - KV heads are repeated for grouped query attention (GQA/MQA support)
            - The implementation is causal by default (queries attend only to past)
        """
        # Validate inputs
        assert len(start_q) == 1  # start_q should be a single-element tensor
        bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM_Q = q.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K = k.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_V = v.shape

        # Calculate total number of attention heads (for grouped query attention)
        # n_kv_heads = number of key/value heads
        # repeat_kv = how many query heads per KV head (e.g., 8 for GQA)
        # n_heads = total query heads (e.g., 64 with 8 KV heads and repeat_kv=8)
        n_heads = n_kv_heads * repeat_kv

        # Reshape query to combine repeat_kv into head dimension
        q = q.view(bs, n_ctx, n_heads, HEAD_DIM_Q)
        k = k.view(bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K)

        # Ensure all head dimensions match and are supported
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}  # Triton kernel optimized for these sizes

        # Transpose to [batch, heads, seq_len, head_dim] for kernel compatibility
        # This layout is more efficient for the Triton kernel's memory access pattern
        q = q.transpose(1, 2).contiguous()

        # Repeat KV heads to match query heads for grouped query attention
        # This expands n_kv_heads to n_heads by repeating each KV head repeat_kv times
        # E.g., with 8 KV heads and repeat_kv=8, we get 64 total heads
        k = k.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()
        v = v.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()

        # Define block sizes for tiling
        # These control the granularity of the Flash Attention computation
        # BLOCK_M: number of query tokens processed per block
        # BLOCK_N: number of key tokens processed per block
        # 64x64 is empirically good for most GPUs (balances occupancy and memory)
        BLOCK_M = 64
        BLOCK_N = 64

        # Pad tensors to multiples of block sizes for efficient kernel execution
        # This ensures all blocks are full-sized, avoiding edge case handling in kernel
        m_pad_size = BLOCK_M - n_ctx % BLOCK_M if n_ctx % BLOCK_M != 0 else 0
        q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))  # Pad sequence dimension

        n_pad_size = BLOCK_N - n_kv_ctx % BLOCK_N if n_kv_ctx % BLOCK_N != 0 else 0
        k = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))  # Pad KV sequence dimension
        v = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        # Allocate output tensors
        o = torch.empty_like(q)  # Attention output
        M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)  # Max values for backward

        # Define kernel launch grid
        # grid[0]: number of query blocks (each handles BLOCK_M queries)
        # grid[1]: number of batch*head combinations (each kernel instance handles one)
        # grid[2]: always 1 (not used in this kernel)
        grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)

        # Launch Triton kernel
        # TensorDescriptor wraps tensors with block size information for efficient loading
        # The descriptor specifies how the kernel should tile and access the tensor
        _attn_fwd[grid](
            TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, HEAD_DIM_K]),  # [batch, heads, BLOCK_M, head_dim]
            TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, HEAD_DIM_K]),  # [batch, heads, BLOCK_N, head_dim]
            TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, HEAD_DIM_K]),  # [batch, heads, BLOCK_N, head_dim]
            sinks,
            sm_scale,
            M,
            TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, HEAD_DIM_K]),  # [batch, heads, BLOCK_M, head_dim]
            start_q,
            q.shape[0],  # Z: batch size
            q.shape[1],  # H: number of heads
            N_Q_CTX=n_ctx + m_pad_size,
            N_KV_CTX=n_kv_ctx,
            HEAD_DIM=HEAD_DIM_K,
            BANDWIDTH=bandwidth,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        # Save tensors for backward pass (though backward is not implemented)
        ctx.save_for_backward(q, k, v, sinks, o, M, start_q)
        ctx.sm_scale = sm_scale
        ctx.bandwidth = bandwidth

        # Remove padding and transpose back to [batch, seq_len, heads, head_dim]
        o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous()

        # Flatten head dimensions to [batch, seq_len, heads * head_dim]
        o = o.view(bs, n_ctx, n_heads * HEAD_DIM_V)
        return o


attention = _attention.apply


def attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
):
    """
    Reference implementation of attention with learned sinks and sliding window.

    This is a standard PyTorch implementation used for testing and validation
    of the Triton kernel. It computes attention using standard operations without
    the memory optimizations of Flash Attention.

    Unlike the Triton kernel which uses tiling and online softmax, this reference
    implementation materializes the full attention matrix in memory. This is slower
    and uses more memory (O(N^2)) but is easier to verify for correctness.

    Args:
        query: Query tensor [batch, num_queries, n_kv_heads, repeat_kv, head_dim]
        key: Key tensor [batch, num_keys, n_kv_heads, head_dim]
        value: Value tensor [batch, num_keys, n_kv_heads, head_dim]
        sinks: Learned attention sinks [num_heads]
            Additional learned bias terms added to attention weights
        sm_scale: Softmax scaling factor (default: 0.125 = 1/sqrt(64))
        sliding_window: Size of sliding attention window, or None for full attention
            If set, only attend to tokens within this window
        start_q: Starting position of queries in the full sequence
            Used with KV cache to properly align positions

    Returns:
        Attention output [batch, num_queries, num_heads * head_dim]

    Note:
        This function is primarily for testing. For production use, the Triton
        kernel (attention function) is much faster and more memory efficient.
    """
    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape

    # Reshape sinks to broadcast correctly with attention logits
    # Shape: [1, n_kv_heads, repeat_kv, 1, 1] for broadcasting
    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()

    # Add dimension for key/value groups (for grouped query attention)
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    # Create position tensors for causal masking
    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q

    # Create causal mask: queries can only attend to earlier or equal positions
    # True positions will be masked out (set to -inf before softmax)
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    # Apply sliding window mask if specified
    if sliding_window:
        # Mask out keys that are too far in the past (outside the window)
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    # Compute attention logits: Q @ K^T, scaled
    # einsum notation: b=batch, q=query_pos, k=key_pos, h=kv_heads, m=groups, d=head_dim
    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale

    # Apply causal and sliding window masks
    logits = logits + mask[None, None, None, :, :]

    # Compute softmax with learned sinks (numerically stable version)
    # Find maximum between logits and sinks for numerical stability
    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)

    # Compute exponentials (subtract max for numerical stability)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)

    # Normalize: divide by sum of all attention weights (including sinks)
    # Sinks act as additional attention weight that's always present
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    # Apply attention scores to values: scores @ V
    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    # Reshape output to [batch, queries, total_embedding_dim]
    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups * head_dim).bfloat16()
    return output


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_queries", [1, 128])
@pytest.mark.parametrize("num_keys", [128, 32])
@pytest.mark.parametrize("num_key_value_heads", [8])
@pytest.mark.parametrize("num_key_value_groups", [8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("sm_scale", [0.125])
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("start_q", [0, 5])
def test_eq(batch_size, num_queries, num_keys, num_key_value_heads, num_key_value_groups, head_dim, sm_scale, sliding_window, start_q):
    """
    Test that Triton Flash Attention kernel produces the same results as reference implementation.

    This comprehensive test validates the Triton kernel against a standard PyTorch
    implementation across various configurations including:
    - Different batch sizes and sequence lengths
    - Grouped query attention (GQA) configurations
    - Sliding window attention patterns
    - Different starting positions (for KV cache testing)

    The test ensures numerical correctness of the optimized Triton kernel while
    maintaining all the features: learned sinks, causal masking, and sliding windows.

    Args:
        batch_size: Number of sequences in batch (1 or 2)
        num_queries: Number of query tokens (1 or 128)
        num_keys: Number of key/value tokens (32 or 128)
        num_key_value_heads: Number of KV heads for grouped attention (8)
        num_key_value_groups: Number of query groups per KV head (8)
            Total attention heads = num_key_value_heads * num_key_value_groups
        head_dim: Dimension of each attention head (64)
        sm_scale: Softmax scaling factor (0.125 = 1/sqrt(64))
        sliding_window: Sliding window size (None for full attention, 128 for local)
        start_q: Starting position in sequence (0 or 5, tests KV cache behavior)
    """
    if num_queries > num_keys:
        pytest.skip("too many queries")

    q = torch.randn(batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim).bfloat16().cuda()
    k = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    v = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    sinks = torch.randn(num_key_value_heads * num_key_value_groups).bfloat16().cuda()

    start_q = torch.tensor([start_q], dtype=torch.int32).cuda()

    o1 = attention(q, k, v, sinks, sm_scale, sliding_window, start_q)
    o2 = attention_ref(q, k, v, sinks, sm_scale, sliding_window, start_q)

    torch.testing.assert_close(o1, o2)
