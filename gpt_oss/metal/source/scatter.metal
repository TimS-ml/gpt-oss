/*
 * scatter.metal
 *
 * Metal kernels for scattering token data to expert-specific buffers in MoE models.
 * Performs the inverse operation of gather: distributes tokens from sequential layout
 * to expert-grouped layout based on routing decisions.
 *
 * This is a critical preprocessing step for MoE inference:
 * 1. Tokens arrive in sequential order (batch order)
 * 2. Routing network determines which experts process each token
 * 3. Scatter kernel reorganizes tokens into expert-specific groups
 * 4. Each expert processes its assigned tokens in parallel
 * 5. Gather kernel reassembles results back to token order
 *
 * Key challenges:
 * - Irregular memory access (scatter pattern is data-dependent)
 * - Each token writes to multiple locations (top-k experts)
 * - Memory bandwidth critical (reading input, writing to k outputs)
 *
 * Optimizations:
 * - Vectorized float4 operations for bandwidth
 * - Direct 2D grid mapping for parallelism
 * - Minimal computation (mostly memory movement)
 * - TODO: Amortize metadata reads across multiple float4s
 */

#include <internal/kernel-args.h>
#include <metal_integer>
#include <metal_math>
#include <metal_stdlib>

/**
 * gptoss_f32_scatter_e4
 *
 * Scatters tokens to 4 expert buffers based on routing decisions.
 *
 * For each token assigned to 4 experts (top-4 routing):
 * 1. Read expert IDs and offsets for this token
 * 2. Read token's embedding vector
 * 3. Write same vector to all 4 expert buffers at computed positions
 *
 * Memory layout:
 * - Input: [tokens * embedding_dim] - Sequential token order
 * - Output: [total_expert_assignments * embedding_dim] - Expert-grouped layout
 *   where total_expert_assignments = sum of tokens assigned to each expert
 *
 * Thread organization:
 * - 2D grid: X = embedding dimension (float4 units), Y = tokens
 * - Each thread handles one float4 (4 floats) for one token
 * - Broadcasts to 4 expert locations (one per assigned expert)
 *
 * Performance notes:
 * - Scatter is memory-intensive: 1 read + 4 writes per thread
 * - Memory accesses are irregular (expert routing determines addresses)
 * - Potential for write conflicts minimized by expert_offsets design
 * - TODO: Process multiple float4s per thread to amortize metadata reads
 *
 * Parameters:
 * @param args                 Configuration (tokens, active_experts_per_token, token_stride)
 * @param in                   Input tokens in sequential order [tokens * embedding_dim]
 * @param expert_predictions   Expert IDs for each token [tokens * k]
 * @param expert_offsets       Starting offset for each expert in output buffer
 * @param intra_expert_offsets Position within expert's buffer for each assignment
 * @param out                  Output buffer in expert-grouped layout
 * @param gid                  2D thread position (X: embedding dim, Y: token index)
 */
kernel void gptoss_f32_scatter_e4(
    constant gptoss_scatter_args& args [[ buffer(0) ]],
    const device float* in [[ buffer(1) ]],
    const device gptoss_expert_prediction* __restrict__ expert_predictions [[ buffer(2) ]],
    const device uint* __restrict__ expert_offsets [[ buffer(3) ]],
    const device uint* __restrict__ intra_expert_offsets [[ buffer(4) ]],
    device float* out [[ buffer(5) ]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint total_tokens = args.tokens;
    const uint active_experts_per_token = args.active_experts_per_token;
    const uint embedding_dim = args.token_stride;

    // Algorithm requirements
    assert(embedding_dim % 4 == 0);  // Must be divisible by 4 for float4 vectorization
    assert(active_experts_per_token == 4);  // This kernel is specialized for top-4

    // Calculate token index from 2D grid Y coordinate
    const uint row_in = gid.y;
    if (row_in >= total_tokens) {
        return;  // Out of bounds check for tokens
    }

    // Calculate embedding dimension index from 2D grid X coordinate
    // Consecutive threads in a threadgroup process consecutive columns (coalescing)
    const uint col_in_vec4 = gid.x;        // Index in float4 units
    const uint col_in = col_in_vec4 * 4;   // Convert to float units
    if (col_in >= embedding_dim) {
        return;  // Out of bounds check for embedding dimension
    }

    // Set up pointer to input data (one float4 vector for this token)
    const device float4* src4 =
        reinterpret_cast<const device float4*>(in + row_in * embedding_dim + col_in);

    // Load expert routing information for this token
    // Each token is assigned to 4 experts (top-4 routing)
    const uint base = row_in * active_experts_per_token;

    // Get expert IDs (which experts will process this token)
    const uint expert0_id = expert_predictions[base].expert_id;
    const uint expert1_id = expert_predictions[base + 1].expert_id;
    const uint expert2_id = expert_predictions[base + 2].expert_id;
    const uint expert3_id = expert_predictions[base + 3].expert_id;

    // Get starting offsets for each expert's buffer section
    const uint expert0_offset = expert_offsets[expert0_id];
    const uint expert1_offset = expert_offsets[expert1_id];
    const uint expert2_offset = expert_offsets[expert2_id];
    const uint expert3_offset = expert_offsets[expert3_id];

    // Get position within each expert's buffer for this token
    const uint expert0_intra_expert_offset = intra_expert_offsets[base];
    const uint expert1_intra_expert_offset = intra_expert_offsets[base + 1];
    const uint expert2_intra_expert_offset = intra_expert_offsets[base + 2];
    const uint expert3_intra_expert_offset = intra_expert_offsets[base + 3];

    // Calculate output pointers for all 4 expert destinations
    // Each expert's section starts at expert_offset, and this token goes at intra_expert_offset within that
    device float4* dst4_0 = reinterpret_cast<device float4*>(
        out + (expert0_offset + expert0_intra_expert_offset) * embedding_dim + col_in);
    device float4* dst4_1 = reinterpret_cast<device float4*>(
        out + (expert1_offset + expert1_intra_expert_offset) * embedding_dim + col_in);
    device float4* dst4_2 = reinterpret_cast<device float4*>(
        out + (expert2_offset + expert2_intra_expert_offset) * embedding_dim + col_in);
    device float4* dst4_3 = reinterpret_cast<device float4*>(
        out + (expert3_offset + expert3_intra_expert_offset) * embedding_dim + col_in);

    // Read once, write to all 4 expert buffers (broadcast pattern)
    const float4 data = *src4;
    *dst4_0 = data;
    *dst4_1 = data;
    *dst4_2 = data;
    *dst4_3 = data;
}
