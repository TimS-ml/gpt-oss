/*
 * gather_and_accumulate.metal
 *
 * Metal kernels for gathering and accumulating expert outputs in Mixture of Experts (MoE) models.
 * Performs indirect memory gathering based on expert routing decisions and accumulates weighted
 * expert outputs back to token positions.
 *
 * This operation is critical in MoE inference where:
 * 1. Tokens are dynamically routed to different experts
 * 2. Each expert processes its assigned tokens
 * 3. Expert outputs must be gathered and combined back to original token positions
 *
 * Key optimizations:
 * - Direct 2D grid mapping for parallelism (tokens x embedding_dim)
 * - Vectorized float4 operations for memory bandwidth
 * - Fused multiply-add (FMA) for efficient accumulation
 * - Restrict qualifiers to enable compiler optimizations
 *
 * Performance considerations:
 * - Memory access is irregular due to expert routing (potential cache misses)
 * - Each thread performs 4 random reads from expert outputs
 * - TODO: Future optimization could amortize metadata reads across multiple float4s
 */

#include <internal/kernel-args.h>
#include <metal_integer>
#include <metal_math>
#include <metal_stdlib>

/**
 * gptoss_f32_gather_and_accumulate_e4
 *
 * Gathers expert outputs and accumulates them back to token positions with learned weights.
 *
 * For each token that was processed by 4 experts, this kernel:
 * 1. Reads expert predictions (which experts and their scores)
 * 2. Computes the memory locations of each expert's output for this token
 * 3. Gathers the outputs from 4 different expert buffers
 * 4. Accumulates weighted results: output[token] += sum(expert_output[i] * score[i])
 *
 * Thread organization:
 * - 2D grid: X dimension = embedding dimension (in float4 units), Y dimension = tokens
 * - Each thread handles one float4 element (4 floats) for one token
 * - No threadgroup cooperation (future optimization opportunity)
 *
 * Memory layout:
 * - in: Expert outputs in expert-specific order [total_expert_assignments * D]
 * - expert_predictions: Which experts processed each token and their scores [T * k]
 * - expert_offsets: Starting position of each expert's output buffer [num_experts]
 * - intra_expert_offsets: Position within expert's buffer for each assignment [T * k]
 * - out: Accumulated output in token order [T * D]
 *
 * Parameters:
 * @param args                Configuration (tokens, active_experts_per_token, token_stride)
 * @param in                  Expert outputs in expert-specific layout
 * @param expert_predictions  Expert IDs and scores for each token
 * @param expert_offsets      Starting offset for each expert in the input buffer
 * @param intra_expert_offsets  Offset within expert's buffer for each assignment
 * @param out                 Output buffer for accumulated results (read-modify-write)
 * @param gid                 Thread position in 2D grid (X: embedding dim, Y: token)
 *
 * TODO: Optimize by having each thread process multiple float4s to amortize metadata reads
 */
kernel void gptoss_f32_gather_and_accumulate_e4(
    constant gptoss_gather_args& args [[ buffer(0) ]],
    const device float* in [[ buffer(1) ]],
    const device gptoss_expert_prediction* __restrict__ expert_predictions [[ buffer(2) ]],
    const device uint* expert_offsets [[ buffer(3) ]],
    const device uint* intra_expert_offsets [[ buffer(4) ]],
    device float* out [[ buffer(5) ]],
    uint3 gid [[thread_position_in_grid]]) 
{
    // Load configuration parameters
    const uint T = args.tokens;                         // Number of tokens in batch
    const uint k = args.active_experts_per_token;       // Number of experts per token (should be 4)
    const uint D = args.token_stride;                   // Embedding dimension

    // Compile-time assertions for algorithm requirements
    assert((D & 3u) == 0);  // D must be divisible by 4 (for float4 vectorization)
    assert(k == 4);         // This kernel is specialized for 4 experts

    // Calculate token index from 2D grid Y coordinate
    const uint row = gid.y;
    if (row >= T) {
        return;  // Out of bounds check for token dimension
    }

    // Calculate embedding dimension index from 2D grid X coordinate
    const uint col_vec4 = gid.x;        // Index in float4 units
    const uint col = col_vec4 * 4u;     // Convert to float units
    if (col >= D) {
        return;  // Out of bounds check for embedding dimension
    }

    // Set up output pointer (read-modify-write: we accumulate into existing values)
    device float4* dst4 = reinterpret_cast<device float4*>(out + row * D + col);

    // Load expert routing metadata for this token
    // Each token has k expert predictions (expert IDs and scores)
    const uint base = row * k;
    const gptoss_expert_prediction expert0 = expert_predictions[base];
    const gptoss_expert_prediction expert1 = expert_predictions[base + 1];
    const gptoss_expert_prediction expert2 = expert_predictions[base + 2];
    const gptoss_expert_prediction expert3 = expert_predictions[base + 3];

    // Extract expert IDs (which expert processed this token)
    const uint expert0_id = expert0.expert_id;
    const uint expert1_id = expert1.expert_id;
    const uint expert2_id = expert2.expert_id;
    const uint expert3_id = expert3.expert_id;

    // Extract expert scores (weights for combining expert outputs)
    const float scale0 = expert0.score;
    const float scale1 = expert1.score;
    const float scale2 = expert2.score;
    const float scale3 = expert3.score;

    // Load all 4 intra-expert offsets as a single uint4 for efficiency
    // These tell us the position within each expert's output buffer
    const uint4 current_intra_expert_offsets =
        *reinterpret_cast<const device uint4*>(&intra_expert_offsets[base]);

    // Calculate actual row indices in the expert output buffer
    // expert_offsets[id] gives the starting row for expert 'id'
    // intra_expert_offset gives the row within that expert's section
    const uint r0 = expert_offsets[expert0_id] + current_intra_expert_offsets.x;
    const uint r1 = expert_offsets[expert1_id] + current_intra_expert_offsets.y;
    const uint r2 = expert_offsets[expert2_id] + current_intra_expert_offsets.z;
    const uint r3 = expert_offsets[expert3_id] + current_intra_expert_offsets.w;

    // Set up input pointers for each expert's output
    // These are indirect/gather reads - each expert may be at a different location
    const device float4* src0 =
        reinterpret_cast<const device float4*>(in + r0 * D + col);
    const device float4* src1 =
        reinterpret_cast<const device float4*>(in + r1 * D + col);
    const device float4* src2 =
        reinterpret_cast<const device float4*>(in + r2 * D + col);
    const device float4* src3 =
        reinterpret_cast<const device float4*>(in + r3 * D + col);

    // Accumulate weighted expert outputs
    // Start with current output value (may have been written by previous operations)
    float4 acc = *dst4;
    // Add each expert's contribution: acc += expert_output * expert_score
    acc = metal::fma(*src0, scale0, acc);  // FMA: acc = src0 * scale0 + acc
    acc = metal::fma(*src1, scale1, acc);
    acc = metal::fma(*src2, scale2, acc);
    acc = metal::fma(*src3, scale3, acc);
    // Write accumulated result back to output
    *dst4 = acc;
}