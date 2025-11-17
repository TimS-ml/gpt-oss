/*
 * embeddings.metal
 *
 * Metal kernels for embedding table lookups in transformer models.
 * Converts discrete token IDs to continuous vector representations by
 * looking up pre-trained embedding weights.
 *
 * Key optimizations:
 * - Parallel loading of embedding vectors using threadgroup parallelism
 * - Vectorized memory access using float4/bfloat4 for bandwidth efficiency
 * - Strided access pattern within threadgroups for memory coalescing
 * - On-the-fly type conversion from bfloat16 to float32
 */

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


/**
 * gptoss_bf16_f32_embeddings
 *
 * Performs embedding table lookup and converts from bfloat16 to float32.
 *
 * For each token ID in the input, this kernel:
 * 1. Looks up the corresponding embedding vector in the weight table
 * 2. Converts from bfloat16 to float32 precision
 * 3. Writes the result to the output buffer
 *
 * Thread organization:
 * - 1D threadgroup grid, one threadgroup per token
 * - Each threadgroup cooperatively loads one embedding vector
 * - Threads within a threadgroup load disjoint chunks with strided access
 *
 * Memory layout:
 * - tokens: [batch_size] - Token IDs to look up
 * - weights: [vocab_size * embedding_dim] - Embedding table in bfloat16
 * - output: [batch_size * embedding_dim] - Output embeddings in float32
 *
 * Performance considerations:
 * - Each threadgroup performs one large coalesced read from the embedding table
 * - Memory bandwidth is the primary bottleneck
 * - Type conversion is essentially free (just bit reinterpretation with extension)
 *
 * Parameters:
 * @param args       Configuration (num_vecs = embedding_dim / 4)
 * @param tokens     Input token IDs (one per threadgroup)
 * @param weights    Embedding weight table in bfloat16 (bfloat4 vectors)
 * @param output     Output embedding vectors in float32 (float4 vectors)
 * @param control    Control structure for early termination
 * @param gid        Threadgroup position (token index in batch)
 * @param tid        Thread position within threadgroup
 * @param threadgroup_size  Number of threads per threadgroup
 */
kernel void gptoss_bf16_f32_embeddings(
    constant gptoss_embeddings_args& args [[ buffer(0) ]],
    const device uint* tokens [[ buffer(1) ]],
    const device bfloat4* weights [[ buffer(2) ]],
    device float4* output [[ buffer(3) ]],
    const device gptoss_control* control [[ buffer(4) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    // Early exit if kernel has been signaled to abort
    if (control->abort != 0) {
        return;
    }

    // Load the token ID for this threadgroup (one token per threadgroup)
    const uint t = tokens[gid];

    // Position weight pointer to start of this token's embedding vector
    // Each embedding is num_vecs * 4 floats (stored as bfloat4 vectors)
    weights += t * args.num_vecs;

    // Position output pointer to this token's output location
    output += gid * args.num_vecs;

    // Parallel copy: each thread loads non-overlapping chunks with stride
    // This ensures coalesced memory access across the threadgroup
    for (uint i = tid; i < args.num_vecs; i += threadgroup_size) {
        // Load one bfloat4 vector (4 bfloat16 values)
        const bfloat4 w = weights[i];
        // Convert to float4 and write to output
        // Conversion is efficient as bfloat16 has same exponent range as float32
        output[i] = static_cast<float4>(w);
    }
}
