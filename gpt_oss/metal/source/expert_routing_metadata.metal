/*
 * expert_routing_metadata.metal
 *
 * Metal kernels for computing routing metadata in Mixture of Experts (MoE) models.
 * Prepares data structures needed for efficient expert execution by:
 * 1. Counting how many tokens are routed to each expert
 * 2. Computing offset arrays for gathering/scattering token data
 * 3. Assigning positions within each expert's input buffer
 *
 * This metadata enables:
 * - Efficient batching of tokens sent to the same expert
 * - Correct reconstruction of token order after expert processing
 * - Parallel expert execution with proper memory layout
 *
 * Key optimizations:
 * - Threadgroup-shared memory for atomic counting (faster than device memory)
 * - Relaxed memory ordering for atomic operations (no cross-threadgroup sync needed)
 * - Parallel histogram construction using all threads
 * - Single-threaded prefix sum for offset computation (small array)
 */

#include <internal/kernel-args.h>
#include <metal_integer>
#include <metal_math>
#include <metal_stdlib>

// Maximum number of experts supported (determines threadgroup memory allocation)
constant uint kMaxExperts = 128;

/**
 * gptoss_f32_expert_routing_metadata
 *
 * Computes routing metadata for MoE expert execution.
 *
 * This kernel performs three main operations:
 * 1. Histogram: Count how many tokens are assigned to each expert
 * 2. Prefix sum: Compute starting offset for each expert's inputs
 * 3. Assignment: For each token, compute its position within its expert's buffer
 *
 * Algorithm:
 * Phase 1 - Initialize counters:
 *   - All threads cooperatively zero threadgroup memory counters
 *
 * Phase 2 - Count and assign (parallel):
 *   - Each thread processes a subset of tokens
 *   - For each token, atomically increment its expert's counter
 *   - The old counter value becomes this token's intra-expert offset
 *
 * Phase 3 - Compute offsets (single-threaded):
 *   - Thread 0 performs prefix sum over expert counts
 *   - Creates offset array showing where each expert's data starts
 *
 * Thread organization:
 * - Single threadgroup kernel (all work done by one threadgroup)
 * - Threadgroup memory used for atomic counters (fast shared memory)
 * - Multiple synchronization barriers to coordinate phases
 *
 * Memory layout:
 * - expert_predictions: [tokens] - Which expert processes each token
 * - expert_offsets: [num_experts + 1] - Starting position for each expert (prefix sum)
 * - intra_expert_offsets: [tokens] - Position within expert's buffer for each token
 *
 * Example:
 *   If 4 tokens are assigned to experts [1, 0, 1, 1]:
 *   - expert_offsets would be [0, 3, 4] (expert 0 gets 1 token, expert 1 gets 3)
 *   - intra_expert_offsets would be [0, 0, 1, 2] (position within respective expert)
 *
 * Parameters:
 * @param args                 Configuration (tokens, num_experts)
 * @param expert_predictions   Expert ID for each token
 * @param expert_offsets       Output: starting offset for each expert [num_experts + 1]
 * @param intra_expert_offsets Output: position within expert buffer [tokens]
 * @param tg_size              Threads per threadgroup
 * @param tid                  Thread position within threadgroup
 */
kernel void gptoss_f32_expert_routing_metadata(
    constant gptoss_expert_routing_metadata_args& args [[ buffer(0) ]],
    const device gptoss_expert_prediction* __restrict__ expert_predictions [[ buffer(1) ]],
    device uint* __restrict__ expert_offsets [[ buffer(2) ]],
    device uint* __restrict__ intra_expert_offsets [[ buffer(3) ]],
    uint tg_size [[threads_per_threadgroup]],
    uint tid [[thread_position_in_threadgroup]])
{
    assert(args.num_experts <= kMaxExperts);

    // Phase 1: Initialize threadgroup memory
    // Allocate atomic counters in fast threadgroup-shared memory
    threadgroup metal::atomic_uint tg_counts[kMaxExperts];

    // Parallel initialization: each thread zeros a subset of counters
    for (uint e = tid; e < args.num_experts; e += tg_size) {
        metal::atomic_store_explicit(&tg_counts[e], 0u, metal::memory_order_relaxed);
    }

    // Wait for all threads to finish initialization
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Phase 2: Count tokens per expert and assign intra-expert offsets
    // Each thread processes tokens with stride equal to threadgroup size
    for (uint i = tid; i < args.tokens; i += tg_size) {
        // Load which expert this token is assigned to
        const uint e = expert_predictions[i].expert_id;

        // Atomically increment the counter for this expert
        // The OLD value (before increment) becomes this token's position within the expert
        // Using relaxed memory order is safe since all operations are within threadgroup
        const uint r = metal::atomic_fetch_add_explicit(&tg_counts[e], 1u, metal::memory_order_relaxed);

        // Store the intra-expert offset (position within this expert's inputs)
        intra_expert_offsets[i] = r;
    }

    // Wait for all threads to finish counting
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Phase 3: Compute prefix sum of expert counts (single-threaded)
    // Only thread 0 performs this step since it's sequential
    if (tid == 0) {
        uint total = 0;
        // For each expert, store the cumulative count (starting offset)
        for (uint e = 0; e < args.num_experts; ++e) {
            // Load final count for this expert
            const uint bin = metal::atomic_load_explicit(&tg_counts[e], metal::memory_order_relaxed);
            // Store starting offset for this expert
            expert_offsets[e] = total;
            // Accumulate for next expert
            total += bin;
        }
        // Store final total (useful for knowing total size needed)
        expert_offsets[args.num_experts] = total;
    }
}
