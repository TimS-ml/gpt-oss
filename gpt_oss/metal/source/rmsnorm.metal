/*
 * rmsnorm.metal
 *
 * Root Mean Square Layer Normalization (RMSNorm) for transformer models.
 * A simplified variant of LayerNorm that only normalizes by RMS (no mean centering).
 *
 * RMSNorm Background:
 * - Simpler than LayerNorm: only RMS normalization, no mean subtraction
 * - Computes: output = (input / RMS(input)) * weight
 * - RMS = sqrt(mean(input^2) + epsilon)
 * - Used in modern LLMs (LLaMA, Mistral, etc.) for efficiency
 *
 * Algorithm:
 * 1. Compute sum of squares across all dimensions
 * 2. Calculate RMS: sqrt(sum_squares / num_dims + epsilon)
 * 3. Scale input by 1/RMS
 * 4. Apply learned per-dimension weights
 *
 * Key optimizations:
 * - Two-level parallel reduction (simdgroup then threadgroup)
 * - Single threadgroup per sequence element
 * - Vectorized operations using float4
 * - Minimal synchronization (one barrier)
 * - Fused normalization and weight multiplication
 */

#include <metal_compute>
#include <metal_math>
#include <metal_simdgroup>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


/**
 * gptoss_f32_bf16w_rmsnorm
 *
 * RMSNorm with bfloat16 weights and float32 computation.
 *
 * Thread organization:
 * - One threadgroup per sequence element (token)
 * - Up to 1024 threads per threadgroup (32 simdgroups * 32 threads)
 * - All threads cooperate to compute global RMS
 * - Two-stage reduction: simdgroup sum, then threadgroup sum
 *
 * Memory layout:
 * - Input/output: [batch * embedding_dim] in float32
 * - Weights: [embedding_dim] in bfloat16 (learned parameters)
 *
 * Algorithm stages:
 * 1. Parallel computation of partial sum-of-squares per thread
 * 2. Simdgroup reduction (32 threads -> 1 value per simdgroup)
 * 3. Threadgroup reduction (32 simdgroups -> 1 global value)
 * 4. Compute RMS and scaling factor
 * 5. Parallel application of normalization and weights
 *
 * Numerical stability:
 * - Epsilon prevents division by zero
 * - Uses precise rsqrt for reciprocal square root
 *
 * Parameters:
 * @param args       Configuration (num_vecs, num_channels, epsilon)
 * @param input      Input activations [batch * embedding_dim] in float32
 * @param weights    Learned scaling weights [embedding_dim] in bfloat16
 * @param output     Normalized output [batch * embedding_dim] in float32
 * @param control    Control structure for early termination
 * @param gid        Threadgroup position (batch/sequence index)
 * @param tid        Thread position within threadgroup
 * @param threadgroup_size  Number of threads per threadgroup
 */
[[max_total_threads_per_threadgroup(1024)]]
kernel void gptoss_f32_bf16w_rmsnorm(
    constant gptoss_rmsnorm_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device bfloat4* weights [[ buffer(2) ]],
    device float4* output [[ buffer(3) ]],
    const device gptoss_control* control [[ buffer(4) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    const uint simdgroup_size = 32;
    threadgroup float threadgroup_buffer[32];  // One value per simdgroup

    if (control->abort != 0) {
        return;
    }

    // Position pointers to this sequence element's data
    input += gid * args.num_vecs;
    output += gid * args.num_vecs;

    // Phase 1: Compute partial sum of squares
    // Each thread processes multiple float4 vectors with stride
    float4 sumsq4 = 0.0f;
    for (uint i = tid; i < args.num_vecs; i += threadgroup_size) {
        const float4 val = input[i];
        // Accumulate element-wise squares using FMA
        sumsq4 = metal::fma(val, val, sumsq4);
    }

    // Phase 2: Reduce float4 to scalar within each thread
    const float2 sumsq2 = sumsq4.xy + sumsq4.zw;  // Horizontal sum: 4 -> 2
    float sumsq = sumsq2.x + sumsq2.y;             // Horizontal sum: 2 -> 1

    // Phase 3: Simdgroup reduction (parallel sum across 32 threads)
    // Note: This works only for simdgroup_size=32 and threadgroup_size=32*32=1024
    sumsq = metal::simd_sum(sumsq);

    // Phase 4: Write simdgroup sums to threadgroup memory
    if (metal::simd_is_first()) {
        const uint simdgroup_idx = tid / simdgroup_size;
        threadgroup_buffer[simdgroup_idx] = sumsq;
    }

    // Synchronize to ensure all simdgroup sums are written
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Phase 5: Final reduction across simdgroups
    // Each thread loads one simdgroup's sum, then simd_sum reduces them
    const uint simdgroup_tid = tid % simdgroup_size;
    sumsq = threadgroup_buffer[simdgroup_tid];
    sumsq = metal::simd_sum(sumsq);  // Now all threads have the global sum

    // Phase 6: Compute normalization scale
    // RMS = sqrt(mean(x^2) + epsilon) = sqrt(sum(x^2)/N + epsilon)
    const float avgsq = sumsq / args.num_channels;
    // Reciprocal square root for efficient scaling: 1/RMS
    const float scale = metal::precise::rsqrt(avgsq + args.epsilon);

    // Phase 7: Apply normalization and weights
    // Each thread processes multiple vectors with stride
    for (uint i = tid; i < args.num_vecs; i += threadgroup_size) {
        // Normalize: input / RMS
        const float4 val = input[i] * scale;
        // Load and convert weight from bfloat16 to float32
        const float4 weight_val = static_cast<float4>(weights[i]);
        // Apply learned weight and write output
        output[i] = val * weight_val;
    }
}
