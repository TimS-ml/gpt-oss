/*
 * accumulate.metal
 *
 * Metal kernels for accumulating expert outputs in Mixture of Experts (MoE) models.
 * These kernels combine outputs from multiple experts by applying learned weights
 * (scores) and accumulating results into a final output tensor.
 *
 * Key optimizations:
 * - Uses float4 SIMD vectors for 4x parallelism per thread
 * - Fused multiply-add (FMA) operations for efficiency
 * - Threadgroup-based work distribution for GPU parallelism
 * - Stride-based memory access to maximize memory bandwidth
 */

#include <metal_integer>
#include <metal_math>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


/**
 * gptoss_f32_accumulate_e4
 *
 * Accumulates outputs from 4 experts in a Mixture of Experts layer.
 *
 * This kernel performs weighted accumulation:
 *   output[i] += expert0_output[i] * score0 + expert1_output[i] * score1 +
 *                expert2_output[i] * score2 + expert3_output[i] * score3
 *
 * Thread organization:
 * - 2D threadgroup grid: X dimension for data parallelism, Y dimension for batch/sequence
 * - Each threadgroup processes a chunk of vectors (num_vecs_per_threadgroup)
 * - Threads within a threadgroup process vectors with strided access
 *
 * Memory layout:
 * - Input: [batch * num_vecs * 4 experts] - All expert outputs concatenated
 * - Expert predictions: [batch * 4] - Scores for each expert per batch element
 * - Output: [batch * num_vecs] - Accumulated results
 *
 * Parameters:
 * @param args       Configuration parameters (num_vecs, num_vecs_per_threadgroup, etc.)
 * @param input      Device memory containing all expert outputs (float4 vectors)
 * @param expert     Device memory containing expert scores/predictions for weighting
 * @param output     Device memory for accumulated output (read-modify-write)
 * @param control    Control structure for early kernel termination (abort flag)
 * @param gid        2D threadgroup position in grid (X: data chunks, Y: batch index)
 * @param tid        Thread index within threadgroup
 * @param threadgroup_size  Number of threads per threadgroup
 */
kernel void gptoss_f32_accumulate_e4(
    constant gptoss_accumulate_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device gptoss_expert_prediction* expert [[ buffer(2) ]],
    device float4* output [[ buffer(3) ]],
    const device gptoss_control* control [[ buffer(4) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 threadgroup_size [[ threads_per_threadgroup ]])
{
    const uint num_active_experts = 4;

    // Early exit if kernel has been signaled to abort (error handling)
    if (control->abort != 0) {
        return;
    }

    // Calculate the range of vectors this threadgroup will process
    const uint num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const uint threadgroup_start = gid.x * num_vecs_per_threadgroup;  // Start index for this threadgroup
    const uint num_vecs = args.num_vecs;
    const uint threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, num_vecs);  // Clamp to total vectors

    // Each thread processes vectors with stride equal to threadgroup size
    const uint thread_start = threadgroup_start + tid;
    // Calculate number of iterations this thread will perform (ceil division)
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size.x - 1)) / threadgroup_size.x);

    // Set up pointers and scaling factors for all 4 experts
    // Each expert's output occupies num_vecs_per_expert consecutive vectors
    const uint num_vecs_per_expert = args.num_vecs_per_expert;

    // Expert 0: Load score and set up input pointer
    const float scale0 = expert[gid.y * num_active_experts + 0].score;
    const device float4* input0 = input + gid.y * num_vecs + thread_start;

    // Expert 1: Offset by num_vecs_per_expert from expert 0
    const float scale1 = expert[gid.y * num_active_experts + 1].score;
    const device float4* input1 = input0 + num_vecs_per_expert;

    // Expert 2: Offset by num_vecs_per_expert from expert 1
    const float scale2 = expert[gid.y * num_active_experts + 2].score;
    const device float4* input2 = input1 + num_vecs_per_expert;

    // Expert 3: Offset by num_vecs_per_expert from expert 2
    const float scale3 = expert[gid.y * num_active_experts + 3].score;
    const device float4* input3 = input2 + num_vecs_per_expert;

    // Position output pointer for this thread
    output += gid.y * num_vecs + thread_start;

    // Main accumulation loop - each iteration processes one float4 vector
    for (; num_iter != 0; num_iter--) {
        // Load current accumulated value from output (read-modify-write pattern)
        float4 acc = *output;

        // Load one vector from each expert's output
        const float4 val0 = *input0;
        const float4 val1 = *input1;
        const float4 val2 = *input2;
        const float4 val3 = *input3;

        // Advance input pointers by stride (threadgroup_size.x) for coalesced access
        input0 += threadgroup_size.x;
        // Accumulate expert 0: acc += val0 * scale0 (using FMA for efficiency)
        acc = metal::fma(val0, scale0, acc);

        input1 += threadgroup_size.x;
        // Accumulate expert 1: acc += val1 * scale1
        acc = metal::fma(val1, scale1, acc);

        input2 += threadgroup_size.x;
        // Accumulate expert 2: acc += val2 * scale2
        acc = metal::fma(val2, scale2, acc);

        input3 += threadgroup_size.x;
        // Accumulate expert 3: acc += val3 * scale3
        acc = metal::fma(val3, scale3, acc);

        // Write accumulated result back to output
        *output = acc;
        // Advance output pointer by stride
        output += threadgroup_size.x;
    }
}
