/*
 * random.metal
 *
 * Pseudo-random number generation kernels for GPU using the Squares RNG algorithm.
 * Provides deterministic, high-quality random number generation with good statistical
 * properties and GPU-friendly parallelization.
 *
 * Algorithm: Squares RNG (by Bernard Widynski)
 * - Counter-based PRNG optimized for parallel execution
 * - Each thread generates independent random values based on offset
 * - Uses only integer arithmetic (multiply, add, rotate)
 * - Passes statistical tests (PractRand, BigCrush)
 * - Fast on GPU with no shared state or synchronization
 *
 * Key features:
 * - Deterministic: Same seed + offset always produces same output
 * - Parallel-friendly: No dependencies between threads
 * - High quality: Good statistical properties for ML applications
 * - Efficient: Only integer operations (no divisions or branches)
 *
 * Use cases:
 * - Weight initialization (e.g., Xavier/Kaiming)
 * - Dropout masks during training
 * - Data augmentation
 * - Stochastic operations in models
 */

#include <metal_integer>
#include <metal_math>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


/**
 * rng_squares32
 *
 * Squares random number generator (Widynski, 2022).
 * Generates a 32-bit pseudo-random number from an offset and seed.
 *
 * Algorithm:
 * 1. Initialize: y = offset * seed, z = y + seed
 * 2. Four rounds of: x = x*x + (y or z), then rotate bits
 * 3. Return upper 32 bits of final 64-bit state
 *
 * Properties:
 * - Period: Extremely long (2^64 for each offset)
 * - Parallel: Each offset produces independent stream
 * - Quality: Passes rigorous statistical tests
 *
 * Parameters:
 * @param offset  Position in random stream (unique per thread/element)
 * @param seed    Global seed for reproducibility
 * @return        32-bit pseudo-random unsigned integer
 */
inline static uint rng_squares32(ulong offset, ulong seed) {
    const ulong y = offset * seed;
    const ulong z = y + seed;

    /* Round 1: Square-add-rotate */
    ulong x = y * y + y;
    x = metal::rotate(x, 32ul);  // Rotate left by 32 bits

    /* Round 2 */
    x = x * x + z;
    x = metal::rotate(x, 32ul);

    /* Round 3 */
    x = x * x + y;
    x = metal::rotate(x, 32ul);

    /* Round 4 */
    x = x * x + z;

    // Return upper 32 bits (better randomness than lower bits)
    return as_type<uint2>(x).y;
}

/**
 * gptoss_u32_fill_random
 *
 * Fills a buffer with pseudo-random 32-bit unsigned integers.
 *
 * Thread organization:
 * - 1D threadgroup grid with strided access
 * - Each thread generates independent random values
 * - No synchronization or shared state required
 *
 * Parameters:
 * @param args   Configuration (num_vecs, num_vecs_per_threadgroup, seed, offset)
 * @param output Output buffer to fill with random uint32 values
 * @param gid, tid, threadgroup_size  Thread organization
 */
kernel void gptoss_u32_fill_random(
    constant gptoss_u32_fill_random_args& args [[ buffer(0) ]],
    device uint* output [[ buffer(1) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    // Calculate range of elements for this threadgroup
    const ulong num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const ulong threadgroup_start = gid * num_vecs_per_threadgroup;
    const ulong threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, args.num_vecs);

    // Each thread starts at different position with stride
    const ulong thread_start = threadgroup_start + tid;
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size - 1)) / threadgroup_size);

    // Position output pointer and initialize offset for RNG
    output += thread_start;
    ulong offset = args.offset + thread_start;

    // Generate random values with strided access
    for (; num_iter != 0; num_iter--) {
        *output = rng_squares32(offset, args.seed);
        output += threadgroup_size;
        offset += threadgroup_size;
    }
}

/**
 * gptoss_f32_fill_random
 *
 * Fills a buffer with pseudo-random float32 values in a specified range.
 *
 * Conversion from uint32 to float:
 * 1. Generate random uint32
 * 2. Reinterpret as signed int32
 * 3. Convert to float and scale: float_val = int_val * scale + bias
 *
 * This maps the full int32 range to a symmetric float distribution.
 * Common usage: scale and bias set for uniform or normal-like distributions.
 *
 * Parameters:
 * @param args   Configuration (num_vecs, seed, offset, scale, bias)
 * @param output Output buffer to fill with random float32 values
 * @param gid, tid, threadgroup_size  Thread organization
 */
kernel void gptoss_f32_fill_random(
    constant gptoss_f32_fill_random_args& args [[ buffer(0) ]],
    device float* output [[ buffer(1) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    const ulong num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const ulong threadgroup_start = gid * num_vecs_per_threadgroup;
    const ulong threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, args.num_vecs);
    const ulong thread_start = threadgroup_start + tid;
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size - 1)) / threadgroup_size);

    output += thread_start;
    ulong offset = args.offset + thread_start;

    for (; num_iter != 0; num_iter--) {
        // Generate random uint, reinterpret as signed int, scale and bias
        const uint word = rng_squares32(offset, args.seed);
        *output = metal::fma(static_cast<float>(as_type<int>(word)), args.scale, args.bias);
        output += threadgroup_size;
        offset += threadgroup_size;
    }
}

/**
 * gptoss_bf16_fill_random
 *
 * Fills a buffer with pseudo-random bfloat16 values.
 * Similar to f32 version but with final conversion to bfloat16.
 *
 * Useful for initializing bfloat16 weight tensors.
 *
 * Parameters:
 * @param args   Configuration (num_vecs, seed, offset, scale, bias)
 * @param output Output buffer to fill with random bfloat16 values
 * @param gid, tid, threadgroup_size  Thread organization
 */
kernel void gptoss_bf16_fill_random(
    constant gptoss_f32_fill_random_args& args [[ buffer(0) ]],
    device bfloat* output [[ buffer(1) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    const ulong num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const ulong threadgroup_start = gid * num_vecs_per_threadgroup;
    const ulong threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, args.num_vecs);
    const ulong thread_start = threadgroup_start + tid;
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size - 1)) / threadgroup_size);

    output += thread_start;
    ulong offset = args.offset + thread_start;

    for (; num_iter != 0; num_iter--) {
        const uint word = rng_squares32(offset, args.seed);
        // Convert to float, scale, and convert to bfloat16
        *output = static_cast<bfloat>(metal::fma(static_cast<float>(as_type<int>(word)), args.scale, args.bias));
        output += threadgroup_size;
        offset += threadgroup_size;
    }
}
