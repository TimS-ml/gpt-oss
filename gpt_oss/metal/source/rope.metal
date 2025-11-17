/*
 * rope.metal
 *
 * Rotary Position Embeddings (RoPE) for transformer attention mechanisms.
 * Encodes positional information by rotating embedding vectors in complex space.
 *
 * RoPE Background:
 * - Alternative to absolute/relative position embeddings
 * - Treats pairs of dimensions as complex numbers
 * - Rotates by angle proportional to position
 * - Preserves dot product relationships with relative positions
 * - Enables length extrapolation beyond training context
 *
 * Algorithm:
 * For each dimension pair [x, y] at position t:
 *   θ = t * freq(dim)
 *   output = [x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ)]
 *
 * YaRN Extensions (Yet another RoPE extensioN):
 * - Frequency interpolation for long contexts
 * - Extrapolation for very long sequences
 * - Alpha blending between interpolation/extrapolation
 * - Attention temperature scaling
 *
 * Key optimizations:
 * - Parallel processing of all dimension pairs
 * - Direct writes to KV cache (no intermediate storage)
 * - Efficient sincos computation
 * - 2D grid mapping for parallelism
 */

#include <metal_common>
#include <metal_math>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


/**
 * gptoss_f32_rope
 *
 * Applies Rotary Position Embeddings to Q and K tensors.
 *
 * Thread organization:
 * - 2D grid: X = dimension pairs, Y = tokens
 * - Each thread handles one complex number (2 floats)
 * - Each simdgroup handles one attention head (32 dimension pairs = 64 dims)
 *
 * Memory layout:
 * - Activations: [tokens * heads * head_dim]
 * - KV cache: [heads * max_tokens * 2 * head_dim] (K and V interleaved)
 *
 * Frequency calculation:
 * - Base frequencies decrease with dimension index (high to low freq)
 * - YaRN interpolation scales frequencies for context extension
 * - Alpha blending between extrapolation and interpolation modes
 *
 * Parameters:
 * @param args         Configuration (token_offset, freq_scale, YaRN parameters)
 * @param activations  Input/output Q and K activations (in-place)
 * @param kv           KV cache for writing K values
 * @param control      Control structure for early termination
 * @param gid          2D thread position (X: dim pairs, Y: tokens)
 */
kernel void gptoss_f32_rope(
    constant gptoss_rope_args& args [[ buffer(0) ]],
    device float2* activations [[ buffer(1) ]],
    device float2* kv [[ buffer(2) ]],
    const device gptoss_control* control [[ buffer(3) ]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint num_head_dims = 64;
    if (control->abort != 0) {
        return;
    }

    // Calculate dimension index and token index
    const float dim_idx = static_cast<float>(gid.x % (num_head_dims / 2));
    const uint token_idx = args.token_offset + gid.y;

    // Position activation pointer
    activations += gid.y * args.token_stride + gid.x;

    // Load input values (complex number: real and imaginary parts)
    const float2 input_vals = *activations;

    // Compute rotation frequency using YaRN scaling
    // Extrapolation frequency: base exponential decay with dimension
    const float inv_extrapolation_freq = metal::precise::exp(dim_idx * args.freq_scale);
    // Interpolation frequency: scaled for context extension
    const float inv_interpolation_freq = inv_extrapolation_freq * args.interpolation_scale;
    // Alpha blending between modes (saturated linear function of dimension)
    const float alpha = metal::saturate(metal::fma(dim_idx, args.yarn_scale, args.yarn_offset));
    // Final frequency is blend of extrapolation and interpolation
    const float inv_freq = metal::mix(inv_extrapolation_freq, inv_interpolation_freq, alpha);

    // Calculate rotation angle: position * frequency
    const float phi = static_cast<float>(token_idx) * inv_freq;

    // Apply attention temperature scaling (YaRN)
    const float yarn_multiplier = args.yarn_multiplier;

    // Compute sine and cosine simultaneously (efficient)
    float cosphi;
    const float sinphi = metal::precise::sincos(phi, cosphi) * yarn_multiplier;
    cosphi *= yarn_multiplier;

    // Apply rotation: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
    const float output_re = input_vals.x * cosphi - input_vals.y * sinphi;
    const float output_im = input_vals.x * sinphi + input_vals.y * cosphi;

    // Write rotated values
    *activations = (float2) { output_re, output_im };

    // Determine which head this dimension pair belongs to
    const uint head_dim = 64;
    const uint num_q_heads = 64;
    const uint num_kv_heads = 8;
    const uint head_idx = gid.x / (head_dim / 2);

    float2 vals = (float2) { output_re, output_im };

    if ((head_idx < num_q_heads)) {
        // Q heads: already written to activations above
        *activations = vals;
    } else if (head_idx < num_q_heads + num_kv_heads) {
        // K heads: write directly to KV cache
        const uint kv_head_idx = head_idx - num_q_heads;
        const uint dim_pair_idx = gid.x % (head_dim / 2);
        kv[(kv_head_idx * args.max_tokens + token_idx) * head_dim + dim_pair_idx] = vals;
    }
    // V heads are not rotated (head_idx >= num_q_heads + num_kv_heads)
}
