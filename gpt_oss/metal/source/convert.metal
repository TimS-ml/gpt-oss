/*
 * convert.metal
 *
 * Metal kernels for converting quantized weight formats to full precision.
 * Implements high-performance dequantization of MF4 (Metal Float 4-bit) format
 * to standard 32-bit floating point.
 *
 * MF4 Format:
 * - 4-bit floating point representation for model weights
 * - Each block contains 32 values packed into 128 bits (uint4)
 * - Uses block-wise scaling for better dynamic range
 * - Optimized for memory bandwidth and storage reduction
 *
 * Key optimizations:
 * - SIMD bit manipulation for parallel unpacking
 * - Vectorized float4 outputs for memory efficiency
 * - Strided memory access for threadgroup coalescing
 * - Minimal branching for GPU efficiency
 */

#include <metal_integer>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


/**
 * gptoss_mf4_f32_convert
 *
 * Converts MF4 (4-bit quantized) weights to full precision float32.
 *
 * Each thread processes blocks of 32 quantized values and produces 32 float32 values.
 * The conversion involves:
 * 1. Unpacking 4-bit values from packed uint4 representation
 * 2. Converting to floating point through bit manipulation
 * 3. Scaling by per-block scale factor
 * 4. Transposing data for optimal output layout
 *
 * Memory layout:
 * - Input blocks: Each uint4 contains 32 4-bit values (128 bits)
 * - Input scales: One uchar per block for scaling
 * - Output: 8 consecutive float4 vectors per block (32 floats)
 *
 * Thread organization:
 * - 1D threadgroup grid
 * - Each threadgroup processes num_vecs_per_threadgroup blocks
 * - Strided access pattern for memory coalescing
 *
 * Parameters:
 * @param args       Configuration (num_vecs, num_vecs_per_threadgroup)
 * @param blocks     Packed 4-bit weights (uint4 format)
 * @param scales     Per-block scaling factors (uchar format)
 * @param output     Dequantized float32 weights (float4 vectors)
 * @param gid        Threadgroup position in grid
 * @param tid        Thread position within threadgroup
 * @param threadgroup_size  Number of threads per threadgroup
 */
kernel void gptoss_mf4_f32_convert(
    constant gptoss_convert_args& args [[ buffer(0) ]],
    const device uint4* blocks [[ buffer(1) ]],
    const device uchar* scales [[ buffer(2) ]],
    device float4* output [[ buffer(3) ]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[ threads_per_threadgroup ]])
{
    // Calculate the range of blocks this threadgroup will process
    const ulong num_vecs_per_threadgroup = args.num_vecs_per_threadgroup;
    const ulong threadgroup_start = gid * num_vecs_per_threadgroup;
    const ulong threadgroup_end = metal::min(threadgroup_start + num_vecs_per_threadgroup, args.num_vecs);

    // Each thread processes blocks with stride equal to threadgroup size
    const ulong thread_start = threadgroup_start + tid;
    // Calculate number of iterations (ceil division)
    uint num_iter = static_cast<uint>((threadgroup_end - thread_start + (threadgroup_size - 1)) / threadgroup_size);

    // Position input and output pointers for this thread
    blocks += thread_start;
    scales += thread_start;
    output += 8 * thread_start;  // Each block produces 8 float4 vectors

    // Main conversion loop - each iteration processes one block (32 values)
    for (; num_iter != 0; num_iter--) {
        // Load packed 4-bit block (128 bits containing 32 4-bit values)
        const uint4 block = *blocks;

        // Decode scale factor: Convert uchar to float using bit manipulation
        // Scale is stored as exponent offset: final_scale = 2^(scale_byte - 14)
        // We construct the float by shifting (scale + 14) into the exponent field (bits 23-30)
        const float scale = as_type<float>((static_cast<uint>(*scales) + 14) << 23);

        // Step 1: Split into even and odd nibbles (4-bit values)
        // Even nibbles (positions 0,2,4,6,8,A,C,E,G,I,K,M,O,Q,S,U)
        uint4 block02468ACEGIKMOQSU = block + block;  // Left shift by 1 bit (multiply by 2)
        // Odd nibbles (positions 1,3,5,7,9,B,D,F,H,J,L,N,P,R,T,V)
        uint4 block13579BDFHJLNPRTV = block >> 3;  // Right shift by 3 bits

        // Step 2: Mask to isolate 4-bit values (0x1E = 0b00011110, keeps bits 1-4)
        block02468ACEGIKMOQSU &= 0x1E1E1E1Eu;
        block13579BDFHJLNPRTV &= 0x1E1E1E1Eu;

        // Step 3: Add bias (0x70 = 0b01110000) for conversion to floating point range
        block02468ACEGIKMOQSU += 0x70707070u;
        block13579BDFHJLNPRTV += 0x70707070u;

        // Step 4: Mask to prepare for half-precision float interpretation
        // (0x8E = 0b10001110, sets sign bit and keeps relevant mantissa bits)
        block02468ACEGIKMOQSU &= 0x8E8E8E8Eu;
        block13579BDFHJLNPRTV &= 0x8E8E8E8Eu;

        // Step 5: Further split each stream into two groups for final layout
        // Extract bytes at positions 2,6,A,E,I,M,Q,U from even stream
        const uint4 block26AEIMQU = block02468ACEGIKMOQSU & 0xFF00FF00u;
        // Extract bytes at positions 0,4,8,C,G,K,O,S from even stream (shift left to align)
        const uint4 block048CGKOS = (block02468ACEGIKMOQSU << 8) & 0xFF00FF00u;
        // Extract bytes at positions 3,7,B,F,J,N,R,V from odd stream
        const uint4 block37BFJNRV = block13579BDFHJLNPRTV & 0xFF00FF00u;
        // Extract bytes at positions 1,5,9,D,H,L,P,T from odd stream (shift left to align)
        const uint4 block159DHLPT = (block13579BDFHJLNPRTV << 8) & 0xFF00FF00u;

        // Step 6: Convert to half-precision float, then to float32, and apply scale
        // Process first 16 values (split into 4 groups of 4)
        const float4 block048C = static_cast<float4>(as_type<half4>(block048CGKOS.xy)) * scale;
        const float4 blockGKOS = static_cast<float4>(as_type<half4>(block048CGKOS.zw)) * scale;
        const float4 block26AE = static_cast<float4>(as_type<half4>(block26AEIMQU.xy)) * scale;
        const float4 blockIMQU = static_cast<float4>(as_type<half4>(block26AEIMQU.zw)) * scale;
        // Process second 16 values (split into 4 groups of 4)
        const float4 block159D = static_cast<float4>(as_type<half4>(block159DHLPT.xy)) * scale;
        const float4 blockHLPT = static_cast<float4>(as_type<half4>(block159DHLPT.zw)) * scale;
        const float4 block37BF = static_cast<float4>(as_type<half4>(block37BFJNRV.xy)) * scale;
        const float4 blockJNRV = static_cast<float4>(as_type<half4>(block37BFJNRV.zw)) * scale;

        // Step 7: Transpose and write output in sequential order (0-31)
        // Interleave values from different groups to restore original order
        output[0] = (float4) { block048C.x, block159D.x, block26AE.x, block37BF.x };  // Values 0-3
        output[1] = (float4) { block048C.y, block159D.y, block26AE.y, block37BF.y };  // Values 4-7
        output[2] = (float4) { block048C.z, block159D.z, block26AE.z, block37BF.z };  // Values 8-11
        output[3] = (float4) { block048C.w, block159D.w, block26AE.w, block37BF.w };  // Values 12-15
        output[4] = (float4) { blockGKOS.x, blockHLPT.x, blockIMQU.x, blockJNRV.x };  // Values 16-19
        output[5] = (float4) { blockGKOS.y, blockHLPT.y, blockIMQU.y, blockJNRV.y };  // Values 20-23
        output[6] = (float4) { blockGKOS.z, blockHLPT.z, blockIMQU.z, blockJNRV.z };  // Values 24-27
        output[7] = (float4) { blockGKOS.w, blockHLPT.w, blockIMQU.w, blockJNRV.w };  // Values 28-31

        // Advance pointers by stride for next iteration
        blocks += threadgroup_size;
        scales += threadgroup_size;
        output += 8 * threadgroup_size;  // Skip 8 float4 vectors (32 floats)
    }
}
