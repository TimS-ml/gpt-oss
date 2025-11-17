/**
 * @file datatype.h
 * @brief Reduced-precision floating-point data type definitions
 *
 * This header defines structs for various reduced-precision floating-point
 * formats used by the GPT-OSS library for efficient model storage and computation.
 * These types are used primarily for model weights to reduce memory footprint
 * and increase throughput on Metal GPUs.
 *
 * All types are defined as tightly-packed structs containing raw bit representations.
 * They are not meant for direct arithmetic in C code, but rather for:
 * - Memory layout compatibility with Metal shaders
 * - Type-safe handling of quantized weights
 * - Efficient data transfer to GPU
 *
 * Supported formats:
 * - BFloat16: Brain float, 1 sign + 8 exp + 7 mantissa bits
 * - Float16: IEEE half precision, 1 sign + 5 exp + 10 mantissa bits
 * - Float8: Various 8-bit formats for extreme compression
 * - Float4: 4-bit formats packed 2 per byte
 */
#pragma once

#include <stdint.h>

#include <internal/macros.h>

/**
 * @struct gptoss_bfloat16
 * @brief BFloat16 (Brain Floating Point) data type
 *
 * BFloat16 is a 16-bit floating-point format with the same exponent range
 * as FP32 (8 bits) but reduced mantissa precision (7 bits + implicit 1).
 * Format: 1 sign bit, 8 exponent bits, 7 mantissa bits
 *
 * Properties:
 * - Easy conversion to/from FP32 (truncate/extend mantissa)
 * - Same dynamic range as FP32
 * - Lower precision than FP16 but better range
 * - Widely supported on modern AI accelerators
 *
 * Memory layout: Densely packed with no padding, 2-byte aligned
 */
typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(2) uint16_t bits;  /* Raw 16-bit representation */
} gptoss_bfloat16;
static_assert(sizeof(gptoss_bfloat16) == 2, "bfloat16 size is not 2 bytes");

/**
 * @struct gptoss_float16
 * @brief IEEE 754 half-precision (FP16) data type
 *
 * Standard IEEE 754 16-bit floating-point format.
 * Format: 1 sign bit, 5 exponent bits, 10 mantissa bits
 *
 * Properties:
 * - Higher precision than BFloat16
 * - Smaller dynamic range than BFloat16
 * - Native support on many GPUs
 * - Standard interchange format
 *
 * Memory layout: Densely packed with no padding, 2-byte aligned
 */
typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(2) uint16_t bits;  /* Raw 16-bit representation */
} gptoss_float16;
static_assert(sizeof(gptoss_float16) == 2, "float16 size is not 2 bytes");

/**
 * @struct gptoss_float8ue8m0
 * @brief 8-bit float with unusual encoding (8 exponent, 0 mantissa bits)
 *
 * Experimental 8-bit format with maximum dynamic range but minimal precision.
 * Useful for extremely compressed representations where only magnitude matters.
 *
 * Memory layout: Densely packed with no padding, 1-byte aligned
 */
typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(1) uint8_t bits;  /* Raw 8-bit representation */
} gptoss_float8ue8m0;
static_assert(sizeof(gptoss_float8ue8m0) == 1, "gptoss_float8ue8m0 size is not 1 bytes");

/**
 * @struct gptoss_float8e5m2
 * @brief 8-bit float with 5 exponent and 2 mantissa bits
 *
 * Format: 1 sign bit, 5 exponent bits, 2 mantissa bits
 * Prioritizes dynamic range over precision, suitable for weights.
 *
 * Memory layout: Densely packed with no padding, 1-byte aligned
 */
typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(1) uint8_t bits;  /* Raw 8-bit representation */
} gptoss_float8e5m2;
static_assert(sizeof(gptoss_float8e5m2) == 1, "float8e5m2 size is not 1 bytes");

/**
 * @struct gptoss_float8e4m3
 * @brief 8-bit float with 4 exponent and 3 mantissa bits
 *
 * Format: 1 sign bit, 4 exponent bits, 3 mantissa bits
 * Balances range and precision, commonly used in quantized models.
 *
 * Memory layout: Densely packed with no padding, 1-byte aligned
 */
typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(1) uint8_t bits;  /* Raw 8-bit representation */
} gptoss_float8e4m3;
static_assert(sizeof(gptoss_float8e4m3) == 1, "float8e4m3 size is not 1 bytes");

/**
 * @struct gptoss_float4e2m1x2
 * @brief 4-bit float with 2 exponent and 1 mantissa bit, packed 2 per byte
 *
 * Format: 1 sign bit, 2 exponent bits, 1 mantissa bit (4 bits total)
 * Two values are packed into a single byte for extreme compression.
 *
 * Used for maximally compressed model weights, typically with per-channel
 * or per-block scaling factors to maintain acceptable accuracy.
 *
 * Memory layout: Densely packed, two 4-bit values per byte, 1-byte aligned
 */
typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(1) uint8_t bits;  /* Two 4-bit floats packed in 8 bits */
} gptoss_float4e2m1x2;
static_assert(sizeof(gptoss_float4e2m1x2) == 1, "gptoss_float4e2m1x2 size is not 1 bytes");
