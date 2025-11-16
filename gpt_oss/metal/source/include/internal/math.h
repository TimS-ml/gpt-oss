/**
 * @file math.h
 * @brief Mathematical utility functions for GPT-OSS
 *
 * This header provides inline math helper functions for:
 * - Integer division with rounding
 * - Min/max operations
 * - Saturating arithmetic
 * - Power-of-2 alignment
 *
 * All functions are implemented as static inline for zero-overhead abstraction.
 * Used throughout the codebase for buffer size calculations, alignment, etc.
 */
#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Ceiling division (divide and round up)
 *
 * Computes ceil(numer / denom) using only integer arithmetic.
 * Equivalent to (numer + denom - 1) / denom.
 *
 * @param numer Numerator
 * @param denom Denominator (must be > 0)
 * @return Quotient rounded up to nearest integer
 *
 * Example: math_ceil_div(10, 3) = 4
 */
inline static size_t math_ceil_div(size_t numer, size_t denom) {
    return (numer + denom - 1) / denom;
}

/**
 * @brief Maximum of two values
 * @return The larger of a or b
 */
inline static size_t math_max(size_t a, size_t b) {
    return a >= b ? a : b;
}

/**
 * @brief Minimum of two values
 * @return The smaller of a or b
 */
inline static size_t math_min(size_t a, size_t b) {
    return a < b ? a : b;
}

/**
 * @brief Saturating subtraction (clamps to 0)
 *
 * Computes max(0, a - b). Returns 0 if b >= a instead of underflowing.
 *
 * @param a Minuend
 * @param b Subtrahend
 * @return a - b if a > b, otherwise 0
 */
inline static size_t math_sub_sat(size_t a, size_t b) {
    return a > b ? a - b : 0;
}

/**
 * @brief Round down to nearest power-of-2 multiple
 *
 * Aligns a number downward to a power-of-2 boundary.
 *
 * @param number Value to align
 * @param multiple Power-of-2 alignment (must be 2^n)
 * @return Largest multiple of 'multiple' that is <= number
 *
 * Example: math_round_down_po2(13, 8) = 8
 */
static size_t math_round_down_po2(size_t number, size_t multiple) {
    assert(multiple != 0);                      /* Multiple must be non-zero */
    assert((multiple & (multiple - 1)) == 0);   /* Multiple must be power of 2 */

    return number & -multiple;
}

/**
 * @brief Round up to nearest power-of-2 multiple
 *
 * Aligns a number upward to a power-of-2 boundary.
 *
 * @param number Value to align
 * @param multiple Power-of-2 alignment (must be 2^n)
 * @return Smallest multiple of 'multiple' that is >= number
 *
 * Example: math_round_up_po2(13, 8) = 16
 */
static size_t math_round_up_po2(size_t number, size_t multiple) {
    assert(multiple != 0);                      /* Multiple must be non-zero */
    assert((multiple & (multiple - 1)) == 0);   /* Multiple must be power of 2 */

    const size_t multiple_mask = multiple - 1;
    if ((number & multiple_mask) != 0) {
        number |= multiple_mask;
        number += 1;
    }
    return number;
}
