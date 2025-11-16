/**
 * @file rng.h
 * @brief Pseudo-random number generator for sampling
 *
 * This header provides a fast, high-quality RNG based on the Squares algorithm.
 * Used for token sampling during text generation.
 *
 * The Squares RNG is:
 * - Fast: Suitable for GPU execution
 * - High quality: Passes statistical randomness tests
 * - Seedable: Deterministic for reproducibility
 * - Counter-based: Can generate independent streams
 *
 * Reference: "Squares: A Fast Counter-Based RNG" by Bernard Widynski
 */
#pragma once

#include <stdint.h>

/**
 * @brief Generate a 32-bit pseudo-random number using Squares algorithm
 *
 * Counter-based RNG that produces deterministic random numbers from
 * an offset and seed. Each (offset, seed) pair produces a unique
 * pseudo-random value.
 *
 * @param offset Counter/position value (unique per sample)
 * @param seed Random seed for reproducibility
 * @return 32-bit pseudo-random value
 *
 * Properties:
 * - Deterministic: Same (offset, seed) always produces same output
 * - Independent: Different offsets produce independent random values
 * - Fast: 4 rounds of multiply-add-rotate operations
 * - Quality: Passes BigCrush statistical test suite
 *
 * Usage:
 * @code
 * uint32_t rand1 = rng_squares32(0, seed);  // First random number
 * uint32_t rand2 = rng_squares32(1, seed);  // Second random number
 * @endcode
 */
inline static uint32_t rng_squares32(uint64_t offset, uint64_t seed) {
    const uint64_t y = offset * seed;
    const uint64_t z = y + seed;

    /* Round 1 */
    uint64_t x = y * y + y;
    x = (x >> 32) | (x << 32);

    /* Round 2 */
    x = x * x + z;
    x = (x >> 32) | (x << 32);

    /* Round 3 */
    x = x * x + y;
    x = (x >> 32) | (x << 32);

    /* Round 4 */
    x = x * x + z;
    return (uint32_t) (x >> 32);
}
