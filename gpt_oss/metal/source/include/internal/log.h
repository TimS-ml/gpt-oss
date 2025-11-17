/**
 * @file log.h
 * @brief Internal logging utilities for GPT-OSS
 *
 * This header provides logging functions for error and warning messages
 * from the GPT-OSS library internals. The logging system uses printf-style
 * formatting and writes to stderr by default.
 *
 * Usage:
 * @code
 * GPTOSS_LOG_ERROR("Failed to allocate %zu bytes", size);
 * GPTOSS_LOG_WARNING("Using default value %d", default_val);
 * @endcode
 *
 * The logging implementation is defined in log.c and can be customized
 * if needed for integration with application-specific logging systems.
 */
#pragma once

#include <stdarg.h>

/**
 * @brief Internal function to format and output log messages
 *
 * This function is called by gptoss_log() with a va_list of arguments.
 * It formats the message according to the printf-style format string
 * and writes it to the configured output (typically stderr).
 *
 * @param format Printf-style format string
 * @param args Variable argument list
 *
 * Thread safety: Implementation should be thread-safe for concurrent logging.
 */
void gptoss_format_log(const char* format, va_list args);

/**
 * @brief Log a formatted message
 *
 * Printf-style logging function. The format attribute enables compiler
 * checking of format string vs. arguments for type safety.
 *
 * @param format Printf-style format string
 * @param ... Variable arguments matching format specifiers
 */
__attribute__((__format__(__printf__, 1, 2)))
inline static void gptoss_log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    gptoss_format_log(format, args);
    va_end(args);
}

/**
 * @def GPTOSS_LOG_ERROR
 * @brief Log an error message
 *
 * Logs an error message prefixed with "Error: " and followed by newline.
 * Use this for recoverable errors that don't require program termination.
 *
 * @param message Format string (without newline)
 * @param ... Format arguments
 *
 * Example: GPTOSS_LOG_ERROR("Failed to open file: %s", filename);
 */
#define GPTOSS_LOG_ERROR(message, ...) \
    gptoss_log("Error: " message "\n", ##__VA_ARGS__)

/**
 * @def GPTOSS_LOG_WARNING
 * @brief Log a warning message
 *
 * Logs a warning message prefixed with "Warning: " and followed by newline.
 * Use this for non-critical issues that may affect performance or behavior.
 *
 * @param message Format string (without newline)
 * @param ... Format arguments
 *
 * Example: GPTOSS_LOG_WARNING("Using fallback value: %d", value);
 */
#define GPTOSS_LOG_WARNING(message, ...) \
    gptoss_log("Warning: " message "\n", ##__VA_ARGS__)
