/**
 * @file macros.h
 * @brief Preprocessor macros for the GPT-OSS API
 *
 * This header defines macros used throughout the GPT-OSS public API,
 * primarily for controlling symbol visibility and calling conventions.
 */
#pragma once

/**
 * @def GPTOSS_ABI
 * @brief Application Binary Interface attribute for public API functions
 *
 * This macro controls the calling convention and symbol visibility for all
 * public GPT-OSS API functions. It can be defined before including gpt-oss.h
 * to customize the ABI:
 *
 * - Default (undefined): No special attributes, standard C linkage
 * - __declspec(dllexport): Windows DLL export (when building shared library)
 * - __declspec(dllimport): Windows DLL import (when using shared library)
 * - __attribute__((visibility("default"))): ELF symbol visibility control
 *
 * Example usage when building a Windows DLL:
 * @code
 * #define GPTOSS_ABI __declspec(dllexport)
 * #include <gpt-oss.h>
 * @endcode
 *
 * For most users, the default (empty) definition is appropriate.
 */
#ifndef GPTOSS_ABI
    #define GPTOSS_ABI
#endif  // GPTOSS_ABI
