/**
 * @file gpt-oss.h
 * @brief Main header file for the GPT-OSS Metal inference library
 *
 * This is the primary umbrella header that includes all public API definitions
 * for the GPT-OSS library. Include this single header to access the complete
 * Metal-accelerated GPT inference API.
 *
 * The GPT-OSS library provides a C API for running GPT-based language models
 * on Apple Silicon using the Metal Performance Shaders framework. It supports:
 * - Loading models in GPT-OSS format
 * - Token encoding/decoding via built-in tokenizer
 * - Context management with KV caching
 * - Customizable sampling strategies
 * - Multiple concurrent inference contexts
 *
 * Basic usage:
 * 1. Create a model: gptoss_model_create_from_file()
 * 2. Create a context: gptoss_context_create()
 * 3. Add tokens: gptoss_context_append_tokens() or gptoss_context_append_chars()
 * 4. Process context: gptoss_context_process()
 * 5. Sample next token: gptoss_context_sample()
 * 6. Clean up: gptoss_context_release(), gptoss_model_release()
 *
 * All objects use reference counting for memory management.
 */
#pragma once

#include <gpt-oss/macros.h>
#include <gpt-oss/types.h>
#include <gpt-oss/functions.h>
