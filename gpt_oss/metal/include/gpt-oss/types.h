/**
 * @file types.h
 * @brief Core type definitions for the GPT-OSS API
 *
 * This header defines all public types used by the GPT-OSS library including:
 * - Status codes for error handling
 * - Special token identifiers
 * - Opaque handle types for API objects
 *
 * All API functions return a gptoss_status enum to indicate success or failure.
 * Objects (Model, Context, Tokenizer, Sampler) are represented as opaque pointers
 * and managed via reference counting.
 */
#pragma once

/**
 * @enum gptoss_status
 * @brief Status codes returned by all GPT-OSS API functions
 *
 * All API functions return one of these status codes to indicate success or the
 * specific type of error that occurred. Applications should check for
 * gptoss_status_success before using any output values.
 */
enum gptoss_status {
    /** Operation completed successfully */
    gptoss_status_success = 0,

    /** One or more arguments to the function were invalid (e.g., NULL pointer, out of range) */
    gptoss_status_invalid_argument = 1,

    /** The argument values are valid but not supported by this implementation */
    gptoss_status_unsupported_argument = 2,

    /** The operation is not valid in the current state of the object */
    gptoss_status_invalid_state = 3,

    /** An I/O error occurred (e.g., file not found, read/write failure) */
    gptoss_status_io_error = 4,

    /** Insufficient system memory to complete the operation */
    gptoss_status_insufficient_memory = 5,

    /** Insufficient GPU or other resources to complete the operation */
    gptoss_status_insufficient_resources = 6,

    /** The current system configuration is not supported (e.g., no Metal support) */
    gptoss_status_unsupported_system = 7,

    /** The context would overflow its maximum capacity with this operation */
    gptoss_status_context_overflow = 8,
};

/**
 * @enum gptoss_special_token
 * @brief Identifiers for special tokens in the vocabulary
 *
 * Special tokens are control tokens used for structuring model input/output.
 * These tokens mark message boundaries, indicate content types, and control
 * generation behavior. The actual token IDs vary per model and must be queried
 * via gptoss_tokenizer_get_special_token_id().
 */
enum gptoss_special_token {
    /** Invalid/undefined special token */
    gptoss_special_token_invalid = 0,

    /** Newline/return token for line breaks */
    gptoss_special_token_return = 1,

    /** Start of message/sequence token */
    gptoss_special_token_start = 2,

    /** Message content marker */
    gptoss_special_token_message = 3,

    /** End of message/sequence token */
    gptoss_special_token_end = 4,

    /** Token marking start of refusal content */
    gptoss_special_token_refusal = 5,

    /** Token for content constraints/conditions */
    gptoss_special_token_constrain = 6,

    /** Channel selection/routing token */
    gptoss_special_token_channel = 7,

    /** Function/tool call invocation token */
    gptoss_special_token_call = 8,

    /** Start of untrusted/user-provided content */
    gptoss_special_token_untrusted = 9,

    /** End of untrusted/user-provided content */
    gptoss_special_token_end_untrusted = 10,

    /** Sentinel value marking the end of the enum */
    gptoss_special_token_max,
};

/**
 * @typedef gptoss_model_t
 * @brief Opaque handle to a loaded GPT model
 *
 * The Model object encapsulates all resources needed to run inference:
 * - Model weights (parameters) loaded from disk
 * - Model architecture configuration (layer count, dimensions, etc.)
 * - Metal compute pipeline states for GPU execution
 * - Temporary computation buffers
 * - Associated tokenizer
 *
 * Models are loaded from GPT-OSS format files via gptoss_model_create_from_file().
 * Multiple Context objects can share the same Model for concurrent inference.
 * Models use reference counting and must be released with gptoss_model_release().
 *
 * Thread safety: Model objects can be safely shared across threads after creation.
 */
typedef struct gptoss_model* gptoss_model_t;

/**
 * @typedef gptoss_tokenizer_t
 * @brief Opaque handle to a tokenizer for text encoding/decoding
 *
 * The Tokenizer object provides text-to-token conversion and vice versa:
 * - Vocabulary of text tokens and special tokens
 * - Byte-level token representations
 * - Special token ID mappings
 *
 * Tokenizers are obtained from Model objects via gptoss_model_get_tokenizer()
 * and share the Model's lifetime. They use reference counting and should be
 * released with gptoss_tokenizer_release() when no longer needed.
 *
 * Thread safety: Tokenizer objects are immutable and thread-safe for concurrent use.
 */
typedef struct gptoss_tokenizer* gptoss_tokenizer_t;

/**
 * @typedef gptoss_context_t
 * @brief Opaque handle to an inference context with cached state
 *
 * The Context object maintains the state for text generation:
 * - Sequence of input tokens processed so far
 * - KV (key-value) cache for efficient autoregressive generation
 * - Output logits distribution over vocabulary
 * - Position in the sequence
 *
 * Contexts are created for a specific Model via gptoss_context_create().
 * Multiple independent Contexts can be used with the same Model for parallel
 * or multi-turn conversations. Each Context has a maximum token capacity.
 * Contexts use reference counting and must be released with gptoss_context_release().
 *
 * Memory: Each context allocates GPU memory for its KV cache proportional to
 * (max_context_length * hidden_dim * num_layers).
 *
 * Thread safety: Contexts are NOT thread-safe. Each thread should use its own Context.
 */
typedef struct gptoss_context* gptoss_context_t;

/**
 * @typedef gptoss_sampler_t
 * @brief Opaque handle to a sampling configuration
 *
 * The Sampler object encapsulates parameters that control token sampling:
 * - Temperature: Controls randomness (0.0 = greedy, higher = more random)
 * - Top-p (nucleus sampling): Cumulative probability threshold
 * - Frequency penalty: Reduces repetition based on token frequency
 * - Presence penalty: Reduces repetition based on token presence
 *
 * Samplers are created via gptoss_sampler_create() and can be reused across
 * multiple sampling operations. They use reference counting and must be released
 * with gptoss_sampler_release().
 *
 * Thread safety: Samplers are mutable and NOT thread-safe. Use separate Samplers
 * per thread or synchronize access.
 */
typedef struct gptoss_sampler* gptoss_sampler_t;
