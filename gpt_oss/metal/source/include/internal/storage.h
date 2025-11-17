/**
 * @file storage.h
 * @brief File format definitions for GPT-OSS model and tokenizer files
 *
 * This header defines the binary file format structures for:
 * - Model files (.gptoss): Architecture config + weights
 * - Tokenizer files: Vocabulary + special tokens
 *
 * File structure:
 * 1. gptoss_file_header: Magic number and version
 * 2. Type-specific header (model or tokenizer)
 * 3. Binary data (weights, vocabulary, etc.)
 *
 * Files are designed for memory mapping (mmap) with proper alignment.
 * All multi-byte values are stored in native endianness.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * @struct gptoss_file_header
 * @brief Common header for all GPT-OSS binary files
 *
 * Appears at the start of every GPT-OSS file. The magic number
 * identifies the file type, and zero is reserved for future version info.
 */
struct gptoss_file_header {
    char magic[12];    /* File type magic string (e.g., "GPTOSS\0\0\0\0\0\0") */
    uint32_t zero;     /* Reserved for version/flags (currently must be 0) */
};

/**
 * @struct gptoss_gptoss_model_header
 * @brief Model architecture configuration header
 *
 * Follows the file header in model files. Contains all hyperparameters
 * needed to construct the model architecture and allocate buffers.
 * Matches the fields in struct gptoss_model for easy loading.
 */
struct gptoss_gptoss_model_header {
    uint32_t context_length;        /* Maximum sequence length */
    uint32_t num_blocks;            /* Number of transformer layers */
    uint32_t num_experts;           /* Experts per MoE layer (0 for dense) */
    uint32_t num_active_experts;    /* Active experts per token (K in top-K) */
    uint32_t embedding_dim;         /* Model hidden dimension */
    uint32_t mlp_dim;               /* MLP intermediate dimension */
    float swiglu_limit;             /* SwiGLU activation clamping limit */
    uint32_t head_dim;              /* Dimension per attention head */
    uint32_t num_heads;             /* Number of query heads */
    uint32_t num_kv_heads;          /* Number of key/value heads (GQA) */
    uint32_t attention_window;      /* Sliding window size (0 = full attention) */
    float rope_theta;               /* RoPE base frequency */
    float interpolation_scale;      /* Position interpolation scale */
    float yarn_offset;              /* YaRN offset parameter */
    float yarn_scale;               /* YaRN scale factor */
    float yarn_multiplier;          /* YaRN frequency multiplier */
    float rmsnorm_epsilon;          /* RMSNorm epsilon for stability */
};

/**
 * @struct gptoss_tiktoken_tokenizer_header
 * @brief Tokenizer vocabulary header
 *
 * Follows the file header in tokenizer files. Describes the vocabulary
 * layout and sizes of variable-length data that follows.
 */
struct gptoss_tiktoken_tokenizer_header {
    uint32_t num_special_tokens;  /* Count of special (control) tokens */
    uint32_t num_text_tokens;     /* Count of regular text tokens */
    uint32_t regex_size;          /* Size of tokenization regex pattern in bytes */
    uint32_t tokens_size;         /* Size of token string table in bytes */
};
