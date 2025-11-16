/**
 * @file model.h
 * @brief Internal structure definitions for GPT-OSS model, tokenizer, and context
 *
 * This header defines the internal representation of the main GPT-OSS objects:
 * - gptoss_tokenizer: Vocabulary and token conversion
 * - gptoss_model: Model weights, architecture config, Metal pipelines
 * - gptoss_context: Inference state, KV cache, activation buffers
 *
 * These are implementation details not exposed in the public API. The structures
 * are accessed only through opaque pointers in the public API (gpt-oss/types.h).
 *
 * Memory management:
 * All structures use atomic reference counting for thread-safe lifetime management.
 * Models and contexts allocate GPU buffers for weights and activations.
 *
 * Architecture:
 * - Models are immutable after loading and can be shared across threads
 * - Contexts are per-inference-session and NOT thread-safe
 * - Tokenizers are immutable and thread-safe
 */
#pragma once

#ifndef __cplusplus
    #include <stdatomic.h>
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "internal/metal.h"

/**
 * @struct gptoss_tokenizer
 * @brief Internal tokenizer structure containing vocabulary and mappings
 *
 * The tokenizer handles conversion between text and token IDs. It contains:
 * - Memory-mapped vocabulary data
 * - Regex pattern for tokenization (if applicable)
 * - Token ID to byte string mappings
 * - Special token ID mappings
 *
 * Vocabulary layout:
 * - Text tokens: IDs 0 to num_text_tokens-1
 * - Special tokens: IDs num_text_tokens to num_text_tokens+num_special_tokens-1
 *
 * Memory ownership:
 * - mapping_ptr points to mmap'd file data (read-only, shared)
 * - regex_ptr and tokens_ptr point into the mapped region
 *
 * Thread safety: Immutable after initialization, safe for concurrent reads.
 */
struct gptoss_tokenizer {
#ifndef __cplusplus
    atomic_uint_least64_t ref_count;  /* Atomic reference count for thread-safe lifetime */
#else
    uint_least64_t ref_count;          /* Non-atomic for C++ (wrapped differently) */
#endif

    void* mapping_ptr;      /* Memory-mapped tokenizer file base address */
    size_t mapping_size;    /* Size of memory-mapped region in bytes */

    const char* regex_ptr;  /* Pointer to tokenization regex pattern (within mapping) */
    const char* tokens_ptr; /* Pointer to token strings table (within mapping) */

    uint32_t num_text_tokens;    /* Number of regular text tokens in vocabulary */
    uint32_t num_special_tokens; /* Number of special control tokens */

    /* Mapping from special token enum to actual token ID in vocabulary */
    uint32_t special_token_id[gptoss_special_token_max - 1];
};

/**
 * @struct gptoss_model
 * @brief Internal model structure containing weights, config, and Metal resources
 *
 * The model represents a loaded GPT transformer including:
 * - Architecture configuration (layers, dimensions, attention params)
 * - Model weights (memory-mapped from file, uploaded to GPU)
 * - Metal compute pipelines for each operation
 * - Associated tokenizer
 *
 * Memory layout:
 * - Model file is memory-mapped (read-only, potentially shared across processes)
 * - Weights are uploaded to GPU in Metal buffers
 * - Shared weights (embeddings, etc.) in one buffer
 * - Per-block weights (MoE experts) in separate buffers for each layer
 *
 * Architecture:
 * Supports GPT-style transformers with:
 * - Multi-head attention with optional grouped-query attention
 * - RMSNorm for normalization
 * - SwiGLU activation
 * - Mixture of Experts (MoE) layers
 * - RoPE (Rotary Position Embeddings)
 *
 * Thread safety: Immutable after creation, safe for concurrent inference
 * (multiple Contexts can share one Model).
 */
struct gptoss_model {
#ifndef __cplusplus
    atomic_uint_least64_t ref_count;  /* Atomic reference count */
#else
    uint_least64_t ref_count;          /* Non-atomic for C++ */
#endif

    struct gptoss_tokenizer* tokenizer;  /* Associated tokenizer (owned reference) */

    /* Memory-mapped model file */
    void* mapping_ptr;      /* Base address of mmap'd model file */
    size_t mapping_size;    /* Size of mapped region in bytes */

    /* Architecture configuration - loaded from model file header */
    uint32_t context_length;        /* Maximum sequence length (e.g., 2048, 4096) */
    uint32_t num_blocks;            /* Number of transformer blocks/layers */
    uint32_t num_experts;           /* Total experts per MoE layer (0 if not MoE) */
    uint32_t num_active_experts;    /* Active experts per token (top-K for MoE) */
    uint32_t embedding_dim;         /* Hidden dimension / model dimension */
    uint32_t mlp_dim;               /* MLP intermediate dimension */
    float swiglu_limit;             /* SwiGLU clamping limit for numerical stability */
    uint32_t head_dim;              /* Dimension per attention head */
    uint32_t num_heads;             /* Number of query heads (multi-head attention) */
    uint32_t num_kv_heads;          /* Number of key/value heads (GQA if < num_heads) */
    uint32_t attention_window;      /* Sliding window size (0 for full attention) */

    /* RoPE (Rotary Position Embedding) configuration */
    float rope_theta;               /* Base frequency for RoPE (often 10000.0) */
    float interpolation_scale;      /* Position interpolation scale for long context */
    float yarn_offset;              /* YaRN (Yet another RoPE) offset parameter */
    float yarn_scale;               /* YaRN scaling factor */
    float yarn_multiplier;          /* YaRN frequency multiplier */

    /* Normalization */
    float rmsnorm_epsilon;          /* Epsilon for RMSNorm numerical stability */

    uint32_t vocabulary_size;       /* Total tokens in vocabulary */

    bool lock_memory;               /* Whether to lock mapped memory pages (mlock) */

    /* Memory allocation sizes */
    size_t weights_size;      /* Total size of model weights in bytes */
    size_t allocation_size;   /* Total GPU memory allocated for weights */

    /* Metal GPU resources - device, queues, and compute library */
    struct gptoss_metal_device device;            /* Metal device (GPU) */
    size_t max_threadgroups;                      /* Max concurrent threadgroups for this device */
    struct gptoss_metal_command_queue command_queue;  /* Command queue for submitting work */
    struct gptoss_metal_library library;          /* Compiled Metal shader library */

    /* Metal compute pipeline functions - each represents a compiled shader kernel */
    /* Naming convention: output_input_operation (e.g., f32_bf16w_matmul = FP32 output, BF16 weights, matmul) */
    struct gptoss_metal_function bf16_f32_embeddings_fn;                /* Token embedding lookup */
    struct gptoss_metal_function f32_bf16w_rmsnorm_fn;                  /* RMSNorm with BF16 weights */
    struct gptoss_metal_function f32_bf16w_matmul_fn;                   /* General matmul (sparse MoE) */
    struct gptoss_metal_function f32_bf16w_matmul_qkv_fn;               /* QKV projection (sparse MoE) */
    struct gptoss_metal_function f32_bf16w_dense_matmul_qkv_fn;         /* QKV projection (dense) */
    struct gptoss_metal_function f32_bf16w_dense_matmul_attn_output_fn; /* Attention output projection */
    struct gptoss_metal_function f32_bf16w_dense_matmul_mlp_gate_fn;    /* MLP gating projection */
    struct gptoss_metal_function f32_bf16w_unembedding_fn;              /* Final logits projection */
    struct gptoss_metal_function f32_rope_fn;                           /* RoPE (rotary position embeddings) */
    struct gptoss_metal_function f32_mf4w_moe_matmul_swiglu_fn;         /* MoE matmul + SwiGLU (4-bit weights) */
    struct gptoss_metal_function f32_mf4w_moe_matmul_fn;                /* MoE matmul (4-bit weights) */
    struct gptoss_metal_function f32_accumulate_e4_fn;                  /* Accumulate expert outputs (up to 4) */
    struct gptoss_metal_function f32_scatter_e4_fn;                     /* Scatter tokens to experts (up to 4) */
    struct gptoss_metal_function f32_mf4w_moe_dense_matmul_swiglu_fn;   /* Dense MoE matmul + SwiGLU */
    struct gptoss_metal_function f32_mf4w_moe_dense_matmul_fn;          /* Dense MoE matmul */
    struct gptoss_metal_function f32_gather_and_accumulate_e4_fn;       /* Gather+accumulate expert outputs */
    struct gptoss_metal_function f32_expert_routing_metadata_fn;        /* Compute expert routing metadata */
    struct gptoss_metal_function f32_topk_softmax_e32_k4_fn;            /* Top-K softmax (32 experts, K=4) */
    struct gptoss_metal_function f32_topk_softmax_e128_k4_fn;           /* Top-K softmax (128 experts, K=4) */
    struct gptoss_metal_function f32_sdpa_q8_d64_fn;                    /* Scaled dot-product attention */
    struct gptoss_metal_function f32_softmax_fn;                        /* Standard softmax */
    struct gptoss_metal_function f32_sample_fn;                         /* Token sampling from logits */

    /* Weight buffer sizes */
    size_t per_block_shared_weights_size;   /* Shared weights per transformer block (bytes) */
    size_t per_expert_block_weight_size;    /* MoE expert weights per block (bytes) */

    /* Threadgroup sizes for each kernel - determines GPU occupancy */
    size_t embeddings_threadgroup_size;     /* Threads per threadgroup for embeddings */
    size_t attn_qkv_threadgroup_size;       /* Threads per threadgroup for attention QKV projection */
    size_t attn_out_threadgroup_size;       /* Threads per threadgroup for attention output */
    size_t mlp_gate_threadgroup_size;       /* Threads per threadgroup for MLP gating */
    size_t mlp_swiglu_threadgroup_size;     /* Threads per threadgroup for SwiGLU */
    size_t mlp_out_threadgroup_size;        /* Threads per threadgroup for MLP output */
    size_t mlp_acc_threadgroup_size;        /* Threads per threadgroup for MLP accumulation */
    size_t unembedding_threadgroup_size;    /* Threads per threadgroup for unembedding */

    /* Weight buffer offsets - pointers into shared_weight_buffer and block_weight_buffers */
    /* These offsets are relative to the start of each buffer */
    size_t attn_rmsnorm_gain_offset;        /* Offset to attention RMSNorm gains */
    size_t attn_qkv_weight_offset;          /* Offset to QKV projection weights */
    size_t attn_qkv_bias_offset;            /* Offset to QKV projection biases */
    size_t attn_sdpa_sink_offset;           /* Offset to attention sink tokens (for sliding window) */
    size_t attn_out_weight_offset;          /* Offset to attention output projection weights */
    size_t attn_out_bias_offset;            /* Offset to attention output projection biases */
    size_t mlp_rmsnorm_gain_offset;         /* Offset to MLP RMSNorm gains */
    size_t mlp_gate_weight_offset;          /* Offset to MLP/MoE gate weights */
    size_t mlp_gate_bias_offset;            /* Offset to MLP/MoE gate biases */
    size_t mlp_swiglu_scale_offset;         /* Offset to SwiGLU scaling factors */
    size_t mlp_swiglu_bias_offset;          /* Offset to SwiGLU biases */
    size_t mlp_out_block_offset;            /* Offset to MLP output block start */
    size_t mlp_out_scale_offset;            /* Offset to MLP output scaling factors */
    size_t mlp_out_bias_offset;             /* Offset to MLP output biases */
    size_t rmsnorm_weight_offset;           /* Offset to final RMSNorm weights */
    size_t unembedding_weight_offset;       /* Offset to unembedding matrix (vocab projection) */

    /* GPU weight buffers */
    struct gptoss_metal_buffer shared_weight_buffer;  /* Shared weights (embeddings, norms, gates) */
    struct gptoss_metal_buffer block_weight_buffers[];  /* Flexible array: per-block MoE expert weights */
};

/**
 * @def GPTOSS_DEFAULT_BATCH_SIZE
 * @brief Default maximum batch size for token processing
 *
 * This is the default number of tokens that can be processed in a single
 * forward pass (prefill or decode). Larger values improve throughput for
 * long prompts but require more GPU memory for activation buffers.
 */
#define GPTOSS_DEFAULT_BATCH_SIZE 128

/**
 * @struct gptoss_context
 * @brief Internal context structure for inference state and KV cache
 *
 * The context maintains all state for a single inference session:
 * - Token sequence being processed
 * - KV cache for autoregressive generation
 * - Activation buffers for intermediate computations
 * - Input/output buffers for tokens and logits
 *
 * Memory layout:
 * - All buffers are allocated in GPU memory (Metal buffers)
 * - KV cache size is proportional to (max_tokens * num_kv_heads * head_dim * num_blocks)
 * - Activation buffers sized for max_batch_tokens
 *
 * Processing flow:
 * 1. Append tokens to token_buffer
 * 2. Process: Run transformer layers, updating KV cache and computing logits
 * 3. Sample: Choose next token(s) from logits distribution
 * 4. Repeat or reset for new sequence
 *
 * Thread safety: NOT thread-safe. Each thread should use its own Context.
 */
struct gptoss_context {
#ifndef __cplusplus
    atomic_uint_least64_t ref_count;  /* Atomic reference count */
#else
    uint_least64_t ref_count;          /* Non-atomic for C++ */
#endif

    struct gptoss_model* model;  /* Associated model (owned reference) */

    /* Token sequence state */
    size_t num_tokens;           /* Number of tokens appended to context (total sequence length) */
    size_t num_kv_tokens;        /* Number of tokens with cached K/V (may be < num_tokens if batching) */
    size_t max_tokens;           /* Maximum context length (capacity) */
    size_t max_batch_tokens;     /* Maximum tokens processed per batch (activation buffer size) */

    /* Memory allocation tracking */
    size_t kvcache_size;         /* Total KV cache size in bytes */
    size_t allocation_size;      /* Total GPU memory allocated for this context */

    /* Activation buffers - intermediate computation results during forward pass */
    /* TODO: Merge these into a single buffer with offsets for better memory efficiency */
    struct gptoss_metal_buffer residual_activation_buffer;        /* Residual stream (main data flow) */
    struct gptoss_metal_buffer rmsnorm_activation_buffer;         /* RMSNorm output (attention & MLP) */
    struct gptoss_metal_buffer qkv_activation_buffer;             /* QKV projection output (before SDPA) */
    struct gptoss_metal_buffer sdpa_activation_buffer;            /* Scaled dot-product attention output */
    struct gptoss_metal_buffer gate_activation_buffer;            /* MoE gating logits (expert selection) */
    struct gptoss_metal_buffer expert_activation_buffer;          /* MoE expert predictions (all experts) */
    struct gptoss_metal_buffer expert_offset_buffer;              /* MoE expert histogram cumsum (routing) */
    struct gptoss_metal_buffer token_to_expert_routing_buffer;    /* MoE token-to-expert assignment map */
    struct gptoss_metal_buffer swiglu_input_buffer;               /* SwiGLU input (prefill path only) */
    struct gptoss_metal_buffer swiglu_activation_buffer;          /* SwiGLU gated output */
    struct gptoss_metal_buffer moe_activation_buffer;             /* MoE MLP output per active expert */

    /* Input/output buffers */
    struct gptoss_metal_buffer control_buffer;    /* Control/metadata buffer for kernel params */
    struct gptoss_metal_buffer token_buffer;      /* Input token IDs (uint32) */
    struct gptoss_metal_buffer score_buffer;      /* Output logits (unembedding results) */
    struct gptoss_metal_buffer prob_buffer;       /* Output probabilities (after softmax) */
    struct gptoss_metal_buffer sum_buffer;        /* Intermediate sum reductions */
    struct gptoss_metal_buffer argmax_buffer;     /* Argmax results for token selection */
    struct gptoss_metal_buffer kvcache_buffer;    /* KV cache (keys and values for all layers) */
};
