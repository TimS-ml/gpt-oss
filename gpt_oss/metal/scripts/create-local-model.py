"""
Model Conversion Script for Metal Inference Format

This script converts gpt-oss model checkpoints from standard PyTorch/SafeTensors
format to an optimized binary format for Metal inference on Apple Silicon.

The conversion process includes:
1. Loading model configuration and weights from SafeTensors
2. Setting up the tiktoken tokenizer with special tokens
3. Writing a custom binary format optimized for Metal GPU loading
4. Quantizing MoE weights to MXFP4 format for efficiency
5. Applying optimizations like fused attention scaling

Output Format:
- Custom binary file with headers for model config and tokenizer
- Aligned memory layout for efficient Metal buffer loading
- Quantized weights (FP8, MXFP4) for reduced memory and faster inference
- Special tokens encoded with UUIDs for structured generation

Metal Optimizations:
- 16KB alignment for GPU memory pages
- MXFP4 block quantization for MoE layers (reduces memory 8x)
- FP8 (E4M3) for embeddings and attention weights
- Fused 1/sqrt(head_dim) scaling into Q/K projections
- Optimized weight layout for Metal compute shaders
"""

import argparse
import os
import math
import sys
import json
import itertools
import struct
from uuid import UUID

import tiktoken

import torch
from safetensors import safe_open
from tqdm import tqdm
from openai_harmony import load_harmony_encoding, HarmonyEncodingName

# Command-line argument parser for conversion options
parser = argparse.ArgumentParser(prog='create-local-model.py', description='Convert a checkpoint directory to a local model file')
parser.add_argument('-s', '--src', metavar='DIR', type=str, required=True, help='Path to the input checkpoint directory')
parser.add_argument('-d', '--dst', metavar='FILE', type=str, required=True, help='Path to the output model file')


# Load the base o200k tokenizer (200K vocabulary BPE tokenizer)
# This is the foundation tokenizer similar to GPT-4's encoding
o200k_base = tiktoken.get_encoding("o200k_base")

# Load Harmony encoding to identify special tokens in the gpt-oss vocabulary
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Create extended tokenizer with gpt-oss specific special tokens
# These special tokens enable structured generation, tool use, and safety features
o200k_gptoss = tiktoken.Encoding(
    name="o200k_gptoss",
    pat_str=o200k_base._pat_str,  # Use same regex pattern as o200k_base
    mergeable_ranks=o200k_base._mergeable_ranks,  # Use same BPE merge table
    special_tokens={
        "<|reversed199998|>": 199998,  # unused (reserved)
        "<|endoftext|>": 199999,       # Standard end of text marker
        "<|untrusted|>": 200000,       # Marks beginning of untrusted content
        "<|endofuntrusted|>": 200001,  # Marks end of untrusted content
        "<|return|>": 200002,          # Signals end of assistant turn
        "<|constrain|>": 200003,       # Enables output constraints
        "<|reversed200004|>": 200004,  # unused (reserved)
        "<|channel|>": 200005,         # Separates channel name in structured output
        "<|start|>": 200006,           # Starts a message block
        "<|end|>": 200007,             # Ends a message block
        "<|message|>": 200008,         # Separates role from message content
        "<|reversed200008|>": 200008,  # unused (reserved)
        "<|reversed200009|>": 200009,  # unused (reserved)
        "<|reversed200010|>": 200010,  # unused (reserved)
        "<|reversed200011|>": 200011,  # unused (reserved)
        "<|call|>": 200012,            # Indicates function/tool call
        "<|refusal|>": 200013,         # Indicates content policy refusal
    }
)

# Magic bytes that identify this as a gpt-oss Metal model file (version 1.0)
FILE_MAGIC = struct.pack('ccccccccccccI', b'G', b'P', b'T', b'-', b'O', b'S', b'S', b' ', b'v', b'1', b'.', b'0', 0)

# UUID identifiers for special tokens (enables fast token recognition in Metal)
# Each special token gets a unique UUID stored in the binary format
SPECIAL_TOKEN_UUID = {
    '<|start|>': UUID('55a77c2f-8a01-4c54-8ac2-313bfc7e208d').bytes,
    '<|message|>': UUID('16e40431-f47f-4b22-b59b-8b278fc30a54').bytes,
    '<|end|>': UUID('fcac2f6d-4705-4f6b-b228-642accac7238').bytes,
    '<|return|>': UUID('f799ff69-1992-43c4-a3d8-d831f475dc75').bytes,
    '<|refusal|>': UUID('e15ba702-28c4-4292-ab8f-ffa434709128').bytes,
    '<|constrain|>': UUID('c0bb14c7-6022-49da-ad08-792d67e8b470').bytes,
    '<|channel|>': UUID('fd3dda11-c8ab-4033-876e-d93deb172c93').bytes,
    '<|call|>': UUID('1220f796-e388-4de5-b487-fe2eb5fe03c0').bytes,
    '<|untrusted|>': UUID('07d7da55-b346-4cff-8b37-7cefacf8a3e8').bytes,
    '<|end_untrusted|>': UUID('f265bd9c-c717-469e-a447-920687d65d90').bytes,
}

# List of special tokens to include in the converted model
# Only these special tokens will be written to the output file
INCLUDE_SPECIAL_TOKENS = [
    "<|start|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|refusal|>",
    "<|constrain|>",
    "<|channel|>",
    "<|call|>",
    "<|untrusted|>",
    "<|end_untrusted|>",
]

# UUID identifiers for model architecture and layout versions
# These enable the Metal loader to verify compatibility
GPTOSS_MODEL_UUID = UUID('df52dc86-1789-4ed0-a295-66f10508145b').bytes  # Identifies gpt-oss architecture
APPLE_GPU_LAYOUT_UUID = UUID('229177a8-5775-4268-bfd8-d588b351c56d').bytes  # Metal-optimized weight layout
TIKTOKEN_TOKENIZER_UUID = UUID('7401aded-2a95-40cb-b782-9ccebaafe72b').bytes  # Tiktoken-based tokenizer

# Offset added to MXFP4 block scales to make all values positive (enables uint8 storage)
# MXFP4 uses shared exponents per block, this bias shifts them into 0-255 range
UE8_OFFSET = 14  # bias to MXFP4 block scales

def write_file_header(f):
    """
    Write the file magic header identifying this as a gpt-oss Metal model.

    Args:
        f: Binary file handle to write to
    """
    f.write(FILE_MAGIC)

def write_tokenizer_header(f,
                           num_special_tokens: int,
                           num_text_tokens: int,
                           regex_size: int,
                           tokens_size: int):
    """
    Write tokenizer metadata to the model file.

    This header enables the Metal loader to correctly parse the tokenizer data
    that follows, including special tokens, regex pattern, and BPE token bytes.

    Args:
        f: Binary file handle to write to
        num_special_tokens: Count of special tokens (e.g., <|start|>, <|end|>)
        num_text_tokens: Count of regular BPE tokens (typically 200000)
        regex_size: Size in bytes of the BPE regex pattern string
        tokens_size: Total size in bytes of all token byte strings
    """
    f.write(TIKTOKEN_TOKENIZER_UUID)
    f.write(struct.pack('<I', num_special_tokens))  # Little-endian unsigned int
    f.write(struct.pack('<I', num_text_tokens))
    f.write(struct.pack('<I', regex_size))
    f.write(struct.pack('<I', tokens_size))

def write_model_header(f,
                       context_length : int,
                       num_blocks : int,
                       num_experts : int,
                       num_active_experts : int,
                       embedding_dim : int,
                       mlp_dim : int,
                       swiglu_limit : float,
                       head_dim: int,
                       num_heads : int,
                       num_kv_heads : int,
                       attention_window : int,
                       rope_theta : float,
                       interpolation_scale : float,
                       yarn_offset : float,
                       yarn_scale : float,
                       yarn_multiplier : float,
                       rmsnorm_epsilon : float):
    """
    Write model architecture configuration to the file header.

    This header contains all hyperparameters needed to configure the Metal
    inference engine, including model dimensions, RoPE settings, and MoE config.

    Args:
        f: Binary file handle to write to
        context_length: Maximum sequence length the model can handle
        num_blocks: Number of transformer layers
        num_experts: Total number of experts in MoE layers
        num_active_experts: Number of experts activated per token (typically 4)
        embedding_dim: Hidden dimension size (e.g., 2048)
        mlp_dim: MLP intermediate dimension per expert
        swiglu_limit: Clipping threshold for SwiGLU activation
        head_dim: Dimension per attention head (typically 64)
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value heads (GQA - fewer than query heads)
        attention_window: Sliding window size for local attention
        rope_theta: Base frequency for RoPE position embeddings
        interpolation_scale: RoPE frequency scaling factor for context extension
        yarn_offset: YaRN low-frequency adjustment offset
        yarn_scale: YaRN frequency scaling factor
        yarn_multiplier: YaRN attention scale multiplier
        rmsnorm_epsilon: Small constant for RMSNorm numerical stability
    """
    f.write(GPTOSS_MODEL_UUID)
    f.write(struct.pack('<I', context_length))          # Max context length
    f.write(struct.pack('<I', num_blocks))              # Number of transformer layers
    f.write(struct.pack('<I', num_experts))             # Total MoE experts
    f.write(struct.pack('<I', num_active_experts))      # Active experts per token
    f.write(struct.pack('<I', embedding_dim))           # Hidden dimension
    f.write(struct.pack('<I', mlp_dim))                 # MLP dimension per expert
    f.write(struct.pack('<f', swiglu_limit))            # SwiGLU clipping threshold
    f.write(struct.pack('<I', head_dim))                # Attention head dimension
    f.write(struct.pack('<I', num_heads))               # Number of Q heads
    f.write(struct.pack('<I', num_kv_heads))            # Number of KV heads (GQA)
    f.write(struct.pack('<I', attention_window))        # Sliding window size
    f.write(struct.pack('<f', rope_theta))              # RoPE base frequency
    f.write(struct.pack('<f', interpolation_scale))     # RoPE scaling factor
    f.write(struct.pack('<f', yarn_offset))             # YaRN offset parameter
    f.write(struct.pack('<f', yarn_scale))              # YaRN scale parameter
    f.write(struct.pack('<f', yarn_multiplier))         # YaRN multiplier
    f.write(struct.pack('<f', rmsnorm_epsilon))         # RMSNorm epsilon
    f.write(APPLE_GPU_LAYOUT_UUID)                      # Weight layout identifier


def write_padding(out_file, alignment_multiple=16384):
    """
    Write zero-padding bytes to align file offset to a specified boundary.

    Memory alignment is critical for efficient Metal buffer loading. The GPU
    can read aligned memory much faster than unaligned memory.

    Args:
        out_file: Binary file handle to write to
        alignment_multiple: Alignment boundary in bytes (default 16KB for GPU pages)
    """
    offset = out_file.tell()
    alignment_size = -offset % alignment_multiple
    if alignment_size != 0:
        alignment = bytes(alignment_size)
        out_file.write(alignment)


def write_embedding_weight(out_file, weight):
    """
    Write embedding weight matrix in FP8 or BF16 format.

    Embeddings are stored in FP8 E4M3 format for memory efficiency while
    maintaining acceptable precision for token embeddings.

    Args:
        out_file: Binary file handle to write to
        weight: Embedding weight tensor [vocab_size, embedding_dim]
    """
    write_padding(out_file, alignment_multiple=16)

    assert weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.bfloat16
    out_file.write(weight.view(torch.uint8).numpy().tobytes())


def write_rmsnorm_gain(out_file, gain):
    """
    Write RMSNorm gain (scale) parameters in BF16 format.

    RMSNorm gains are kept in BF16 for precision as they directly scale
    normalized activations and impact model output quality.

    Args:
        out_file: Binary file handle to write to
        gain: RMSNorm gain tensor [embedding_dim]
    """
    write_padding(out_file, alignment_multiple=16)

    assert gain.dtype == torch.bfloat16
    out_file.write(gain.view(torch.uint8).numpy().tobytes())


def write_attn_sink(out_file, sink):
    """
    Write attention sink tokens in BF16 format.

    Attention sinks are special learned embeddings added to the beginning of
    the KV cache to stabilize attention patterns over long contexts.

    Args:
        out_file: Binary file handle to write to
        sink: Attention sink tensor [num_sinks, num_kv_heads, head_dim]
    """
    write_padding(out_file, alignment_multiple=16)

    assert sink.dtype == torch.bfloat16
    out_file.write(sink.view(torch.uint8).numpy().tobytes())


def write_linear_weight(out_file, *args):
    """
    Write one or more linear layer weight/bias tensors.

    This is used for attention projections and other dense layers that
    are stored in their native precision (typically FP8 or BF16).

    Args:
        out_file: Binary file handle to write to
        *args: Variable number of weight/bias tensors to write sequentially
    """
    write_padding(out_file, alignment_multiple=16)

    for t in args:
        out_file.write(t.view(torch.uint8).numpy().tobytes())


def main(args):
    """
    Main conversion function that transforms a checkpoint to Metal format.

    This function orchestrates the entire conversion process:
    1. Loads model configuration from config.json
    2. Calculates tokenizer metadata and YaRN RoPE parameters
    3. Writes headers (file, model, tokenizer)
    4. Writes tokenizer data (special tokens, regex, BPE tokens)
    5. Writes model weights layer by layer with optimizations
    6. Applies quantization to MoE weights (MXFP4 format)

    Args:
        args: Command-line arguments (typically sys.argv[1:])

    The output file uses a custom binary format optimized for:
    - Fast loading into Metal GPU buffers (aligned memory)
    - Reduced memory footprint (quantized weights)
    - Efficient inference on Apple Silicon (optimized layout)
    """
    options = parser.parse_args(args)

    # Load model configuration from the checkpoint directory
    with open(os.path.join(options.src, "config.json"), "r") as f:
        config = json.load(f)

    # Extract model architecture hyperparameters from config
    num_blocks = config["num_hidden_layers"]           # Number of transformer layers
    num_experts = config["num_experts"]                # Total experts in MoE
    num_active_experts = 4                             # Top-k experts per token
    num_q_heads = config["num_attention_heads"]        # Query heads (attention)
    num_kv_heads = config["num_key_value_heads"]       # KV heads (GQA)
    head_dim = config["head_dim"]                      # Dimension per head
    embedding_dim = config["hidden_size"]              # Model hidden dimension
    mlp_dim = config["intermediate_size"]              # MLP dimension per expert
    swiglu_limit = config.get("swiglu_limit", 7.0)     # SwiGLU clipping threshold
    rope_theta = config["rope_theta"]                  # RoPE base frequency
    attention_window = config["sliding_window"]        # Local attention window
    initial_context_length = config["initial_context_length"]  # Base context length
    rope_scaling_factor = config["rope_scaling_factor"]        # Context extension factor
    rope_ntk_alpha = config["rope_ntk_alpha"]          # YaRN NTK alpha parameter
    rope_ntk_beta = config["rope_ntk_beta"]            # YaRN NTK beta parameter

    # Calculate tokenizer metadata for the binary format
    # Text tokens come first, then special tokens
    tokens_size = 0
    num_text_tokens = 0

    # Count all text (non-special) tokens and calculate storage size
    # Each token is stored as: uint16_t length + byte array
    for t in range(o200k_gptoss.n_vocab):
        if not harmony_encoding.is_special_token(t):
            token_bytes = o200k_gptoss.decode_single_token_bytes(t)
            assert len(token_bytes) > 0
            tokens_size += len(token_bytes) + 2  # 2 bytes for length prefix
            num_text_tokens += 1

    # Total vocabulary includes all text tokens plus special tokens
    # 200013 is the last special token ID, +1 for inclusive count
    num_included_tokens = 200013 + 1
    print(f"Tokenizer: {num_included_tokens} tokens")

    # Load SafeTensors files from the checkpoint directory
    # Large models may be sharded across multiple .safetensors files
    safetensor_files = [
        os.path.join(options.src, fname)
        for fname in os.listdir(options.src)
        if fname.endswith(".safetensors")
    ]

    # Build an index mapping tensor names to their source files
    # This allows efficient loading of individual tensors from sharded checkpoints
    tensor_name_to_file = {}
    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, framework="pt", device="cpu") as src:
            for key in src.keys():
                tensor_name_to_file[key] = safetensor_file

    def get_tensor(name):
        """Helper function to load a tensor by name from the correct file."""
        with safe_open(tensor_name_to_file[name], framework="pt", device="cpu") as src:
            return src.get_tensor(name)

    # Open output file and begin writing the Metal model format
    with open(options.dst, "wb") as dst:
        # Write file magic header (identifies format version)
        write_file_header(dst)

        # Calculate YaRN RoPE parameters for context length extension
        # YaRN (Yet another RoPE extensioN method) adjusts RoPE frequencies
        # based on wavelength ranges to better handle extended context
        # These formulas compute the frequency band boundaries
        yarn_low = (
            head_dim / 2
            * math.log(initial_context_length / (rope_ntk_beta * 2 * math.pi))
            / math.log(rope_theta)
        )
        yarn_high = (
            head_dim / 2
            * math.log(initial_context_length / (rope_ntk_alpha * 2 * math.pi))
            / math.log(rope_theta)
        )

        # Write model architecture header with all hyperparameters
        write_model_header(dst,
                            # Extended context length (scaled from initial)
                            context_length=int(initial_context_length * rope_scaling_factor),
                            num_blocks=num_blocks,
                            num_experts=num_experts,
                            num_active_experts=num_active_experts,
                            embedding_dim=embedding_dim,
                            mlp_dim=mlp_dim,
                            swiglu_limit=swiglu_limit,
                            head_dim=head_dim,
                            num_heads=num_q_heads,
                            num_kv_heads=num_kv_heads,
                            attention_window=attention_window,
                            rope_theta=rope_theta,
                            # RoPE interpolation scale for context extension
                            interpolation_scale=1.0 / rope_scaling_factor,
                            # YaRN parameters derived from frequency band analysis
                            yarn_offset=-yarn_low / (yarn_high - yarn_low),
                            yarn_scale=1.0 / (yarn_high - yarn_low),
                            yarn_multiplier=0.1 * math.log(rope_scaling_factor) + 1.0,
                            rmsnorm_epsilon=1.0e-5)

        # Write tokenizer header with metadata
        write_tokenizer_header(dst,
                                num_special_tokens=num_included_tokens - num_text_tokens,
                                num_text_tokens=num_text_tokens,
                                regex_size=len(o200k_gptoss._pat_str.encode("ascii")) + 1,
                                tokens_size=tokens_size)

        ### Write Tokenizer Data ###

        # Write special token UUIDs (enables fast special token identification)
        # Special tokens are stored as 16-byte UUIDs at the beginning
        for token_idx in range(num_text_tokens, num_included_tokens):
            token = o200k_gptoss.decode_single_token_bytes(token_idx).decode('ascii')
            if token in INCLUDE_SPECIAL_TOKENS:
                # Write UUID for recognized special tokens
                dst.write(SPECIAL_TOKEN_UUID[token])
            else:
                # Write zeros for unused/reserved token slots
                dst.write(bytes(16))

        # Write BPE regex pattern (defines token boundary rules)
        # Null-terminated ASCII string
        dst.write(o200k_gptoss._pat_str.encode("ascii"))
        dst.write(struct.pack('B', 0))  # Null terminator

        # Write all text token byte sequences
        # Each token: uint16_t length + raw bytes
        tokenizer_bytes_written = 0
        for t in range(num_text_tokens):
            token_bytes = o200k_gptoss.decode_single_token_bytes(t)
            assert len(token_bytes) > 0
            dst.write(struct.pack('<H', len(token_bytes)))  # Length prefix
            dst.write(token_bytes)                          # Token bytes
            tokenizer_bytes_written += len(token_bytes) + 2

        # Verify we wrote exactly the expected amount of tokenizer data
        assert(tokenizer_bytes_written == tokens_size), (tokenizer_bytes_written, tokens_size)

        # Align to 16KB boundary for efficient GPU loading
        write_padding(dst)

        ### Write Model Weights ###

        # Write token embedding matrix (vocab_size x embedding_dim)
        embedding_weight = get_tensor("embedding.weight")
        # Truncate to only include used tokens (text + special)
        embedding_weight = embedding_weight[:num_included_tokens, :]
        write_embedding_weight(dst, embedding_weight)

        # Write transformer blocks (layers) with progress bar
        for n in tqdm(range(num_blocks), desc="Writing transformer blocks"):
            # Write attention layer normalization gain
            write_rmsnorm_gain(dst, get_tensor(f"block.{n}.attn.norm.scale"))

            # Load and transform QKV projection weights and biases
            # QKV is stored as a fused projection [Q;K;V] for efficiency
            attn_qkv_weight = get_tensor(f"block.{n}.attn.qkv.weight")
            attn_qkv_bias = get_tensor(f"block.{n}.attn.qkv.bias")

            # Apply RoPE-aware weight transformation and fused scaling
            for qkv in (attn_qkv_weight, attn_qkv_bias):
                # Split QK from V (QK needs RoPE-aware dimension reordering)
                qk = qkv[:head_dim * (num_q_heads + num_kv_heads), ...].contiguous()
                v = qkv[head_dim * (num_q_heads + num_kv_heads):, ...].contiguous()

                # Reorder QK dimensions for RoPE: interleave real/imaginary components
                # Shape: [num_heads, head_dim, *] -> [num_heads, 2, head_dim//2, *]
                qk = qk.view(num_q_heads + num_kv_heads, 2, head_dim // 2, -1).transpose(1, 2).reshape(num_q_heads + num_kv_heads, head_dim, -1)

                # Split into separate Q and K projections
                q = qk[:num_q_heads, ...]
                k = qk[num_q_heads:, ...]

                # Fuse attention scaling (1/sqrt(head_dim)) into Q and K weights
                # This optimization eliminates a runtime multiplication
                # 1/sqrt(64) = 0.125 = 0.5 * 0.25 (split between Q and K)
                assert head_dim == 64
                q *= 0.5    # Apply sqrt(0.5) scaling to Q
                k *= 0.25   # Apply sqrt(0.25) scaling to K

                # Reshape V projection to match head structure
                v = v.view(num_kv_heads, head_dim, -1)

                # Concatenate back into fused QKV format
                qkv.copy_(torch.cat((q, k, v), dim=0).reshape(*qkv.shape))

            # Write transformed QKV projection
            write_linear_weight(dst, attn_qkv_weight, attn_qkv_bias)

            # Write attention sink tokens (learned embeddings for KV cache stability)
            write_attn_sink(dst, get_tensor(f"block.{n}.attn.sinks"))

            # Write attention output projection
            write_linear_weight(dst, get_tensor(f"block.{n}.attn.out.weight"), get_tensor(f"block.{n}.attn.out.bias"))

            # Write MLP layer normalization gain
            write_rmsnorm_gain(dst, get_tensor(f"block.{n}.mlp.norm.scale"))

            # Write MoE gating network (routes tokens to experts)
            write_linear_weight(dst, get_tensor(f"block.{n}.mlp.gate.weight"), get_tensor(f"block.{n}.mlp.gate.bias"))

        # Write final layer normalization gain
        write_rmsnorm_gain(dst, get_tensor("norm.scale"))

        # Write output unembedding matrix (projects to vocabulary logits)
        unembedding_weight = get_tensor("unembedding.weight")
        # Truncate to only include used tokens
        unembedding_weight = unembedding_weight[:num_included_tokens, :]
        write_linear_weight(dst, unembedding_weight)

        ### Write MoE Expert Weights (Quantized) ###
        # MoE weights are written separately in MXFP4 quantized format
        # MXFP4: Mixed-precision FP4 with shared block exponents (8x compression)

        for n in tqdm(range(num_blocks), desc="Writing MoE expert weights"):
            # Load quantized MLP1 weights (input projection for SwiGLU)
            # MXFP4 format: blocks contain 4-bit mantissas, scales contain shared exponents
            mlp1_blocks = get_tensor(f"block.{n}.mlp.mlp1_weight.blocks")
            mlp1_scales = get_tensor(f"block.{n}.mlp.mlp1_weight.scales")
            # Verify scales can be offset into uint8 range
            assert mlp1_scales.min().item() < 254 - UE8_OFFSET
            mlp1_bias = get_tensor(f"block.{n}.mlp.mlp1_bias")

            # Load quantized MLP2 weights (output projection for SwiGLU)
            mlp2_blocks = get_tensor(f"block.{n}.mlp.mlp2_weight.blocks")
            mlp2_scales = get_tensor(f"block.{n}.mlp.mlp2_weight.scales")
            assert mlp2_scales.min().item() < 254 - UE8_OFFSET
            mlp2_bias = get_tensor(f"block.{n}.mlp.mlp2_bias")

            # Write MoE weights grouped by expert for efficient Metal loading
            # This layout allows loading one expert at a time during sparse MoE inference
            write_padding(dst)

            for e in range(num_experts):
                # Write MLP1 (input projection) for expert e
                write_padding(dst, alignment_multiple=16)
                # 4-bit mantissa blocks
                dst.write(mlp1_blocks[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)
                # Shared exponent scales (offset to make positive for uint8 storage)
                dst.write((mlp1_scales + UE8_OFFSET)[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)
                # Bias term (full precision)
                dst.write(mlp1_bias[e, ...].view(torch.uint8).numpy().tobytes())

                # Write MLP2 (output projection) for expert e
                write_padding(dst, alignment_multiple=16)
                # 4-bit mantissa blocks
                dst.write(mlp2_blocks[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)
                # Shared exponent scales
                dst.write((mlp2_scales + UE8_OFFSET)[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)
                # Bias term
                dst.write(mlp2_bias[e, ...].view(torch.uint8).numpy().tobytes())

if __name__ == "__main__":
    main(sys.argv[1:])
