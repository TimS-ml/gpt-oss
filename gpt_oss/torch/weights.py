"""
Weight Loading and MXFP4 Quantization Module

This module provides functionality for loading model checkpoints and handling
MXFP4 (Microscaling FP4) quantization format. MXFP4 is a block-based quantization
format that uses 4-bit floating point numbers with shared exponents per block,
enabling efficient storage and computation for large models.

Key concepts:
- MXFP4: A quantization format where groups of values share a common scale (exponent)
- Safetensors: A safe and efficient format for storing tensors
- Block-based quantization: Reduces memory footprint while maintaining model quality
"""

import math
import os

import torch
from safetensors import safe_open


# MXFP4 Format Constants
# Each block contains 32 FP4 values packed into 16 bytes (2 values per byte)
BYTES_PER_BLOCK = 16

# Lookup table for FP4 (4-bit floating point) mantissa values
# FP4 has 16 possible values (4 bits): 8 positive and 8 negative
# These are the base values before scaling by the shared exponent
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,  # Positive values (0-7)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # Negative values (8-15)
]

# Parameter Name Mapping
# Maps the parameter names used in this implementation to the actual checkpoint names.
# MoE (Mixture of Experts) weights are stored in MXFP4 format with separate blocks and scales.
# Biases are stored as regular tensors.
# The model has 36 transformer blocks (layers).
PARAM_NAME_MAP = {
    # MLP1 biases: stored as regular tensors (not quantized)
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    # MLP1 weights: stored in MXFP4 format (tuple of blocks and scales)
    f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales") for n in range(36)
} | {
    # MLP2 biases: stored as regular tensors (not quantized)
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    # MLP2 weights: stored in MXFP4 format (tuple of blocks and scales)
    f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales") for n in range(36)
}


class Checkpoint:
    """
    Checkpoint loader for model weights stored in SafeTensors format.

    This class handles loading model parameters from checkpoint files, supporting
    both regular tensors and MXFP4-quantized tensors. It automatically discovers
    all safetensor files in the checkpoint directory and builds an index for
    efficient parameter retrieval.

    Attributes:
        device_str (str): Device string for loading tensors (e.g., "cuda:0", "cpu")
        tensor_name_to_file (dict): Mapping from tensor names to their file paths
    """

    def __init__(self, path: str, device: torch.device):
        """
        Initialize the checkpoint loader.

        Args:
            path (str): Directory path containing .safetensors checkpoint files
            device (torch.device): Target device for loading tensors

        Note:
            This constructor scans all .safetensors files in the directory and
            builds an index of available tensors for fast lookup.
        """
        # Convert device object to string format expected by safetensors
        device_str = (
            device.type
            if device.index is None
            else device.type + ":" + str(device.index)
        )
        self.device_str = device_str

        # Discover all safetensor files in the checkpoint directory
        # Large models are often split across multiple files for efficient loading
        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]

        # Build a mapping from tensor name to file path
        # This allows O(1) lookup when loading specific parameters
        tensor_name_to_file = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device=device_str) as f:
                for key in f.keys():
                    tensor_name_to_file[key] = safetensor_file

        self.tensor_name_to_file = tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        """
        Retrieve a parameter tensor by name, handling both regular and MXFP4 formats.

        This method automatically detects the tensor format based on the parameter
        name mapping and returns the appropriately decoded tensor.

        Args:
            name (str): Parameter name (e.g., "block.0.mlp.mlp1_weight")

        Returns:
            torch.Tensor: The loaded and decoded tensor, ready for use in the model

        Note:
            - MXFP4 tensors are automatically dequantized to bfloat16
            - Regular tensors are loaded as-is from the checkpoint
        """
        # Use pattern matching to determine tensor format
        match PARAM_NAME_MAP.get(name, name):
            case (blocks_name, scales_name):
                # MoE weights are stored in block-based MXFP4 format
                # This saves memory by using 4-bit quantization with shared scales
                return self._get_mxfp4_tensor(blocks_name, scales_name, dtype=torch.bfloat16)
            case tensor_name:
                # MoE biases and other weights are stored as regular tensors
                return self._get_tensor(tensor_name)

    def _get_tensor(self, name: str) -> torch.Tensor:
        """
        Load a regular (non-quantized) tensor from the checkpoint.

        Args:
            name (str): Exact tensor name as stored in the checkpoint file

        Returns:
            torch.Tensor: The loaded tensor

        Raises:
            AssertionError: If the tensor name is not found in any checkpoint file
        """
        assert name in self.tensor_name_to_file, f"Tensor {name} not found in checkpoint."
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)

    def _get_mxfp4_tensor(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 16384 * 512,
    ) -> torch.Tensor:
        """
        Dequantize an MXFP4 tensor to the target dtype.

        MXFP4 (Microscaling FP4) stores tensors in a compressed format:
        - Each byte contains two 4-bit FP values (packed as nibbles)
        - Groups of values share a common scale (exponent)
        - This achieves ~8x compression compared to bfloat16

        The dequantization process:
        1. Load packed blocks (4-bit values) and scales (exponents)
        2. Unpack nibbles into separate indices
        3. Look up FP4 mantissa values
        4. Scale by the shared exponent using ldexp (value * 2^exponent)

        Args:
            blocks_name (str): Name of the tensor containing packed 4-bit values
            scales_name (str): Name of the tensor containing shared exponents
            dtype (torch.dtype): Target dtype for dequantized output (default: bfloat16)
            rows_per_chunk (int): Process this many rows at a time to manage memory
                                 (default: ~8M values per chunk)

        Returns:
            torch.Tensor: Dequantized tensor in the target dtype

        Note:
            - Processing in chunks prevents out-of-memory errors on large tensors
            - The scales are stored with a bias of 127 which is subtracted during loading
            - Each block byte is unpacked into 2 values (low and high nibbles)
        """
        # Validate that both components exist in the checkpoint
        assert blocks_name in self.tensor_name_to_file, (
            f"Blocks tensor {blocks_name} not found in checkpoint."
        )
        assert scales_name in self.tensor_name_to_file, (
            f"Scales tensor {scales_name} not found in checkpoint."
        )

        # Load the quantized components
        blocks = self._get_tensor(blocks_name)  # Packed 4-bit values (uint8)
        scales = self._get_tensor(scales_name).to(torch.int32) - 127  # Exponents (remove bias)

        # Verify shape compatibility: scales should match all but the last dimension of blocks
        assert blocks.shape[:-1] == scales.shape, (
            f"{blocks.shape=} does not match {scales.shape=}"
        )

        # Create lookup table for FP4 mantissa values in target dtype
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        # Reshape for efficient processing
        # prefix_shape: leading dimensions (e.g., num_experts)
        # G: number of groups (scale groups)
        # B: number of bytes per group
        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G  # Total number of scale groups to process

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        # Allocate output tensor (each byte unpacks to 2 values)
        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        # Process in chunks to manage memory usage
        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]  # Current chunk of packed blocks
            exp = scales[r0:r1]  # Current chunk of exponents

            # Unpack nibbles: each byte contains 2 FP4 values
            # Low nibble (bits 0-3): value & 0x0F
            # High nibble (bits 4-7): value >> 4
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            # Look up mantissa values and interleave them
            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]  # Even positions: low nibbles
            sub[:, 1::2] = lut[idx_hi]  # Odd positions: high nibbles

            # Scale by the shared exponent: value * 2^exponent
            # ldexp is more numerically stable than manual multiplication
            torch.ldexp(sub, exp, out=sub)

            # Free temporary tensors to reduce memory pressure
            del idx_lo, idx_hi, blk, exp

        # Reshape back to original dimensions (with unpacked last dimension)
        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    def _get_mxfp4_tensor_copy(self, blocks_name: str, scales_name: str, dtype: torch.dtype = torch.bfloat16):
        """
        Alternative MXFP4 dequantization implementation (memory-intensive).

        This is a simpler but more memory-intensive version of _get_mxfp4_tensor.
        It processes the entire tensor at once without chunking, which is easier
        to understand but may cause out-of-memory errors on large tensors.

        Args:
            blocks_name (str): Name of the tensor containing packed 4-bit values
            scales_name (str): Name of the tensor containing shared exponents
            dtype (torch.dtype): Target dtype for dequantized output (default: bfloat16)

        Returns:
            torch.Tensor: Dequantized tensor in the target dtype

        Note:
            This method is kept for reference but _get_mxfp4_tensor should be
            preferred for production use as it handles memory more efficiently.
        """
        # Load the packed blocks
        loaded_blocks = self._get_tensor(blocks_name)

        # Unpack nibbles into separate tensors
        # Each byte contains 2 FP4 values packed as low/high nibbles
        loaded_blocks_lo = loaded_blocks & 0x0F  # Extract low nibble (bits 0-3)
        loaded_blocks_hi = loaded_blocks >> 4     # Extract high nibble (bits 4-7)

        # Stack and interleave the nibbles for SwiGLU activation function
        # SwiGLU expects pairs of values arranged as (glu_0, linear_0, glu_1, linear_1, ...)
        loaded_blocks = torch.stack((loaded_blocks_lo, loaded_blocks_hi), dim=-1)
        loaded_blocks = loaded_blocks.view(*loaded_blocks.shape[:-2], loaded_blocks.shape[-2] * 2)

        # Load and prepare the exponent scales
        loaded_scales = self._get_tensor(scales_name)
        # Remove the bias of 127 that was added during quantization
        loaded_scales = loaded_scales.int() - 127

        # Dequantize: lookup mantissa values and scale by exponents
        fp4_values = torch.tensor(FP4_VALUES, dtype=dtype, device=self.device_str)
        # Index into FP4 lookup table, then scale by 2^exponent
        loaded_tensor = torch.ldexp(fp4_values[loaded_blocks.int()], loaded_scales.unsqueeze(-1))

        # Flatten the last two dimensions to get final shape
        loaded_tensor = loaded_tensor.view(*loaded_tensor.shape[:-2], -1)
        return loaded_tensor
