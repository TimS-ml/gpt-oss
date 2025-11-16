"""
Distributed Training Utilities

This module provides utility functions for setting up and managing distributed
PyTorch training and inference. It includes functions for initializing distributed
processes, suppressing output on non-primary ranks, and warming up communication
backends.
"""

import os
import torch
import torch.distributed as dist


def suppress_output(rank):
    """
    Suppress print output on all ranks except rank 0 to reduce noise during
    distributed training/inference.

    This function replaces the built-in print function with a custom version that:
    - Only prints on rank 0 by default
    - Can force printing on any rank by passing force=True
    - Prefixes forced prints with the rank number for debugging

    Args:
        rank (int): The rank of the current process in the distributed setup.
                   Typically 0 for the primary process.

    Example:
        >>> suppress_output(rank=1)
        >>> print("This won't be printed on rank 1")
        >>> print("This will be printed", force=True)  # Prints "rank #1: This will be printed"
    """
    import builtins as __builtin__

    # Save the original print function before overriding it
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        """
        Custom print function that respects rank-based output suppression.

        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments, including optional 'force' parameter
                     force (bool): If True, print on any rank with rank prefix
        """
        # Extract the force parameter, defaulting to False
        force = kwargs.pop('force', False)

        if force:
            # Force printing on any rank, but prefix with rank number
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            # Normal printing only on the primary rank (rank 0)
            builtin_print(*args, **kwargs)

    # Replace the built-in print function with our custom version
    __builtin__.print = print


def init_distributed() -> torch.device:
    """
    Initialize PyTorch distributed inference/training environment.

    This function sets up distributed processing using NCCL backend for multi-GPU
    training or inference. It handles both single-GPU and multi-GPU scenarios.

    The function performs the following steps:
    1. Reads WORLD_SIZE and RANK from environment variables (set by torchrun or similar)
    2. Initializes the process group if running on multiple GPUs
    3. Sets the current CUDA device based on the rank
    4. Warms up NCCL communication to avoid first-call latency
    5. Suppresses print output on non-primary ranks

    Environment Variables:
        WORLD_SIZE (int): Total number of processes (GPUs) in the distributed setup.
                         Defaults to 1 for single-GPU inference.
        RANK (int): The rank of this process (0 to WORLD_SIZE-1).
                   Defaults to 0 for single-GPU inference.

    Returns:
        torch.device: The CUDA device assigned to this process (e.g., cuda:0, cuda:1, etc.)

    Example:
        >>> device = init_distributed()
        >>> # On a 4-GPU setup, this will return cuda:0, cuda:1, cuda:2, or cuda:3
        >>> # depending on which process is running

    Note:
        - Uses NCCL backend which is optimized for NVIDIA GPUs
        - Assumes one process per GPU (rank == GPU device index)
        - The warmup step performs a dummy all_reduce to initialize NCCL communication
    """
    # Read distributed configuration from environment variables
    # These are typically set by launchers like torchrun or torch.distributed.launch
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # Total number of processes
    rank = int(os.environ.get("RANK", 0))  # Current process rank (0-indexed)

    # Initialize the distributed process group if using multiple GPUs
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",  # NCCL is the recommended backend for CUDA-based distributed training
            init_method="env://",  # Read configuration from environment variables
            world_size=world_size,
            rank=rank
        )

    # Set the CUDA device for this process (one GPU per process)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Warm up NCCL to avoid first-time latency during actual model execution
    # This performs a dummy all_reduce operation to initialize communication channels
    if world_size > 1:
        x = torch.ones(1, device=device)  # Create a small dummy tensor
        dist.all_reduce(x)  # Synchronize across all processes
        torch.cuda.synchronize(device)  # Wait for GPU operations to complete

    # Suppress output on all ranks except rank 0 to reduce clutter
    suppress_output(rank)

    return device
