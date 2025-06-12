import psutil
import torch
import time
from typing import Any


def log_memory_usage(logger: Any, device: str) -> None:
    """
    Logs the memory usage of the CPU and, if applicable, the GPU.

    Args:
        logger: The logging object used to record memory usage.
        device: The device identifier as a string (e.g., "cuda:0" for GPU, "cpu" for CPU).
    """
    # Log CPU memory usage
    process = psutil.Process()
    cpu_memory_usage = process.memory_info().rss / (1024**3)  # in GB
    logger.scalar("profile/cpu_memory_usage_gb", cpu_memory_usage)

    # Log GPU memory usage
    if device.startswith("cuda"):
        gpu_memory_usage = torch.cuda.memory_allocated(device) / (1024**3)  # in GB
        logger.scalar("profile/gpu_memory_usage_gb", gpu_memory_usage)


def log_runtime(logger: Any, start_time: float) -> None:
    """
    Logs the runtime since a given start time in minutes.

    Args:
        logger: The logging object used to record the runtime.
        start_time: The start time in seconds since the epoch.
    """
    runtime_seconds = time.time() - start_time
    runtime_minutes = runtime_seconds / 60
    logger.scalar("profile/runtime_minutes", runtime_minutes)
