"""Device selection with automatic CPU fallback."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device.

    Args:
        prefer_cuda: If True, use CUDA (or MPS) when available.

    Returns:
        torch.device instance.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_cuda and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    """Return a dict of device-related metadata for reproducibility logs."""
    device = get_device()
    info: dict = {"device": str(device)}
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / (1024**3), 2)
    else:
        info["cuda_version"] = "N/A"
        info["gpu_name"] = "N/A"
    return info
