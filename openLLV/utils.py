"""General utility functions for openLLV."""

import torch
import datetime
import sys


def device_display_name(device) -> str:
    """Format a torch device as a human-readable name.

    Args:
        device: Torch device object with a ``type`` attribute.

    Returns:
        Display name for_teach CUDA, CPU, or other device types.
    """
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(index)
        return f"{device_name} ({device})"
    if device.type == "cpu":
        return "CPU"

    return f"{device.type.upper()} ({device})"


def log_info_env(device=None):
    """Print Python, torch, and device environment information.

    When ``device`` is ``None``, all available CUDA devices are printed if CUDA
    is available; otherwise CPU-only information is printed. When ``device`` is
    provided, only the requested device is reported.

    Args:
        device: Optional device specifier accepted by ``torch.device``.

    Returns:
        None.
    """
    current_date = datetime.datetime.now().isoformat(timespec="seconds")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    torch_version = torch.__version__

    if device is None:
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            num_gpus = torch.cuda.device_count()

            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = (
                    torch.cuda.get_device_properties(i).total_memory / 1024**2
                )

                print(
                    f"{current_date} Python-{python_version} "
                    f"torch-{torch_version}+cu{cuda_version} "
                    f"CUDA:{i} ({gpu_name}, {int(gpu_memory)}MiB)"
                )
        else:
            print(f"{current_date} Python-{python_version} torch-{torch_version} CPU. (No CUDA available)")

        return

    device = torch.device(device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            print(f"{current_date} Python-{python_version} torch-{torch_version} (CUDA requested but not available)")

            return

        index = device.index if device.index is not None else torch.cuda.current_device()
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(index)
        gpu_memory = torch.cuda.get_device_properties(index).total_memory / 1024**2

        print(f"{current_date} Python-{python_version} "
            f"torch-{torch_version}+cu{cuda_version} "
            f"CUDA:{index} ({gpu_name}, {int(gpu_memory)}MiB)")

    elif device.type == "cpu":
        print(f"{current_date} Python-{python_version} torch-{torch_version} CPU")

    else:
        print(f"{current_date} Python-{python_version} torch-{torch_version} device:{device}")
