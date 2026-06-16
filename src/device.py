"""
device.py
=========
Shared GPU/CPU detection utility for the pdm_pronostia project.
"""

from __future__ import annotations

import torch

_device_cache: str | None = None


def get_device(verbose: bool = True) -> str:
    """Detect the best available compute device and return its name.

    Priority: CUDA GPU  ->  Apple MPS  ->  CPU.

    The result is cached so the banner is printed only once per process.

    Parameters
    ----------
    verbose:
        Print the detected device on the first call.

    Returns
    -------
    str
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        _device_cache = "cuda"
        if verbose:
            print(f"[Device] CUDA GPU detected: {name}  ->  using cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _device_cache = "mps"
        if verbose:
            print("[Device] Apple MPS detected  ->  using mps")
    else:
        _device_cache = "cpu"
        if verbose:
            print("[Device] No GPU detected  ->  falling back to cpu")

    return _device_cache
