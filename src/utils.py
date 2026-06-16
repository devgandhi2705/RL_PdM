"""Shared utilities: seed control, config loading, running stats, array verification."""

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        # resolve relative to project root (two levels up from this file)
        path = Path(__file__).resolve().parent.parent / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class AverageMeter:
    """Numerically stable Welford running mean and variance."""

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def add(self, value: float) -> None:
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._M2 += delta * delta2

    def result(self) -> Tuple[float, float]:
        """Returns (mean, std). std is 0.0 when n < 2."""
        if self._n < 2:
            return self._mean, 0.0
        return self._mean, float(np.sqrt(self._M2 / (self._n - 1)))

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def n(self) -> int:
        return self._n


def verify_array(
    arr: np.ndarray,
    name: str,
    expected_shape: Optional[tuple] = None,
    expected_range: Optional[Tuple[float, float]] = None,
    allow_nan: bool = False,
) -> bool:
    """Prints PASS/FAIL for each check. Returns True only when all checks pass."""
    ok = True
    prefix = f"[verify] {name}"

    if expected_shape is not None:
        if arr.shape == expected_shape:
            print(f"{prefix} shape {arr.shape}: PASS")
        else:
            print(f"{prefix} shape FAIL — expected {expected_shape}, got {arr.shape}")
            ok = False

    if not allow_nan and np.isnan(arr).any():
        print(f"{prefix} NaN check: FAIL — {int(np.isnan(arr).sum())} NaNs found")
        ok = False
    elif not allow_nan:
        print(f"{prefix} NaN check: PASS")

    if expected_range is not None:
        lo, hi = expected_range
        out_of = np.sum((arr < lo) | (arr > hi))
        if out_of == 0:
            print(f"{prefix} range [{lo}, {hi}]: PASS")
        else:
            print(
                f"{prefix} range FAIL — {out_of}/{arr.size} values outside [{lo}, {hi}]"
                f"  (min={arr.min():.4f}, max={arr.max():.4f})"
            )
            ok = False

    return ok
