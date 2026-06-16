"""
rul_predictor.py
================
Multi-scale Conv-SA RUL predictor with MC Dropout uncertainty quantification.

Architecture
------------
Input : (batch, seq_len, 32)

1. Multi-scale encoder — 3 parallel Conv1d branches (k=3/5/3), each 64 ch,
   concatenated → Linear(192→128)
2. Sinusoidal positional encoding
3. 2× TransformerEncoderLayer(d_model=128, nhead=4, dim_ff=256, dropout=0.1)
4. Mean pool → MC Dropout(0.5)
5. Head: Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid [0,1]
   Multiply by MAX_RUL=125 to recover original scale.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.device import get_device
except ImportError:
    from device import get_device

logger = logging.getLogger(__name__)

MAX_RUL: float = 125.0
N_FEATURES: int = 32


# ---------------------------------------------------------------------------
# Sliding-window helpers
# ---------------------------------------------------------------------------

def create_sliding_windows(
    features: np.ndarray,
    rul_labels: np.ndarray,
    window_size: int = 32,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    N = features.shape[0]
    if N < window_size:
        raise ValueError(f"Bearing has only {N} timesteps but window_size={window_size}.")
    windows, ruls = [], []
    for start in range(0, N - window_size + 1, stride):
        windows.append(features[start : start + window_size])
        ruls.append(rul_labels[start + window_size - 1])
    return np.array(windows, dtype=np.float32), np.array(ruls, dtype=np.float32)


def build_windows_from_dict(
    feature_dict: Dict[str, Dict[str, np.ndarray]],
    bearing_ids: Optional[List[str]] = None,
    window_size: int = 32,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    keys = bearing_ids or list(feature_dict.keys())
    X_parts, y_parts = [], []
    for bid in keys:
        if bid not in feature_dict:
            logger.warning("Bearing %s not in feature_dict; skipped.", bid)
            continue
        W, R = create_sliding_windows(
            feature_dict[bid]["features"], feature_dict[bid]["rul"],
            window_size=window_size, stride=stride,
        )
        X_parts.append(W); y_parts.append(R)
    if not X_parts:
        raise ValueError("No bearings found to build windows from.")
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Multi-scale Conv-SA model
# ---------------------------------------------------------------------------

class ConvSARULPredictor(nn.Module):
    """Three parallel Conv1d branches + Transformer encoder + MC Dropout head."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        window_size: int = 32,
        dropout_mc: float = 0.5,
    ) -> None:
        super().__init__()

        # Parallel multi-scale branches
        self.branch_k3a = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.branch_k3b = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.proj = nn.Linear(192, 128)   # 3×64 → d_model

        self.pos_enc = _PositionalEncoding(d_model=128, max_len=max(512, window_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.mc_dropout = nn.Dropout(p=dropout_mc)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            # No bounding activation — linear output avoids vanishing gradients
            # near saturation.  Output is clamped to [0,1] at inference time.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        h = x.permute(0, 2, 1)                               # (B, F, L)
        h = torch.cat([self.branch_k3a(h),
                       self.branch_k5(h),
                       self.branch_k3b(h)], dim=1)           # (B, 192, L)
        h = h.permute(0, 2, 1)                               # (B, L, 192)
        h = self.proj(h)                                      # (B, L, 128)
        h = self.pos_enc(h)
        h = self.transformer(h)                               # (B, L, 128)
        h = h.mean(dim=1)                                     # (B, 128)
        h = self.mc_dropout(h)
        return self.head(h).squeeze(-1)                       # (B,)


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def mc_dropout_inference(
    model: ConvSARULPredictor,
    x: np.ndarray | torch.Tensor,
    n_samples: int = 50,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo Dropout inference.

    n_samples == 1  → deterministic (model.eval(), Dropout off).
    n_samples > 1   → stochastic (model.train(), Dropout active).

    Returns (mean_rul, sigma2, samples) in original RUL scale (×125).
    sigma2 = 0 when n_samples == 1.
    """
    dev_str = device if device is not None else get_device(verbose=False)
    dev = torch.device(dev_str)

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x = x.to(dev)
    model.to(dev)
    B = x.shape[0]

    with torch.no_grad():
        if n_samples == 1:
            # Deterministic: eval mode, no dropout
            model.eval()
            preds   = torch.clamp(model(x), 0.0, 1.0) * MAX_RUL
            samples = preds.cpu().numpy().reshape(1, B)
        else:
            # MC Dropout: freeze BN running stats (eval); enable ONLY the
            # head Dropout(0.3) for sampling.  Activating mc_dropout(0.5) at
            # the bottleneck produces high variance that biases the MC mean
            # away from the deterministic prediction, breaking calibration.
            model.eval()
            model.head[2].train()   # head.2 = Dropout(0.3) before final linear
            if B == 1:
                # Fast path: tile → one batched pass
                x_tiled = x.expand(n_samples, -1, -1).contiguous()
                preds   = torch.clamp(model(x_tiled), 0.0, 1.0) * MAX_RUL
                samples = preds.cpu().numpy().reshape(n_samples, 1)
            else:
                all_preds = []
                for _ in range(n_samples):
                    all_preds.append(
                        torch.clamp(model(x), 0.0, 1.0).cpu().numpy() * MAX_RUL
                    )
                samples = np.stack(all_preds, axis=0)

    mean_rul = samples.mean(axis=0)
    sigma2   = samples.var(axis=0)

    if mean_rul.shape == (1,):
        mean_rul = mean_rul[0]
        sigma2   = sigma2[0]
        samples  = samples.squeeze(1)

    return mean_rul, sigma2, samples


# ---------------------------------------------------------------------------
# Cumulative Failure Probability
# ---------------------------------------------------------------------------

def compute_cfp(rul_samples: np.ndarray, tau: float = 30.0) -> float:
    """Fraction of MC samples predicting RUL ≤ tau."""
    return float(np.mean(rul_samples <= tau))


# ---------------------------------------------------------------------------
# Health Index from RUL
# ---------------------------------------------------------------------------

def compute_hi_from_rul(mean_rul_sequence: np.ndarray) -> np.ndarray:
    """HI = mean_rul / 125, smoothed with centred rolling mean (window=5).

    Parameters
    ----------
    mean_rul_sequence : (N,) — MC-mean RUL values in [0, 125].

    Returns
    -------
    np.ndarray, shape (N,), float32 — HI in [0, 1].
    """
    hi      = np.clip(mean_rul_sequence / MAX_RUL, 0.0, 1.0)
    kernel  = np.ones(5, dtype=np.float64) / 5
    padded  = np.pad(hi.astype(np.float64), (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_model(model: ConvSARULPredictor, path: str | Path) -> None:
    torch.save(model.state_dict(), path)
    logger.info("Saved model to %s", path)


def load_model(
    path: str | Path,
    n_features: int = N_FEATURES,
    window_size: int = 32,
    dropout_mc: float = 0.5,
    device: Optional[str] = None,
) -> ConvSARULPredictor:
    """Load ConvSARULPredictor from checkpoint. Returns model in eval mode."""
    dev_str = device if device is not None else get_device(verbose=False)
    dev     = torch.device(dev_str)
    model   = ConvSARULPredictor(n_features=n_features, window_size=window_size,
                                 dropout_mc=dropout_mc)
    ckpt       = torch.load(path, map_location=dev, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.to(dev).eval()
    logger.info("Loaded model from %s (device=%s)", path, dev_str)
    return model


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(0)
    SEQ, FEAT = 32, 32

    model  = ConvSARULPredictor(n_features=FEAT, window_size=SEQ)
    x_fake = torch.randn(4, SEQ, FEAT)
    with torch.no_grad():
        out = model(x_fake)
    print(f"forward: {tuple(x_fake.shape)} -> {tuple(out.shape)}  "
          f"range=[{out.min():.3f}, {out.max():.3f}]")

    feat_seq = rng.standard_normal((200, FEAT)).astype(np.float32)
    rul_seq  = np.linspace(125, 0, 200).astype(np.float32)
    windows, ruls = create_sliding_windows(feat_seq, rul_seq, window_size=SEQ)
    print(f"create_sliding_windows: {windows.shape}  ruls={ruls.shape}")

    mean_r, var_r, samples = mc_dropout_inference(model, windows[:10], n_samples=50)
    print(f"MC inference (batch=10): mean={mean_r.round(1)}")
    mean_d, var_d, _ = mc_dropout_inference(model, windows[0], n_samples=1)
    print(f"Deterministic (n=1): {mean_d:.2f}  var={var_d:.2f}")

    _, _, s = mc_dropout_inference(model, windows[0], n_samples=50)
    print(f"CFP(tau=30): {compute_cfp(s, tau=30):.3f}")

    hi = compute_hi_from_rul(np.linspace(125, 0, 100))
    print(f"compute_hi_from_rul: {hi.shape}  range=[{hi.min():.3f}, {hi.max():.3f}]")
    print("rul_predictor: OK")
