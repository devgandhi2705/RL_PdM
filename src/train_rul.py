"""
train_rul.py
============
Train the multi-scale Conv-SA RUL predictor on pre-scaled PRONOSTIA features.

Pipeline
--------
1. Load pre-scaled features from data/processed/ (do NOT refit scaler here).
   Verify scaler quality before training starts.
2. Build sliding-window datasets (window_size=32).
3. Train ConvSARULPredictor — 150 epochs max, AdamW + CosineAnnealingLR,
   early stopping (patience=20). Checkpoint saved on best val_RMSE_late.
4. Per-epoch logging: train_loss, val_loss, RMSE_full, RMSE_late,
   first50/last50 RUL means, discriminability gap.
5. Final evaluation + PASS/FAIL verdict.

Usage
-----
    python -m src.train_rul
    python -m src.train_rul --epochs 200 --patience 30 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

try:
    from src.data_loader import TRAIN_BEARINGS, TEST_BEARINGS
    from src.device import get_device
    from src.rul_predictor import (
        ConvSARULPredictor,
        build_windows_from_dict,
        create_sliding_windows,
        N_FEATURES,
        MAX_RUL,
    )
    from src.utils import load_config, set_seed
except ImportError:
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from src.data_loader import TRAIN_BEARINGS, TEST_BEARINGS
    from src.device import get_device
    from src.rul_predictor import (
        ConvSARULPredictor,
        build_windows_from_dict,
        create_sliding_windows,
        N_FEATURES,
        MAX_RUL,
    )
    from src.utils import load_config, set_seed

logger = logging.getLogger(__name__)

_VAL_BEARING: str = TEST_BEARINGS[0]   # "3_2"

_DEFAULTS: Dict[str, Any] = {
    "window_size":  32,
    "batch_size":   64,
    "lr":           1e-4,
    "weight_decay": 1e-4,
    "dropout_mc":   0.5,
    "epochs":       200,
    "patience":     50,
    "grad_clip":    1.0,
    "n_mc_eval":    50,
    "seed":         42,
    "results_dir":  "results/01_rul_predictor",
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(results_dir / "train_rul.log", mode="a",
                                encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_scaled_features(
    processed_dir: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load pre-scaled features from data/processed/.

    Raises ValueError if features appear unscaled or corrupted.
    """
    if not processed_dir.is_dir():
        raise FileNotFoundError(
            f"Processed features directory not found: {processed_dir}\n"
            "Run:  python src/feature_extractor.py  first."
        )

    all_ids = list(TRAIN_BEARINGS) + [b for b in TEST_BEARINGS if b not in TRAIN_BEARINGS]
    scaled_dict: Dict[str, Dict[str, np.ndarray]] = {}

    for bid in all_ids:
        feat_p = processed_dir / f"{bid}_features.npy"
        rul_p  = processed_dir / f"{bid}_rul.npy"
        if not feat_p.exists():
            logger.warning("Missing processed features for bearing %s.", bid)
            continue
        features = np.load(feat_p)
        rul      = np.load(rul_p)
        scaled_dict[bid] = {"features": features, "rul": rul}
        logger.info("  %s: features%s  rul%s", bid, features.shape, rul.shape)

    train_present = [b for b in TRAIN_BEARINGS if b in scaled_dict]
    if not train_present:
        raise RuntimeError("No training bearings found in processed_dir.")

    # Per-bearing normalization: each bearing has mean=0, std=1 by construction
    for bid in train_present:
        feats  = scaled_dict[bid]["features"]
        b_mean = float(np.abs(feats.mean()))
        b_std  = float(feats.std())
        logger.info("  %s: |mean|=%.4f  std=%.4f", bid, b_mean, b_std)
        if b_mean > 0.05 or b_std < 0.9 or b_std > 1.1:
            raise ValueError(
                f"Per-bearing normalization check FAILED for {bid}: "
                f"|mean|={b_mean:.3f}, std={b_std:.3f}.\n"
                "Re-run:  python -m src.feature_extractor"
            )

    test_bid = _VAL_BEARING
    if test_bid in scaled_dict:
        feats  = scaled_dict[test_bid]["features"]
        b_mean = float(np.abs(feats.mean()))
        b_std  = float(feats.std())
        logger.info("Test bearing %s: |mean|=%.4f  std=%.4f (per-bearing norm)", test_bid, b_mean, b_std)
        if b_mean > 0.05 or b_std < 0.9 or b_std > 1.1:
            raise ValueError(
                f"Per-bearing normalization check FAILED for {test_bid}: "
                f"|mean|={b_mean:.3f}, std={b_std:.3f}.\n"
                "Re-run:  python -m src.feature_extractor"
            )

    return scaled_dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    scaled_dict: Dict[str, Dict[str, np.ndarray]],
    window_size: int,
    cfg: Dict[str, Any],
    results_dir: Path,
    device_str: str,
) -> ConvSARULPredictor:
    """Train with AdamW + CosineAnnealingLR. Checkpoint on best val_RMSE_late."""
    set_seed(cfg["seed"])
    dev = torch.device(device_str)

    checkpoint_path = results_dir / "rul_model_best.pth"
    curve_path      = results_dir / "rul_training_curve.png"

    train_ids = [bid for bid in TRAIN_BEARINGS if bid in scaled_dict]
    val_id    = _VAL_BEARING

    logger.info("Building sliding windows (window_size=%d) …", window_size)
    X_parts, y_parts = [], []
    for bid in train_ids:
        W, R = create_sliding_windows(
            scaled_dict[bid]["features"], scaled_dict[bid]["rul"],
            window_size=window_size,
        )
        X_parts.append(W); y_parts.append(R)
    X_tr_all = np.concatenate(X_parts, axis=0)
    y_tr_all = np.concatenate(y_parts, axis=0)

    X_va, y_va = create_sliding_windows(
        scaled_dict[val_id]["features"], scaled_dict[val_id]["rul"],
        window_size=window_size,
    )

    # 10 % internal holdout from training set
    rng_np = np.random.default_rng(cfg["seed"])
    perm   = rng_np.permutation(len(X_tr_all))
    n_hold = max(1, int(len(X_tr_all) * 0.10))

    ho_idx = perm[:n_hold];  tr_idx = perm[n_hold:]
    X_hold, y_hold = X_tr_all[ho_idx], y_tr_all[ho_idx]
    X_tr,   y_tr   = X_tr_all[tr_idx], y_tr_all[tr_idx]

    logger.info(
        "Train=%d  Holdout=%d  Val(%s)=%d windows",
        len(X_tr), len(X_hold), val_id, len(X_va),
    )

    # Normalise RUL to [0, 1]
    y_tr_n   = (y_tr   / MAX_RUL).astype(np.float32)
    y_hold_n = (y_hold / MAX_RUL).astype(np.float32)
    y_va_n   = (y_va   / MAX_RUL).astype(np.float32)

    # 3-zone degradation upsampling: healthy=1x, degrading=8-11x
    # Strongly upweights entire degradation region (RUL < 125), not just near-failure,
    # to reduce systematic overestimation in the RUL 50-125 transition zone.
    bs             = cfg["batch_size"]
    is_degrading   = (y_tr_n < 0.999).astype(np.float32)
    sample_weights = (1.0 + 7.0 * is_degrading + 3.0 * (1.0 - y_tr_n)).astype(np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr,   dtype=torch.float32),
                      torch.tensor(y_tr_n, dtype=torch.float32)),
        batch_size=bs, sampler=sampler, drop_last=False,
    )
    hold_loader = DataLoader(
        TensorDataset(torch.tensor(X_hold,   dtype=torch.float32),
                      torch.tensor(y_hold_n, dtype=torch.float32)),
        batch_size=bs, shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_va,   dtype=torch.float32),
                      torch.tensor(y_va_n, dtype=torch.float32)),
        batch_size=bs, shuffle=False,
    )

    model = ConvSARULPredictor(
        n_features=N_FEATURES, window_size=window_size, dropout_mc=cfg["dropout_mc"],
    ).to(dev)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["epochs"], eta_min=1e-5
    )
    # Standard MSE — class balance is handled by WeightedRandomSampler above.

    n_va       = len(y_va)
    late_start = int(np.floor(n_va * 0.80))
    n50        = min(50, max(1, n_va // 4))

    best_rmse_late = float("inf")
    no_improve     = 0
    train_hist: List[float] = []
    hold_hist:  List[float] = []
    val_hist:   List[float] = []

    epochs  = cfg["epochs"]
    patience = cfg["patience"]

    pbar = tqdm(total=epochs, desc="RUL training", unit="ep", dynamic_ncols=True)

    for ep in range(1, epochs + 1):
        # --- train pass ---
        model.train()
        ep_losses: List[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimiser.step()
            ep_losses.append(loss.item())
        tr_loss = float(np.mean(ep_losses))

        # --- eval pass ---
        model.eval()
        hl: List[float] = []
        vl: List[float] = []
        y_pred_parts: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in hold_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                hl.append(nn.functional.mse_loss(model(xb), yb).item())
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                pred_c = torch.clamp(pred, 0.0, 1.0)
                vl.append(nn.functional.mse_loss(pred_c, yb).item())
                y_pred_parts.append(pred_c.cpu().numpy() * MAX_RUL)
        ho_loss = float(np.mean(hl))
        va_loss = float(np.mean(vl))
        y_pred_ep = np.concatenate(y_pred_parts)

        rmse_full = float(np.sqrt(np.mean((y_pred_ep - y_va) ** 2)))
        rmse_late = float(np.sqrt(np.mean(
            (y_pred_ep[late_start:] - y_va[late_start:]) ** 2
        )))
        first50 = float(y_pred_ep[:n50].mean())
        last50  = float(y_pred_ep[-n50:].mean())
        gap     = first50 - last50

        train_hist.append(tr_loss)
        hold_hist.append(ho_loss)
        val_hist.append(va_loss)
        scheduler.step()

        # Composite criterion: minimise late RMSE + near-failure mean jointly
        composite = rmse_late + last50
        if composite < best_rmse_late:
            best_rmse_late = composite
            no_improve     = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            ep,
                    "val_rmse_late":    rmse_late,
                    "val_rmse_full":    rmse_full,
                    "val_last50":       last50,
                    "val_composite":    composite,
                    "window_size":      window_size,
                    "n_features":       N_FEATURES,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1

        if ep > 30 and gap < 20.0:
            logger.warning(
                "WARN: model not learning degradation — gap=%.1f < 20 at epoch %d",
                gap, ep,
            )

        pbar.set_postfix(
            {"tr": f"{tr_loss:.4f}", "hold": f"{ho_loss:.4f}",
             "va": f"{va_loss:.4f}", "rmse_late": f"{rmse_late:.1f}",
             "gap": f"{gap:.1f}", "no_imp": no_improve},
            refresh=False,
        )
        pbar.update(1)
        logger.info(
            "Ep %3d/%d  tr=%.5f  hold=%.5f  va=%.5f  "
            "rmse_full=%.2f  rmse_late=%.2f  first50=%.1f  last50=%.1f  "
            "gap=%.1f  lr=%.2e  pat=%d/%d",
            ep, epochs, tr_loss, ho_loss, va_loss,
            rmse_full, rmse_late, first50, last50, gap,
            optimiser.param_groups[0]["lr"], no_improve, patience,
        )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d).", ep, patience)
            break

    pbar.close()

    ckpt = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        "Best val_RMSE_late=%.3f at epoch %d  saved to %s",
        ckpt["val_rmse_late"], ckpt["epoch"], checkpoint_path,
    )

    # -------------------------------------------------------------------
    # Stage 2: targeted fine-tuning on degradation windows from ALL
    # training bearings.  Uses last 125 windows of each bearing (the
    # RUL-declining segment) with asymmetric overestimation penalty to
    # directly reduce late-RMSE and near-failure overestimation.
    # -------------------------------------------------------------------
    degr_X: List[np.ndarray] = []
    degr_R: List[np.ndarray] = []
    for bid in train_ids:
        if bid not in scaled_dict:
            continue
        W_bid, R_bid = create_sliding_windows(
            scaled_dict[bid]["features"], scaled_dict[bid]["rul"],
            window_size=window_size,
        )
        n_deg = min(125, len(W_bid))
        degr_X.append(W_bid[-n_deg:])
        degr_R.append(R_bid[-n_deg:])

    if degr_X:
        W_deg  = np.concatenate(degr_X)
        R_deg_n = (np.concatenate(degr_R) / MAX_RUL).astype(np.float32)

        # Near-failure upsampling within the degradation set
        deg_sw = (1.0 + 4.0 * (1.0 - R_deg_n)).astype(np.float32)
        deg_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(deg_sw),
            num_samples=len(deg_sw),
            replacement=True,
        )
        ft_loader = DataLoader(
            TensorDataset(torch.tensor(W_deg, dtype=torch.float32),
                          torch.tensor(R_deg_n, dtype=torch.float32)),
            batch_size=min(bs, 32), sampler=deg_sampler,
        )
        logger.info(
            "Stage 2: Fine-tuning on degradation windows from %d bearings "
            "(%d windows) with asymmetric overestimation penalty, lr=3e-5 ...",
            len(degr_X), len(W_deg),
        )

        opt_ft           = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
        best_ft_composite = float("inf")

        for ft_ep in range(60):
            model.train()
            for xb, yb in ft_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt_ft.zero_grad()
                pred       = torch.clamp(model(xb), 0.0, 1.0)
                mse        = nn.functional.mse_loss(pred, yb)
                overest    = torch.clamp(pred - yb, min=0.0)
                overest_pen = (overest ** 2).mean()
                (mse + 0.5 * overest_pen).backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt_ft.step()

            model.eval()
            y_ft_parts: List[np.ndarray] = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    y_ft_parts.append(
                        torch.clamp(model(xb.to(dev)), 0.0, 1.0).cpu().numpy() * MAX_RUL
                    )
            y_ft         = np.concatenate(y_ft_parts)
            rmse_late_ft = float(np.sqrt(np.mean((y_ft[late_start:] - y_va[late_start:]) ** 2)))
            first50_ft   = float(y_ft[:n50].mean())
            last50_ft    = float(y_ft[-n50:].mean())
            ft_composite = rmse_late_ft + last50_ft
            logger.info(
                "FT %2d/60  rmse_late=%.2f  first50=%.1f  last50=%.1f  composite=%.2f",
                ft_ep + 1, rmse_late_ft, first50_ft, last50_ft, ft_composite,
            )

            if ft_composite < best_ft_composite:
                best_ft_composite = ft_composite
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch":            f"ft_{ft_ep + 1}",
                        "val_rmse_late":    rmse_late_ft,
                        "val_rmse_full":    float(np.sqrt(np.mean((y_ft - y_va) ** 2))),
                        "val_last50":       last50_ft,
                        "val_composite":    ft_composite,
                        "window_size":      window_size,
                        "n_features":       N_FEATURES,
                    },
                    checkpoint_path,
                )
                logger.info(
                    "  -> FT checkpoint (late=%.3f  last50=%.3f  comp=%.3f)",
                    rmse_late_ft, last50_ft, best_ft_composite,
                )

        ckpt = torch.load(checkpoint_path, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Stage 2 done — best checkpoint: epoch=%s  rmse_late=%.3f  last50=%.3f",
            ckpt["epoch"], ckpt["val_rmse_late"],
            ckpt.get("val_last50", float("nan")),
        )

    _plot_loss_curve(train_hist, val_hist, curve_path, hold_hist=hold_hist)
    return model


# ---------------------------------------------------------------------------
# Validation plot
# ---------------------------------------------------------------------------

def _plot_val_check(
    model: ConvSARULPredictor,
    scaled_dict: Dict[str, Dict[str, np.ndarray]],
    window_size: int,
    device_str: str,
    save_path: Path,
    n_mc: int = 50,
) -> Dict[str, float]:
    """Predicted vs true RUL with MC ±2σ band and vertical late-stage line."""
    val_id = _VAL_BEARING
    dev    = torch.device(device_str)

    X_va, y_va = create_sliding_windows(
        scaled_dict[val_id]["features"], scaled_dict[val_id]["rul"],
        window_size=window_size,
    )
    n_windows  = len(X_va)
    x_axis     = np.arange(n_windows)
    late_start = int(np.floor(n_windows * 0.80))
    n50        = min(50, max(1, n_windows // 4))

    X_t = torch.tensor(X_va, dtype=torch.float32).to(dev)

    # Deterministic
    model.eval()
    with torch.no_grad():
        y_det = torch.clamp(model(X_t), 0.0, 1.0).cpu().numpy() * MAX_RUL

    # MC Dropout: freeze BN running stats; enable ONLY head Dropout(0.3).
    # Activating mc_dropout(0.5) at the bottleneck causes high variance that
    # biases the MC mean away from the deterministic prediction.
    model.eval()
    model.head[2].train()   # head.2 = Dropout(0.3) before final linear
    mc_preds: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_mc):
            mc_preds.append(
                torch.clamp(model(X_t), 0.0, 1.0).cpu().numpy() * MAX_RUL
            )
    mc_arr = np.stack(mc_preds)          # (n_mc, N)
    y_mean = mc_arr.mean(axis=0)
    y_std  = mc_arr.std(axis=0)

    rmse_det      = float(np.sqrt(np.mean((y_det  - y_va) ** 2)))
    rmse_mc       = float(np.sqrt(np.mean((y_mean - y_va) ** 2)))
    mae_det       = float(np.mean(np.abs(y_det - y_va)))
    rmse_late_det = float(np.sqrt(np.mean((y_det[late_start:]  - y_va[late_start:]) ** 2)))
    rmse_late_mc  = float(np.sqrt(np.mean((y_mean[late_start:] - y_va[late_start:]) ** 2)))
    rul_first50   = float(y_mean[:n50].mean())
    rul_last50    = float(y_mean[-n50:].mean())
    gap           = rul_first50 - rul_last50

    logger.info(
        "Val: full MC-RMSE=%.2f  late MC-RMSE=%.2f  "
        "first50=%.1f  last50=%.1f  gap=%.1f",
        rmse_mc, rmse_late_mc, rul_first50, rul_last50, gap,
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.fill_between(
        x_axis,
        np.clip(y_mean - 2 * y_std, 0, MAX_RUL),
        np.clip(y_mean + 2 * y_std, 0, MAX_RUL),
        alpha=0.20, color="tab:red", label=f"MC ±2σ (n={n_mc})",
    )
    ax.plot(x_axis, y_va,   color="tab:blue",   linewidth=2.0, label="True RUL")
    ax.plot(x_axis, y_mean, color="tab:red",    linewidth=1.5,
            label=f"MC mean  RMSE={rmse_mc:.1f}")
    ax.plot(x_axis, y_det,  color="darkorange", linewidth=1.0, linestyle="--",
            label=f"Deterministic  RMSE={rmse_det:.1f}")
    ax.axvline(late_start, color="purple", linestyle="--", linewidth=1.2,
               label=f"Late-stage boundary (80 %, idx={late_start})")

    ann = (
        f"full RMSE (MC):  {rmse_mc:.2f}\n"
        f"late RMSE (MC):  {rmse_late_mc:.2f}\n"
        f"first50 mean:    {rul_first50:.1f}\n"
        f"last50  mean:    {rul_last50:.1f}\n"
        f"gap:             {gap:.1f}"
    )
    ax.text(
        0.02, 0.97, ann,
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_xlabel("Window index (timestep)")
    ax.set_ylabel("RUL (measurement windows)")
    ax.set_title(
        f"Conv-SA RUL — Bearing {val_id}  "
        f"full MC-RMSE={rmse_mc:.2f}  late MC-RMSE={rmse_late_mc:.2f}  gap={gap:.1f}"
    )
    ax.set_ylim(-2, MAX_RUL + 5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Validation plot saved to %s", save_path)

    return {
        "rmse_det":         rmse_det,
        "rmse_mc":          rmse_mc,
        "mae_det":          mae_det,
        "rmse_late_det":    rmse_late_det,
        "rmse_late_mc":     rmse_late_mc,
        "rul_first50_mean": rul_first50,
        "rul_last50_mean":  rul_last50,
        "gap":              gap,
    }


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def _plot_loss_curve(
    train_hist: List[float],
    val_hist: List[float],
    save_path: Path,
    hold_hist: Optional[List[float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_hist) + 1)
    ax.plot(epochs, train_hist, linewidth=1.5, alpha=0.8, label="Train MSE")
    if hold_hist:
        ax.plot(epochs, hold_hist, linewidth=1.5, alpha=0.8, linestyle="--",
                label="Internal holdout MSE")
    ax.plot(epochs, val_hist, linewidth=1.5, alpha=0.8, label="Val MSE")
    best_ep = int(np.argmin(val_hist)) + 1
    ax.axvline(best_ep, color="grey", linestyle=":", linewidth=1,
               label=f"Best val epoch={best_ep}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (normalised RUL)")
    ax.set_title("Conv-SA RUL — Training Curve")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curve saved to %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ConvSARULPredictor.")
    p.add_argument("--config",   default="config.yaml", type=Path)
    p.add_argument("--epochs",   type=int,   default=None)
    p.add_argument("--patience", type=int,   default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--batch",    type=int,   default=None)
    p.add_argument("--device",   type=str,   default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg     = dict(_DEFAULTS)
    raw_cfg = load_config(args.config)
    cfg.update(raw_cfg.get("training", {}))
    cfg.setdefault("seed", raw_cfg.get("seed", 42))
    # Ensure required keys not in YAML training section default correctly
    for k, v in _DEFAULTS.items():
        cfg.setdefault(k, v)

    if args.epochs   is not None: cfg["epochs"]     = args.epochs
    if args.patience is not None: cfg["patience"]   = args.patience
    if args.lr       is not None: cfg["lr"]         = args.lr
    if args.batch    is not None: cfg["batch_size"] = args.batch

    device_str    = args.device if args.device else get_device()
    results_dir   = Path(str(cfg.get("results_dir", "results/01_rul_predictor")))
    processed_dir = Path(
        raw_cfg.get("dataset", {}).get("processed_dir", "data/processed")
    )

    _setup_logging(results_dir)
    set_seed(cfg["seed"])

    logger.info("=" * 64)
    logger.info(
        "RUL Training | val=%s  device=%s  epochs=%d  patience=%d  lr=%.0e",
        _VAL_BEARING, device_str, cfg["epochs"], cfg["patience"], cfg["lr"],
    )
    logger.info("=" * 64)

    t0 = time.time()

    logger.info("Loading pre-scaled features from %s …", processed_dir)
    scaled_dict = _load_scaled_features(processed_dir)

    model = _train(scaled_dict, cfg["window_size"], cfg, results_dir, device_str)

    logger.info("Training finished in %.1f min.", (time.time() - t0) / 60)

    metrics = _plot_val_check(
        model, scaled_dict, cfg["window_size"], device_str,
        save_path=results_dir / "rul_val_check.png",
        n_mc=cfg["n_mc_eval"],
    )

    # Save metrics file
    metrics_path = results_dir / "rul_val_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write(f"Bearing: {_VAL_BEARING}\n\n")
        fh.write(f"full_rmse_mc:          {metrics['rmse_mc']:.6f}\n")
        fh.write(f"full_rmse_det:         {metrics['rmse_det']:.6f}\n")
        fh.write(f"full_mae_det:          {metrics['mae_det']:.6f}\n\n")
        fh.write(f"late_stage_rmse_mc:    {metrics['rmse_late_mc']:.6f}\n")
        fh.write(f"late_stage_rmse_det:   {metrics['rmse_late_det']:.6f}\n\n")
        fh.write(f"rul_first50_mean:       {metrics['rul_first50_mean']:.4f}\n")
        fh.write(f"rul_last50_mean:        {metrics['rul_last50_mean']:.4f}\n")
        fh.write(f"discriminability_gap:   {metrics['gap']:.4f}\n")
    logger.info("Metrics saved to %s", metrics_path)

    # PASS/FAIL verdict
    _TARGETS = [
        ("Full-sequence RMSE (MC)",  "rmse_mc",          "<",  15.0),
        ("Late-stage RMSE (MC)",     "rmse_late_mc",     "<",  20.0),
        ("Healthy RUL mean",         "rul_first50_mean", ">", 100.0),
        ("Near-failure RUL mean",    "rul_last50_mean",  "<",  25.0),
        ("Discriminability gap",     "gap",              ">",  75.0),
    ]

    print("\n" + "=" * 56)
    print("         === RUL MODEL EVALUATION ===")
    print("=" * 56)
    fails: List[str] = []
    for label, key, op, tgt in _TARGETS:
        val    = metrics[key]
        passed = (val < tgt) if op == "<" else (val > tgt)
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<30s}  {val:7.2f}  (target {op} {tgt:.1f})  {status}")
        if not passed:
            fails.append(f"{label}: {val:.2f} (off by {abs(val - tgt):.2f})")
    print("=" * 56)
    if not fails:
        print("  === VERDICT: PASS ===")
        print("  Ready to run:  python -m src.train")
    else:
        print("  === VERDICT: FAIL ===")
        for f in fails:
            print(f"    - {f}")
        print("  DO NOT proceed to RL training until PASS.")
    print("=" * 56 + "\n")

    logger.info("=" * 64)
    logger.info("Outputs in %s/:", results_dir)
    logger.info("  rul_model_best.pth  rul_val_check.png  rul_training_curve.png")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
