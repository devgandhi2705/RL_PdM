"""
uncertainty_validation.py
==========================
Three inference modes and calibration evaluation for Conv-SA RUL predictor.

Modes
-----
1. Deterministic  -- single forward pass, model.eval(), all dropout off
2. MC Dropout     -- 50 passes with model.train() (all dropout active)
3. Deep Ensemble  -- 5 independent Conv-SA instances trained from scratch,
                     each in eval mode at inference; diversity from init seed

Calibration metrics: PICP, MPIW, ECE (scipy.stats.norm for z-scores)

Outputs (written to results/)
------------------------------
ensemble_{seed}.pth            one checkpoint per ensemble member
fig_reliability.png            reliability diagrams, 3 subplots
fig_uncertainty_ci.png         CI comparison on bearing 3_2
table_uncertainty.csv / .tex   calibration metrics table

Usage
-----
    python src/uncertainty_validation.py
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as _norm
from torch.utils.data import DataLoader, TensorDataset

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from src.device import get_device
    from src.rul_predictor import (
        ConvSARULPredictor, MAX_RUL, N_FEATURES, create_sliding_windows,
    )
except ImportError:
    from device import get_device                                  # type: ignore[no-redef]
    from rul_predictor import (                                    # type: ignore[no-redef]
        ConvSARULPredictor, MAX_RUL, N_FEATURES, create_sliding_windows,
    )

logger = logging.getLogger(__name__)

TRAIN_BEARINGS: List[str] = ["1_1", "1_2", "2_1", "2_2", "3_1"]
TEST_BEARING:   str       = "3_2"
WINDOW_SIZE:    int       = 32
INFER_BATCH:    int       = 256
ENSEMBLE_SEEDS: List[int] = [42, 123, 456, 789, 1024]
MC_SAMPLES:     int       = 50


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_bearing_windows(
    proc_dir:    Path,
    bearing_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for bid in bearing_ids:
        fp = proc_dir / f"{bid}_features.npy"
        rp = proc_dir / f"{bid}_rul.npy"
        if not fp.exists() or not rp.exists():
            logger.warning("Bearing %s missing — skipped.", bid)
            continue
        feat = np.load(fp).astype(np.float32)
        rul  = np.load(rp).astype(np.float32)
        X, y = create_sliding_windows(feat, rul, window_size=WINDOW_SIZE, stride=1)
        Xs.append(X); ys.append(y)
    if not Xs:
        raise FileNotFoundError(f"No bearing files in {proc_dir}")
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _train_val_split(
    X: np.ndarray, y: np.ndarray,
    val_frac: float = 0.10,
    seed:     int   = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X))
    split = int(len(X) * (1.0 - val_frac))
    tr, va = idx[:split], idx[split:]
    return X[tr], y[tr], X[va], y[va]


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def _load_convsa(ckpt_path: Path, device: str = "cpu") -> ConvSARULPredictor:
    model = ConvSARULPredictor()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    return model.to(torch.device(device))


def _forward_batch(
    model: ConvSARULPredictor,
    X_t:   torch.Tensor,
    dev:   torch.device,
) -> np.ndarray:
    """Run one full pass over X_t in mini-batches; return (M,) in [0, 125]."""
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_t), INFER_BATCH):
            xb = X_t[i : i + INFER_BATCH].to(dev)
            preds.append(
                torch.clamp(model(xb), 0.0, 1.0).cpu().numpy() * MAX_RUL
            )
    return np.concatenate(preds).astype(np.float32)


# ---------------------------------------------------------------------------
# Inference Mode 1 — Deterministic
# ---------------------------------------------------------------------------

def infer_deterministic(
    model:  ConvSARULPredictor,
    X:      np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Single forward pass per window; dropout off. Returns (M,) predictions."""
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    return _forward_batch(model, torch.tensor(X, dtype=torch.float32), dev)


# ---------------------------------------------------------------------------
# Inference Mode 2 — MC Dropout
# ---------------------------------------------------------------------------

def infer_mc_dropout(
    model:     ConvSARULPredictor,
    X:         np.ndarray,
    n_samples: int = MC_SAMPLES,
    device:    str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """n_samples stochastic forward passes for MC Dropout uncertainty estimation.

    Uses model.eval() so BatchNorm reads stored running statistics (not batch
    statistics).  Then sets every nn.Dropout module to train() so they stay
    active during inference.  This prevents BatchNorm from corrupting its
    running stats while keeping all dropout paths stochastic.

    Returns
    -------
    mean_rul : (M,) in [0, 125]
    sigma    : (M,) — predictive std across passes
    """
    dev = torch.device(device)
    model.eval()                                     # BN uses running stats
    for mod in model.modules():                      # re-enable every Dropout
        if isinstance(mod, nn.Dropout):
            mod.train()
    model.to(dev)
    X_t      = torch.tensor(X, dtype=torch.float32)
    all_runs: List[np.ndarray] = []
    for _ in range(n_samples):
        all_runs.append(_forward_batch(model, X_t, dev))
    samples = np.stack(all_runs, axis=0)             # (n_samples, M)
    return samples.mean(axis=0).astype(np.float32), samples.std(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Inference Mode 3 — Deep Ensemble
# ---------------------------------------------------------------------------

def _train_one_member(
    seed:         int,
    X_tr:         np.ndarray,
    y_tr:         np.ndarray,
    X_va:         np.ndarray,
    y_va:         np.ndarray,
    device:       str   = "cpu",
    epochs:       int   = 150,
    patience:     int   = 20,
    batch_size:   int   = 64,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
) -> ConvSARULPredictor:
    """Train one Conv-SA from scratch with a fixed seed for diverse initialization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev     = torch.device(device)
    model   = ConvSARULPredictor().to(dev)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    # Targets normalised to [0,1] — model outputs unbounded linear values
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr / MAX_RUL, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    Xv_t     = torch.tensor(X_va, dtype=torch.float32).to(dev)
    yv_scale = y_va.astype(np.float64)   # ground-truth in [0, 125] for RMSE

    best_rmse = float("inf")
    best_wts  = copy.deepcopy(model.state_dict())
    no_imp    = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.clamp(model(Xv_t), 0.0, 1.0).cpu().numpy() * MAX_RUL
        val_rmse = float(np.sqrt(np.mean((val_pred.astype(np.float64) - yv_scale) ** 2)))

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_wts  = copy.deepcopy(model.state_dict())
            no_imp    = 0
        else:
            no_imp += 1

        if ep % 50 == 0:
            print(f"    seed={seed} ep {ep:3d}/{epochs}  val={val_rmse:.2f}  best={best_rmse:.2f}")
        if no_imp >= patience:
            print(f"    seed={seed} early-stop ep {ep}  best={best_rmse:.2f}")
            break

    model.load_state_dict(best_wts)
    return model.cpu()


def load_or_train_ensemble(
    proc_dir:    Path,
    results_dir: Path,
    device:      str = "cpu",
) -> List[ConvSARULPredictor]:
    """Return 5-member ensemble, loading from disk or training if missing."""
    ckpts = [results_dir / f"ensemble_{s}.pth" for s in ENSEMBLE_SEEDS]

    if all(p.exists() for p in ckpts):
        print("  Loading pre-trained ensemble from disk...")
        models: List[ConvSARULPredictor] = []
        for s, p in zip(ENSEMBLE_SEEDS, ckpts):
            m    = ConvSARULPredictor()
            sd   = torch.load(p, map_location="cpu", weights_only=False)
            m.load_state_dict(sd.get("model_state_dict", sd))
            models.append(m)
            print(f"    loaded ensemble_{s}.pth")
        return models

    print("  Training 5-member deep ensemble (this takes a while)...")
    X_all, y_all = _load_bearing_windows(proc_dir, TRAIN_BEARINGS)
    # Fixed data split (seed=0); diversity comes from model init seed only
    X_tr, y_tr, X_va, y_va = _train_val_split(X_all, y_all, val_frac=0.10, seed=0)
    print(f"  train={len(X_tr):,}  val={len(X_va):,}")

    models = []
    for i, seed in enumerate(ENSEMBLE_SEEDS, 1):
        print(f"\n  [Member {i}/{len(ENSEMBLE_SEEDS)}, seed={seed}]")
        m    = _train_one_member(seed, X_tr, y_tr, X_va, y_va, device=device)
        ckpt = results_dir / f"ensemble_{seed}.pth"
        torch.save({"model_state_dict": m.state_dict(), "seed": seed}, ckpt)
        print(f"  Saved -> {ckpt.name}")
        models.append(m)

    return models


def infer_ensemble(
    models: List[ConvSARULPredictor],
    X:      np.ndarray,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Eval-mode pass through each member; return (mean_rul, sigma) shape (M,)."""
    dev  = torch.device(device)
    X_t  = torch.tensor(X, dtype=torch.float32)
    runs: List[np.ndarray] = []
    for m in models:
        m.eval()
        m.to(dev)
        runs.append(_forward_batch(m, X_t, dev))
    stacked = np.stack(runs, axis=0)   # (n_models, M)
    return stacked.mean(axis=0).astype(np.float32), stacked.std(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_calibration(
    pred_means:        np.ndarray,
    pred_stds:         np.ndarray,
    true_rul:          np.ndarray,
    confidence_levels: List[float] = None,
) -> Dict[str, Any]:
    """Compute PICP, MPIW, and calibration error at each confidence level.

    Parameters
    ----------
    pred_means, pred_stds, true_rul : (M,) arrays in [0, 125]
    confidence_levels : list of alpha in (0, 1]

    Returns
    -------
    dict mapping each alpha to {"picp", "mpiw", "cal_error"},
    plus "ECE" (mean calibration error across levels).
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.75, 0.9, 0.95]

    means = np.asarray(pred_means, dtype=np.float64)
    stds  = np.asarray(pred_stds,  dtype=np.float64)
    true  = np.asarray(true_rul,   dtype=np.float64)

    out: Dict[str, Any] = {}
    cal_errors: List[float] = []

    for alpha in confidence_levels:
        z = float(_norm.ppf((1.0 + alpha) / 2.0))   # z_{(1+alpha)/2}

        if math.isinf(z):
            # alpha == 1.0: trivially, the infinite interval covers everything
            picp  = 1.0
            mpiw  = float("inf")
        else:
            lower   = means - z * stds
            upper   = means + z * stds
            covered = (true >= lower) & (true <= upper)
            picp    = float(np.mean(covered))
            mpiw    = float(np.mean(upper - lower))

        cal_err = abs(picp - alpha)
        out[alpha] = {"picp": picp, "mpiw": mpiw, "cal_error": cal_err}
        cal_errors.append(cal_err)

    out["ECE"] = float(np.mean(cal_errors))
    return out


# ---------------------------------------------------------------------------
# Figure: reliability diagrams
# ---------------------------------------------------------------------------

_IEEE_RC: Dict[str, Any] = {
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        10,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.2,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.5,
    "axes.spines.right": False,
    "axes.spines.top":   False,
}

_DIAG_ALPHAS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


def plot_reliability(
    results_dir: Path,
    calib_all:   Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Reliability diagram: expected vs observed coverage, one panel per method.

    Parameters
    ----------
    calib_all : dict mapping method_name -> (picps, alphas)
                each computed at _DIAG_ALPHAS
    """
    plt.rcParams.update(_IEEE_RC)
    methods    = list(calib_all.keys())
    panel_lbls = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)

    for ax, name, lbl in zip(axes, methods, panel_lbls):
        picps, alphas = calib_all[name]
        ece = float(np.mean(np.abs(picps - alphas)))

        # Shading: red where over-confident (PICP < alpha), blue where under-confident
        ax.fill_between(
            alphas, picps, alphas,
            where=(picps <= alphas),
            color="red",  alpha=0.25, interpolate=True,
        )
        ax.fill_between(
            alphas, alphas, picps,
            where=(picps >= alphas),
            color="blue", alpha=0.25, interpolate=True,
        )

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="Perfect")
        ax.plot(alphas, picps, "o-", markersize=4, linewidth=1.3,
                color="#0072B2", label="Observed")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Expected coverage")
        if ax is axes[0]:
            ax.set_ylabel("Observed coverage (PICP)")
        ax.set_title(name, fontsize=9)
        ax.text(
            0.04, 0.93, f"ECE = {ece:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )
        ax.text(0.04, 0.05, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

    fig.tight_layout()
    out = results_dir / "fig_reliability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Reliability diagram -> {out.name}")


# ---------------------------------------------------------------------------
# Figure: confidence interval comparison on bearing 3_2
# ---------------------------------------------------------------------------

def plot_uncertainty_ci(
    results_dir: Path,
    true_rul:    np.ndarray,
    det_pred:    np.ndarray,
    mc_mean:     np.ndarray,
    mc_std:      np.ndarray,
    ens_mean:    np.ndarray,
    ens_std:     np.ndarray,
) -> None:
    """Overlay CI bands for MC Dropout and Deep Ensemble on bearing 3_2."""
    plt.rcParams.update(_IEEE_RC)

    M           = len(true_rul)
    t           = np.arange(M)
    late_start  = int(M * 0.80)

    fig, ax = plt.subplots(figsize=(7, 4))

    # True RUL
    ax.plot(t, true_rul, color="black", linewidth=1.4, label="True RUL", zorder=5)

    # Deterministic — orange dotted, no CI band
    ax.plot(t, det_pred, color="#E69F00", linestyle=":", linewidth=1.2,
            label="Deterministic", zorder=4)

    # MC Dropout — red dashed + shading
    ax.plot(t, mc_mean, color="#D62728", linestyle="--", linewidth=1.2,
            label="MC Dropout (mean)", zorder=4)
    ax.fill_between(
        t,
        np.maximum(mc_mean - 1.96 * mc_std, 0),
        mc_mean + 1.96 * mc_std,
        color="#D62728", alpha=0.15, label="MC Dropout 95% CI",
    )

    # Deep Ensemble — blue dashed + shading
    ax.plot(t, ens_mean, color="#1F77B4", linestyle="--", linewidth=1.2,
            label="Deep Ensemble (mean)", zorder=4)
    ax.fill_between(
        t,
        np.maximum(ens_mean - 1.96 * ens_std, 0),
        ens_mean + 1.96 * ens_std,
        color="#1F77B4", alpha=0.15, label="Deep Ensemble 95% CI",
    )

    # Late-stage boundary
    ax.axvline(late_start, color="grey", linestyle="--", linewidth=0.8,
               alpha=0.7, label="80% mark (late-stage)")

    ax.set_xlabel("Timestep (window index)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("RUL Prediction with Uncertainty — Bearing 3\\_2")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(0, M - 1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = results_dir / "fig_uncertainty_ci.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  CI plot -> {out.name}")


# ---------------------------------------------------------------------------
# Table: calibration summary
# ---------------------------------------------------------------------------

def generate_table(
    results_dir: Path,
    summary:     Dict[str, Dict[str, float]],
) -> None:
    """Write table_uncertainty.csv and .tex.

    summary maps method_name -> {rmse, picp_90, picp_95, mpiw_95, ece}
    """
    methods = list(summary.keys())
    cols    = ["RMSE", "PICP@90%", "PICP@95%", "MPIW@95%", "ECE"]

    # Best (lowest) per column
    def _best(key: str) -> float:
        vals = [summary[m][key] for m in methods if not math.isnan(summary[m][key])]
        return min(vals) if vals else float("nan")

    best = {
        "RMSE":    _best("rmse"),
        "PICP@90%": max(summary[m]["picp_90"] for m in methods),   # higher PICP@90 closer to 0.9 is best
        "PICP@95%": max(summary[m]["picp_95"] for m in methods),
        "MPIW@95%": _best("mpiw_95"),
        "ECE":     _best("ece"),
    }
    # For PICP we want the value closest to the target (90% or 95%), not simply highest or lowest
    # Re-compute "best" as closest to target for PICP columns
    best["PICP@90%"] = min(methods, key=lambda m: abs(summary[m]["picp_90"] - 0.90))
    best["PICP@95%"] = min(methods, key=lambda m: abs(summary[m]["picp_95"] - 0.95))

    def _isnan(v: float) -> bool:
        return math.isnan(v)

    # CSV
    csv_path = results_dir / "table_uncertainty.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["Method"] + cols)
        wr.writeheader()
        for name, m in summary.items():
            wr.writerow({
                "Method":    name,
                "RMSE":      "" if _isnan(m["rmse"])    else f"{m['rmse']:.2f}",
                "PICP@90%":  "" if _isnan(m["picp_90"]) else f"{m['picp_90']:.3f}",
                "PICP@95%":  "" if _isnan(m["picp_95"]) else f"{m['picp_95']:.3f}",
                "MPIW@95%":  "" if _isnan(m["mpiw_95"]) else f"{m['mpiw_95']:.2f}",
                "ECE":       "" if _isnan(m["ece"])      else f"{m['ece']:.4f}",
            })

    # LaTeX
    def _cell(v: float, col: str, name: str) -> str:
        if _isnan(v):
            return "---"
        if col in ("PICP@90%", "PICP@95%"):
            s  = f"{v:.3f}"
            ok = (name == best[col])
        else:
            s  = f"{v:.2f}" if col in ("RMSE", "MPIW@95%") else f"{v:.4f}"
            ok = (abs(v - (_best("rmse") if col == "RMSE"
                           else _best("mpiw_95") if col == "MPIW@95%"
                           else _best("ece"))) < 0.0005)
        return f"\\textbf{{{s}}}" if ok else s

    tex_path = results_dir / "table_uncertainty.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Uncertainty calibration on PRONOSTIA Bearing 3\\_2}\n")
        f.write("\\label{tab:uncertainty}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Method & RMSE & PICP@90\\% & PICP@95\\% & MPIW@95\\% & ECE \\\\\n\\midrule\n")
        for name, m in summary.items():
            cells = [
                _cell(m["rmse"],    "RMSE",    name),
                _cell(m["picp_90"], "PICP@90%", name),
                _cell(m["picp_95"], "PICP@95%", name),
                _cell(m["mpiw_95"], "MPIW@95%", name),
                _cell(m["ece"],     "ECE",      name),
            ]
            f.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  Table -> {csv_path.name}  {tex_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    _proc_dir    = _PROJ / "data" / "processed"
    _results_dir = _PROJ / "results"
    _results_dir.mkdir(parents=True, exist_ok=True)

    _ckpt   = _results_dir / "rul_model_best.pth"
    _device = get_device(verbose=True)

    if not _ckpt.exists():
        print(f"ERROR: {_ckpt} not found. Run src/train_rul.py first.")
        sys.exit(1)

    # ---- Load test data -----------------------------------------------------
    print("\nLoading test windows (bearing 3_2)...")
    X_test, y_test = _load_bearing_windows(_proc_dir, [TEST_BEARING])
    M = len(X_test)
    print(f"  {M} windows  rul_range=[{y_test.min():.0f}, {y_test.max():.0f}]")

    # ---- Mode 1: Deterministic ----------------------------------------------
    print("\n[Mode 1] Deterministic inference...")
    m1 = _load_convsa(_ckpt, device=_device)
    det_pred = infer_deterministic(m1, X_test, device=_device)
    det_rmse = float(np.sqrt(np.mean((det_pred.astype(np.float64) - y_test.astype(np.float64)) ** 2)))
    print(f"  RMSE = {det_rmse:.2f}")

    # ---- Mode 2: MC Dropout -------------------------------------------------
    print(f"\n[Mode 2] MC Dropout ({MC_SAMPLES} passes)...")
    m2 = _load_convsa(_ckpt, device=_device)
    mc_mean, mc_std = infer_mc_dropout(m2, X_test, n_samples=MC_SAMPLES, device=_device)
    mc_rmse = float(np.sqrt(np.mean((mc_mean.astype(np.float64) - y_test.astype(np.float64)) ** 2)))
    print(f"  RMSE = {mc_rmse:.2f}  mean_sigma = {mc_std.mean():.2f}")

    # ---- Mode 3: Deep Ensemble ----------------------------------------------
    print("\n[Mode 3] Deep Ensemble (5 members)...")
    ensemble = load_or_train_ensemble(_proc_dir, _results_dir, device=_device)
    ens_mean, ens_std = infer_ensemble(ensemble, X_test, device=_device)
    ens_rmse = float(np.sqrt(np.mean((ens_mean.astype(np.float64) - y_test.astype(np.float64)) ** 2)))
    print(f"  RMSE = {ens_rmse:.2f}  mean_sigma = {ens_std.mean():.2f}")

    # ---- Calibration --------------------------------------------------------
    print("\nComputing calibration metrics...")

    # For reliability diagrams: 10 points including near-1.0
    _diag_levels = list(_DIAG_ALPHAS)

    def _picps_at(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        c = compute_calibration(means, stds, y_test, confidence_levels=_diag_levels)
        return np.array([c[a]["picp"] for a in _diag_levels])

    # Deterministic has pred_std = 0 everywhere (shows as maximally overconfident)
    det_stds_zero = np.zeros(M, dtype=np.float32)

    calib_diag: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "Deterministic": (_picps_at(det_pred,  det_stds_zero), _DIAG_ALPHAS),
        "MC Dropout":    (_picps_at(mc_mean,   mc_std),        _DIAG_ALPHAS),
        "Deep Ensemble": (_picps_at(ens_mean,  ens_std),       _DIAG_ALPHAS),
    }

    def _calib_at(means: np.ndarray, stds: np.ndarray) -> Dict[str, float]:
        c = compute_calibration(means, stds, y_test,
                                confidence_levels=[0.5, 0.75, 0.9, 0.95])
        return {
            "picp_90": c[0.9]["picp"],
            "picp_95": c[0.95]["picp"],
            "mpiw_95": c[0.95]["mpiw"],
            "ece":     c["ECE"],
        }

    summary: Dict[str, Dict[str, float]] = {
        "Deterministic": {"rmse": det_rmse, **_calib_at(det_pred, det_stds_zero)},
        "MC Dropout":    {"rmse": mc_rmse,  **_calib_at(mc_mean,  mc_std)},
        "Deep Ensemble": {"rmse": ens_rmse, **_calib_at(ens_mean, ens_std)},
    }

    # ---- Figures + table ----------------------------------------------------
    print("\nGenerating outputs...")
    plot_reliability(_results_dir, calib_diag)
    plot_uncertainty_ci(
        _results_dir,
        true_rul=y_test,
        det_pred=det_pred,
        mc_mean=mc_mean, mc_std=mc_std,
        ens_mean=ens_mean, ens_std=ens_std,
    )
    generate_table(_results_dir, summary)

    # ---- Summary to stdout --------------------------------------------------
    _W = 15
    print(f"\n{'='*65}")
    print(f"  {'Method':<{_W}} {'RMSE':>7} {'PICP@90':>9} {'PICP@95':>9} "
          f"{'MPIW@95':>9} {'ECE':>8}")
    print(f"  {'-'*60}")
    for name, m in summary.items():
        picp_95_str = f"{m['picp_95']:.3f}" if not math.isnan(m['picp_95']) else " N/A "
        mpiw_str    = f"{m['mpiw_95']:.2f}" if not math.isnan(m['mpiw_95']) else "  inf "
        print(
            f"  {name:<{_W}} {m['rmse']:7.2f} {m['picp_90']:9.3f} "
            f"{picp_95_str:>9} {mpiw_str:>9} {m['ece']:8.4f}"
        )
    print(f"{'='*65}")
    print(f"  Outputs -> {_results_dir}")
    print("  Perfect calibration: PICP@90%=0.900, PICP@95%=0.950, ECE=0.000")
