"""
evaluate.py
===========
Generate all paper figures and tables for the Risk-Averse PdM paper.

Outputs (written to results/)
------------------------------
fig1_rul_prediction.png   — Conv-SA RUL prediction on bearing 3_2 with MC CI
fig2_health_index.png     — All 6 bearings' HI over normalised time
fig3_policy_comparison.png — Cost and catastrophe-rate bar charts
fig4_action_composition.png — Stacked action fractions per policy
fig5_training_curve.png   — Smoothed training reward + eval checkpoints
table1_rul.tex / .csv     — RUL comparison (Conv-SA vs LSTM vs CNN baselines)
table2_policy.tex / .csv  — Policy comparison table

Usage
-----
    python src/evaluate.py                        # full run
    python src/evaluate.py --dry-run              # fast checks only, no baseline training
    python src/evaluate.py --figures-only         # skip table generation
    python src/evaluate.py --no-baseline-train    # skip LSTM/CNN training in Table 1
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from src.baselines import (
        CorrectiveMaintenance, PeriodicPM, ThresholdPolicy, RiskNeutralDQN,
        evaluate_policy,
    )
    from src.device import get_device
    from src.qrdqn_agent import QRDQNAgent
    from src.rl_environment import make_env_from_processed
    from src.rul_predictor import (
        ConvSARULPredictor, create_sliding_windows,
        mc_dropout_inference, MAX_RUL, N_FEATURES,
    )
except ImportError as _e:
    raise ImportError(f"Could not import src modules: {_e}") from _e

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size":        11,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "axes.spines.right": False,
        "axes.spines.top":   False,
    })


# ---------------------------------------------------------------------------
# Figure 1 — RUL prediction on bearing 3_2
# ---------------------------------------------------------------------------

def fig1_rul_prediction(
    results_dir: Path,
    proc_dir:    Path,
    n_mc:        int = 50,
) -> bool:
    """Plot true vs MC-predicted RUL on bearing 3_2 with uncertainty band."""
    ckpt = results_dir / "rul_model_best.pth"
    feat_path = proc_dir / "3_2_features.npy"
    rul_path  = proc_dir / "3_2_rul.npy"

    missing = [p for p in (ckpt, feat_path, rul_path) if not p.exists()]
    if missing:
        print(f"  [Fig1] SKIP — missing: {[str(p.name) for p in missing]}")
        return False

    print("  [Fig1] Running MC inference on bearing 3_2 …")
    model = ConvSARULPredictor()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    _ep  = state.get("epoch", "?")
    _fr  = state.get("val_rmse_full", float("nan"))
    _lr  = state.get("val_rmse_late", float("nan"))
    print(f"  [Fig1] Checkpoint: epoch={_ep}  stored full-RMSE={_fr:.2f}  late-RMSE={_lr:.2f}")

    features = np.load(feat_path).astype(np.float32)  # (N, 32)
    rul_raw  = np.load(rul_path).astype(np.float32)   # (N,) in [0,1] or [0,125]

    # Normalise RUL to [0,125] if stored as fractions
    if rul_raw.max() <= 1.05:
        rul_full = rul_raw * MAX_RUL
    else:
        rul_full = rul_raw

    W = 32  # window size
    windows, true_rul = create_sliding_windows(features, rul_full, window_size=W, stride=1)
    # windows: (M, W, 32);  true_rul: (M,)
    M = len(windows)

    # MC inference
    mc_means   = np.zeros(M, dtype=np.float32)
    mc_stds    = np.zeros(M, dtype=np.float32)
    determ     = np.zeros(M, dtype=np.float32)

    BATCH = 64
    X_t = torch.tensor(windows, dtype=torch.float32)
    for start in range(0, M, BATCH):
        xb = X_t[start:start + BATCH]
        mn, var, _ = mc_dropout_inference(model, xb.numpy(), n_samples=n_mc)
        mc_means[start:start + BATCH] = mn
        mc_stds[start:start + BATCH]  = np.sqrt(np.maximum(var, 0.0))
        det_mn, _, _ = mc_dropout_inference(model, xb.numpy(), n_samples=1)
        determ[start:start + BATCH] = det_mn

    t = np.arange(M)
    late_start = int(M * 0.80)

    full_rmse = float(np.sqrt(np.mean((mc_means - true_rul) ** 2)))
    late_rmse = float(np.sqrt(np.mean((mc_means[late_start:] - true_rul[late_start:]) ** 2)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, true_rul, color="steelblue", linewidth=1.8, label="True RUL")
    ax.plot(t, mc_means, color="red",    linewidth=1.4, linestyle="--", label=f"MC mean (n={n_mc})")
    ax.fill_between(
        t,
        np.maximum(mc_means - 2 * mc_stds, 0),
        mc_means + 2 * mc_stds,
        color="salmon", alpha=0.30, label="MC ±2σ (95% CI)",
    )
    ax.plot(t, determ, color="darkorange", linewidth=1.0, linestyle="--",
            alpha=0.75, label="Deterministic (dropout off)")
    ax.axvline(late_start, color="grey", linestyle="--", linewidth=0.9, alpha=0.7,
               label="80% of sequence (late-stage)")

    ax.text(
        0.97, 0.97,
        f"Full RMSE: {full_rmse:.2f}\nLate RMSE: {late_rmse:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    ax.set_xlabel("Timestep (window index)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("Conv-SA RUL Prediction — Bearing 3_2 (PRONOSTIA Condition 3)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    out = results_dir / "fig1_rul_prediction.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig1] saved ->{out.name}  (full_RMSE={full_rmse:.2f}, late_RMSE={late_rmse:.2f})")
    return True


# ---------------------------------------------------------------------------
# Figure 2 — Health Index over time (all 6 bearings)
# ---------------------------------------------------------------------------

def fig2_health_index(results_dir: Path, proc_dir: Path) -> bool:
    """All 6 bearings on the same plot with normalised time axis."""
    from src.data_loader import ALL_BEARINGS

    hi_data: Dict[str, np.ndarray] = {}
    for bid in ALL_BEARINGS:
        p = proc_dir / f"{bid}_hi.npy"
        if p.exists():
            hi_data[bid] = np.load(p).astype(np.float32)

    if not hi_data:
        print(f"  [Fig2] SKIP — no *_hi.npy files in {proc_dir}")
        return False

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (bid, hi) in enumerate(sorted(hi_data.items())):
        t = np.linspace(0, 1, len(hi))
        ax.plot(t, hi, color=colors[idx % len(colors)],
                linewidth=1.5, alpha=0.85, label=f"Bearing {bid}")

    ax.set_xlabel("Normalised time  (0 = new, 1 = end-of-life)")
    ax.set_ylabel("Health Index  (1 = healthy, 0 = failed)")
    ax.set_title("PRONOSTIA Bearing Health Index — All 6 Bearings")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.25)

    out = results_dir / "fig2_health_index.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig2] saved ->{out.name}  ({len(hi_data)} bearings)")
    return True


# ---------------------------------------------------------------------------
# Policy evaluation helper (shared by Figs 3/4 and Table 2)
# ---------------------------------------------------------------------------

def _collect_policy_results(
    proc_dir:    Path,
    results_dir: Path,
    n_episodes:  int = 300,
    seed:        int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate 5 policies and return metrics dict."""
    env = make_env_from_processed(proc_dir, seed=seed)

    ckpt = results_dir / "qrdqn_best.pth"
    state_dim = 5

    # CVaR QR-DQN
    cvar_agent = QRDQNAgent(state_dim=state_dim, n_actions=3, risk_mode="cvar")
    if ckpt.exists():
        cvar_agent.load_checkpoint(ckpt)
        cvar_agent.epsilon = 0.0
    else:
        print("  WARNING: no qrdqn_best.pth found — QR-DQN columns will be random policy.")

    # Risk-neutral DQN (same weights, mean aggregation)
    rn_agent = QRDQNAgent(state_dim=state_dim, n_actions=3, risk_mode="mean")
    if ckpt.exists():
        rn_agent.load_checkpoint(ckpt)
        rn_agent.risk_mode = "mean"
        rn_agent.epsilon = 0.0

    policies: Dict[str, Any] = {
        "CM (Corrective)":        CorrectiveMaintenance(),
        "PM (Periodic-50)":       PeriodicPM(interval=50),
        "ThresholdPolicy":        ThresholdPolicy(),
        "RiskNeutral-DQN":        rn_agent,
        "RiskAverse-QR-DQN":      cvar_agent,
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, policy in policies.items():
        print(f"    Evaluating '{name}' ({n_episodes} episodes) …", flush=True)
        results[name] = evaluate_policy(policy, env, n_episodes=n_episodes, seed=seed)

    return results


# ---------------------------------------------------------------------------
# Figure 3 — Policy comparison bar chart
# ---------------------------------------------------------------------------

def fig3_policy_comparison(
    results_dir: Path,
    policy_results: Dict[str, Dict[str, Any]],
) -> bool:
    names  = list(policy_results.keys())
    costs  = [policy_results[n]["mean_total_cost"]  for n in names]
    stds   = [policy_results[n]["std_total_cost"]   for n in names]
    catast = [policy_results[n]["catastrophe_rate"] * 100 for n in names]

    colors = ["#aaaaaa", "#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(names))
    short = [n.split("(")[0].strip() for n in names]

    bars1 = ax1.bar(x, costs, yerr=stds, capsize=4, color=colors[:len(names)], alpha=0.85)
    best_cost_idx = int(np.argmin(costs))
    bars1[best_cost_idx].set_edgecolor("black")
    bars1[best_cost_idx].set_linewidth(2.0)
    ax1.set_xticks(x); ax1.set_xticklabels(short, rotation=20, ha="right")
    ax1.set_ylabel("Mean cost per episode")
    ax1.set_title("Maintenance Cost (lower = better)")
    ax1.grid(axis="y", alpha=0.25)

    bars2 = ax2.bar(x, catast, color=colors[:len(names)], alpha=0.85)
    best_cat_idx = int(np.argmin(catast))
    bars2[best_cat_idx].set_edgecolor("black")
    bars2[best_cat_idx].set_linewidth(2.0)
    ax2.set_xticks(x); ax2.set_xticklabels(short, rotation=20, ha="right")
    ax2.set_ylabel("Catastrophe rate (%)")
    ax2.set_title("Catastrophic Failures (lower = better)")
    ax2.grid(axis="y", alpha=0.25)

    out = results_dir / "fig3_policy_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig3] saved ->{out.name}")
    return True


# ---------------------------------------------------------------------------
# Figure 4 — Action composition stacked horizontal bar
# ---------------------------------------------------------------------------

def fig4_action_composition(
    results_dir:    Path,
    policy_results: Dict[str, Dict[str, Any]],
) -> bool:
    names = list(policy_results.keys())
    dn    = np.array([policy_results[n]["action_distribution"]["do_nothing"] for n in names])
    rp    = np.array([policy_results[n]["action_distribution"]["repair"]     for n in names])
    rx    = np.array([policy_results[n]["action_distribution"]["replace"]    for n in names])

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(names))
    short = [n.split("(")[0].strip() for n in names]

    ax.barh(y, dn, color="#aaaaaa", label="Do nothing")
    ax.barh(y, rp, left=dn,           color="#4c72b0", label="Repair")
    ax.barh(y, rx, left=dn + rp,      color="#c44e52", label="Replace")

    ax.set_yticks(y); ax.set_yticklabels(short)
    ax.set_xlabel("Action fraction")
    ax.set_xlim(0, 1)
    ax.set_title("Action Composition per Policy")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)

    # Zoomed inset — repair + replace only (0-15% range)
    axins = ax.inset_axes([0.40, 0.50, 0.58, 0.46])
    axins.barh(y, rp, color="#4c72b0", label="Repair")
    axins.barh(y, rx, left=rp, color="#c44e52", label="Replace")
    axins.set_xlim(0, 0.15)
    axins.set_yticks(y)
    axins.set_yticklabels(short, fontsize=7)
    axins.set_xticks([0, 0.05, 0.10, 0.15])
    axins.set_xticklabels(["0%", "5%", "10%", "15%"], fontsize=7)
    axins.set_title("Repair & Replace (0–15% zoom)", fontsize=8)
    axins.grid(axis="x", alpha=0.25)
    # Annotate exact values
    for i, (r, x_) in enumerate(zip(rp, rx)):
        if r > 0.0005:
            axins.text(r / 2, i, f"{r:.2%}", ha="center", va="center",
                       fontsize=6, color="white", fontweight="bold")
        if x_ > 0.0005:
            axins.text(r + x_ / 2, i, f"{x_:.2%}", ha="center", va="center",
                       fontsize=6, color="white", fontweight="bold")

    out = results_dir / "fig4_action_composition.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig4] saved ->{out.name}")
    return True


# ---------------------------------------------------------------------------
# Figure 5 — Training curve
# ---------------------------------------------------------------------------

def fig5_training_curve(results_dir: Path) -> bool:
    csv_path = results_dir / "training_log.csv"
    if not csv_path.exists():
        print(f"  [Fig5] SKIP — {csv_path.name} not found")
        return False

    eps_all: List[int]   = []
    rew_all: List[float] = []
    eval_ep: List[int]   = []
    eval_cr: List[float] = []

    with open(csv_path, encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                ep_raw = row.get("episode", "")
                if not ep_raw:
                    continue
                ep = int(float(ep_raw))  # handles "400" or "400.0"
            except (ValueError, TypeError):
                continue  # skip partial/malformed lines written mid-training
            phase = row.get("phase", "train")
            if phase == "eval":
                ec = row.get("eval_catast", "")
                if ec:
                    try:
                        ec_val = float(ec)
                        # CSV may store as fraction (0.40) or legacy percentage (40.0)
                        eval_ep.append(ep)
                        eval_cr.append(ec_val if ec_val > 1.0 else ec_val * 100)
                    except ValueError:
                        pass
            else:
                r = row.get("ep_reward", "")
                if r:
                    try:
                        eps_all.append(ep)
                        rew_all.append(float(r))
                    except ValueError:
                        pass

    if not eps_all:
        print("  [Fig5] SKIP — no training rows found in CSV")
        return False

    eps_arr = np.array(eps_all)
    rew_arr = np.array(rew_all)
    W = 100
    smooth = np.convolve(rew_arr, np.ones(W) / W, mode="same")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.plot(eps_arr, rew_arr, color="steelblue", alpha=0.25, linewidth=0.7, label="Episode reward")
    ax1.plot(eps_arr, smooth,  color="steelblue", linewidth=1.8, label=f"Smoothed (w={W})")
    if eval_ep:
        eval_y = [smooth[max(0, np.searchsorted(eps_arr, e) - 1)] for e in eval_ep]
        ax1.scatter(eval_ep, eval_y,
                    marker="D", s=40, color="orange", zorder=5, label="Eval checkpoint")
    ax1.set_ylabel("Episode reward")
    ax1.set_title("QR-DQN Training Curve")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)

    if eval_ep:
        ax2.plot(eval_ep, eval_cr, marker="D", ms=5, color="coral", label="Eval catastrophe %")
        ax2.axhline(10, ls="--", color="grey", alpha=0.5, label="10% target")
        ax2.set_ylabel("Catastrophe rate (%)"); ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
    ax2.set_xlabel("Episode")

    out = results_dir / "fig5_training_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig5] saved ->{out.name}")
    return True


# ---------------------------------------------------------------------------
# LSTM and CNN baselines for Table 1
# ---------------------------------------------------------------------------

class _LSTMBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=N_FEATURES, hidden_size=64, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class _CNNBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = N_FEATURES
        for _ in range(3):
            layers += [nn.Conv1d(in_ch, 64, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = 64
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.permute(0, 2, 1)).mean(dim=-1)
        return self.head(h).squeeze(-1)


def _train_simple_model(
    model:     nn.Module,
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    X_val:     np.ndarray,
    y_val:     np.ndarray,
    n_epochs:  int = 150,
    batch_size: int = 64,
    device:    str = "cpu",
    name:      str = "model",
) -> nn.Module:
    dev = torch.device(device)
    model = model.to(dev)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train / MAX_RUL, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state: Any = None
    pat_count = 0

    for ep in range(1, n_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()

        if ep % 10 == 0 or ep == n_epochs:
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32).to(dev)
                yv_hat = model(Xv).cpu().numpy() * MAX_RUL
            val_rmse = float(np.sqrt(np.mean((yv_hat - y_val) ** 2)))
            if val_rmse < best_val:
                best_val = val_rmse
                import copy; best_state = copy.deepcopy(model.state_dict())
                pat_count = 0
            else:
                pat_count += 1
            if ep % 50 == 0:
                print(f"    {name} ep {ep}/{n_epochs}  val_RMSE={val_rmse:.3f}  best={best_val:.3f}")
            if pat_count >= 3:   # 30 epochs patience (checked every 10)
                print(f"    {name} early-stop at ep {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


# ---------------------------------------------------------------------------
# Table 1 — RUL model comparison
# ---------------------------------------------------------------------------

def table1_rul_comparison(
    results_dir:   Path,
    proc_dir:      Path,
    train_baseline: bool = True,
    dry_run:        bool = False,
) -> bool:
    ckpt       = results_dir / "rul_model_best.pth"
    feat_32    = proc_dir / "3_2_features.npy"
    rul_32     = proc_dir / "3_2_rul.npy"

    missing = [p for p in (feat_32, rul_32) if not p.exists()]
    if missing:
        print(f"  [Table1] SKIP — missing: {[str(p.name) for p in missing]}")
        return False

    from src.data_loader import TRAIN_BEARINGS
    W = 32

    # Build train windows from training bearings
    X_parts, y_parts = [], []
    for bid in TRAIN_BEARINGS:
        fp = proc_dir / f"{bid}_features.npy"
        rp = proc_dir / f"{bid}_rul.npy"
        if not fp.exists() or not rp.exists():
            continue
        feat = np.load(fp).astype(np.float32)
        rul  = np.load(rp).astype(np.float32)
        if rul.max() <= 1.05:
            rul = rul * MAX_RUL
        X_w, y_w = create_sliding_windows(feat, rul, window_size=W, stride=1)
        X_parts.append(X_w); y_parts.append(y_w)

    if not X_parts:
        print("  [Table1] SKIP — no training bearing feature files found")
        return False

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)

    feat_val = np.load(feat_32).astype(np.float32)
    rul_val  = np.load(rul_32).astype(np.float32)
    if rul_val.max() <= 1.05:
        rul_val = rul_val * MAX_RUL
    X_val, y_val = create_sliding_windows(feat_val, rul_val, window_size=W, stride=1)

    M_val = len(X_val)
    late_start = int(M_val * 0.80)

    rows: Dict[str, Dict[str, float]] = {}

    # --- Conv-SA (main model) ---
    if ckpt.exists():
        model_sa = ConvSARULPredictor()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model_sa.load_state_dict(state.get("model_state_dict", state))
        model_sa.eval()
        _ep  = state.get("epoch", "?")
        _fr  = state.get("val_rmse_full", float("nan"))
        _lr  = state.get("val_rmse_late", float("nan"))
        print(f"  [Table1] Conv-SA checkpoint: epoch={_ep}  stored full-RMSE={_fr:.2f}  late-RMSE={_lr:.2f}")
        mc_means, mc_var, _ = mc_dropout_inference(model_sa, X_val, n_samples=50)
        full_rmse = float(np.sqrt(np.mean((mc_means - y_val) ** 2)))
        late_rmse = float(np.sqrt(np.mean((mc_means[late_start:] - y_val[late_start:]) ** 2)))
        full_mae  = float(np.mean(np.abs(mc_means - y_val)))
        ep_unc    = float(np.mean(np.sqrt(np.maximum(mc_var, 0.0))))
        rows["Conv-SA (ours)"] = {
            "Full-RMSE": full_rmse, "Late-RMSE": late_rmse,
            "MAE": full_mae, "Epist-Unc": ep_unc,
        }
    else:
        print("  [Table1] Conv-SA: no checkpoint — column will be empty")
        rows["Conv-SA (ours)"] = {"Full-RMSE": float("nan"), "Late-RMSE": float("nan"),
                                   "MAE": float("nan"), "Epist-Unc": float("nan")}

    # --- Baseline models ---
    if train_baseline and not dry_run:
        device = get_device(verbose=False)
        print(f"  [Table1] Training LSTM baseline (150 epochs, device={device}) …")
        lstm_m = _train_simple_model(
            _LSTMBaseline(), X_train, y_train, X_val, y_val,
            n_epochs=150, device=device, name="LSTM",
        )
        lstm_m.eval()
        with torch.no_grad():
            Xv_t = torch.tensor(X_val, dtype=torch.float32)
            lstm_pred = lstm_m(Xv_t).numpy() * MAX_RUL
        rows["LSTM-baseline"] = {
            "Full-RMSE": float(np.sqrt(np.mean((lstm_pred - y_val) ** 2))),
            "Late-RMSE": float(np.sqrt(np.mean((lstm_pred[late_start:] - y_val[late_start:]) ** 2))),
            "MAE":       float(np.mean(np.abs(lstm_pred - y_val))),
            "Epist-Unc": float("nan"),
        }

        print(f"  [Table1] Training CNN baseline (150 epochs, device={device}) …")
        cnn_m = _train_simple_model(
            _CNNBaseline(), X_train, y_train, X_val, y_val,
            n_epochs=150, device=device, name="CNN",
        )
        cnn_m.eval()
        with torch.no_grad():
            cnn_pred = cnn_m(Xv_t).numpy() * MAX_RUL
        rows["CNN-baseline"] = {
            "Full-RMSE": float(np.sqrt(np.mean((cnn_pred - y_val) ** 2))),
            "Late-RMSE": float(np.sqrt(np.mean((cnn_pred[late_start:] - y_val[late_start:]) ** 2))),
            "MAE":       float(np.mean(np.abs(cnn_pred - y_val))),
            "Epist-Unc": float("nan"),
        }
    else:
        print("  [Table1] Baseline training skipped (use --no-baseline-train=False to enable)")

    if len(rows) < 1:
        return False

    cols = ["Full-RMSE", "Late-RMSE", "MAE", "Epist-Unc"]

    # Find best (min) per column (ignoring nan)
    def _best(col: str) -> float:
        vals = [r[col] for r in rows.values() if not math.isnan(r[col])]
        return min(vals) if vals else float("nan")

    best = {c: _best(c) for c in cols}

    def _fmt(v: float, best_v: float, bold: bool) -> str:
        if math.isnan(v):
            return "—"
        s = f"{v:.2f}"
        return f"\\textbf{{{s}}}" if (bold and abs(v - best_v) < 0.005) else s

    # LaTeX table
    tex_path = results_dir / "table1_rul.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{RUL prediction comparison on PRONOSTIA Bearing 3\\_2}\n")
        f.write("\\label{tab:rul_comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & Full-RMSE & Late-RMSE & MAE & Epist-Unc \\\\\n\\midrule\n")
        for name, vals in rows.items():
            cells = [_fmt(vals[c], best[c], True) for c in cols]
            f.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # CSV
    csv_path = results_dir / "table1_rul.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + cols)
        w.writeheader()
        for name, vals in rows.items():
            w.writerow({"method": name, **{c: round(vals[c], 4) for c in cols}})

    print(f"  [Table1] saved ->{tex_path.name}, {csv_path.name}")
    for name, vals in rows.items():
        parts = " | ".join(f"{c}={vals[c]:.2f}" for c in cols if not math.isnan(vals[c]))
        print(f"    {name:25s}  {parts}")
    return True


# ---------------------------------------------------------------------------
# Table 2 — Policy comparison
# ---------------------------------------------------------------------------

def table2_policy_comparison(
    results_dir:    Path,
    policy_results: Dict[str, Dict[str, Any]],
) -> bool:
    cols = ["Cost_mean±std", "Catastrophe%", "TTR_mean", "Repairs/ep", "Replaces/ep", "DoNothing%"]

    def _best_col(key: str, prs: Dict[str, Dict[str, Any]]) -> str:
        vals = {n: prs[n][key] for n in prs}
        return min(vals, key=vals.get)

    best_cost   = _best_col("mean_total_cost",  policy_results)
    best_catast = _best_col("catastrophe_rate",  policy_results)
    best_ttr    = _best_col("mean_TTR",          policy_results)  # lower = safer (earlier)

    def _ttr(v: float) -> str:
        return "—" if math.isnan(v) else f"{v:.1f}"

    def _bf(s: str, is_best: bool) -> str:
        return f"\\textbf{{{s}}}" if is_best else s

    tex_path = results_dir / "table2_policy.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Maintenance policy comparison over 300 evaluation episodes}\n")
        f.write("\\label{tab:policy_comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Policy & Cost$_{\\mu \\pm \\sigma}$ & Catast\\% & TTR & "
                "Repairs/ep & Replaces/ep & DoNothing\\% \\\\\n\\midrule\n")

        for name, m in policy_results.items():
            act  = m.get("action_distribution", {})
            cost = f"{m['mean_total_cost']:.1f}±{m['std_total_cost']:.1f}"
            cat  = f"{m['catastrophe_rate']*100:.1f}"
            ttr  = _ttr(m.get("mean_TTR", float("nan")))
            rep  = f"{m.get('mean_n_repairs', 0):.2f}"
            rx   = f"{m.get('mean_n_replacements', 0):.2f}"
            dn   = f"{act.get('do_nothing', 0)*100:.1f}"

            cost = _bf(cost, name == best_cost)
            cat  = _bf(cat,  name == best_catast)
            f.write(f"{name} & {cost} & {cat} & {ttr} & {rep} & {rx} & {dn} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    csv_path = results_dir / "table2_policy.csv"
    fields = ["policy", "mean_total_cost", "std_total_cost", "catastrophe_rate",
              "mean_TTR", "mean_n_repairs", "mean_n_replacements",
              "dn_pct", "rp_pct", "rx_pct", "mean_episode_length"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, m in policy_results.items():
            act = m.get("action_distribution", {})
            ttr = m.get("mean_TTR", float("nan"))
            w.writerow({
                "policy":              name,
                "mean_total_cost":     round(m["mean_total_cost"], 4),
                "std_total_cost":      round(m["std_total_cost"], 4),
                "catastrophe_rate":    round(m["catastrophe_rate"], 4),
                "mean_TTR":            "" if math.isnan(ttr) else round(ttr, 2),
                "mean_n_repairs":      round(m.get("mean_n_repairs", 0), 4),
                "mean_n_replacements": round(m.get("mean_n_replacements", 0), 4),
                "dn_pct":              round(act.get("do_nothing", 0), 4),
                "rp_pct":              round(act.get("repair",     0), 4),
                "rx_pct":              round(act.get("replace",    0), 4),
                "mean_episode_length": round(m.get("mean_episode_length", 0), 2),
            })

    print(f"  [Table2] saved ->{tex_path.name}, {csv_path.name}")
    for name, m in policy_results.items():
        act = m.get("action_distribution", {})
        print(
            f"    {name:30s}  cost={m['mean_total_cost']:.2f}+-{m['std_total_cost']:.2f}"
            f"  catast={m['catastrophe_rate']:.1%}"
            f"  dn={act.get('do_nothing', 0):.3%}"
            f"  rp={act.get('repair', 0):.3%}"
            f"  rx={act.get('replace', 0):.3%}"
        )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--results-dir",    default="results",       help="Output directory")
    parser.add_argument("--proc-dir",       default="data/processed", help="Processed data dir")
    parser.add_argument("--n-episodes",     type=int, default=300,    help="Episodes per policy")
    parser.add_argument("--n-mc",           type=int, default=50,     help="MC samples for RUL CI")
    parser.add_argument("--dry-run",        action="store_true",      help="Fast checks, skip expensive steps")
    parser.add_argument("--figures-only",   action="store_true",      help="Skip table generation")
    parser.add_argument("--no-baseline-train", action="store_true",   help="Skip LSTM/CNN training")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
    _setup_matplotlib()

    results_dir = (_PROJ / args.results_dir).resolve()
    proc_dir    = (_PROJ / args.proc_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = 2 if args.dry_run else args.n_episodes
    n_mc       = 2 if args.dry_run else args.n_mc

    print(f"\n=== EVALUATE — {'DRY RUN' if args.dry_run else 'FULL'} ===")
    print(f"  results_dir: {results_dir}")
    print(f"  proc_dir:    {proc_dir}")

    completed: List[str] = []
    skipped:   List[str] = []

    # Figures 1–5
    print("\n--- Figures ---")
    if fig1_rul_prediction(results_dir, proc_dir, n_mc=n_mc):
        completed.append("fig1")
    else:
        skipped.append("fig1")

    if fig2_health_index(results_dir, proc_dir):
        completed.append("fig2")
    else:
        skipped.append("fig2")

    if fig5_training_curve(results_dir):
        completed.append("fig5")
    else:
        skipped.append("fig5")

    # Policy comparison (shared by Fig3, Fig4, Table2)
    print("\n--- Policy evaluation ---")
    try:
        policy_results = _collect_policy_results(
            proc_dir, results_dir, n_episodes=n_episodes, seed=42,
        )
        if fig3_policy_comparison(results_dir, policy_results):
            completed.append("fig3")
        if fig4_action_composition(results_dir, policy_results):
            completed.append("fig4")
    except Exception as exc:
        print(f"  Policy evaluation failed: {exc}")
        skipped += ["fig3", "fig4"]
        policy_results = {}

    if not args.figures_only:
        print("\n--- Tables ---")
        train_bl = not (args.no_baseline_train or args.dry_run)
        if table1_rul_comparison(results_dir, proc_dir,
                                  train_baseline=train_bl, dry_run=args.dry_run):
            completed.append("table1")
        else:
            skipped.append("table1")

        if policy_results:
            if table2_policy_comparison(results_dir, policy_results):
                completed.append("table2")
        else:
            skipped.append("table2")

    print(f"\n=== EVALUATE COMPLETE ===")
    print(f"  Completed: {' '.join(completed) if completed else 'none'}")
    if skipped:
        print(f"  Skipped:   {' '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
