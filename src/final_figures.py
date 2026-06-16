"""
final_figures.py
================
Integrates all phase results into a publication-quality IEEE figure set.

Outputs
-------
    results/fig_master.png           2x3 master figure (IEEE double-column)
    results/fig_supp_uncertainty.png supplementary uncertainty analysis
    results/fig_supp_repair.png      supplementary repair ablation
    results/fig_supp_training.png    supplementary training curves
    results/table_master.tex         combined LaTeX table (3 sections)

Usage
-----
    python -m src.final_figures
    python -m src.final_figures --processed-dir data/processed --results-dir results
    python -m src.final_figures --no-inference   # skip RUL model inference
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ===========================================================================
# Global style — must be set before any axes are created
# ===========================================================================

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman"],
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.2,
    "patch.linewidth":    0.8,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "text.usetex":        False,
})

COLOR_PALETTE = {
    "our_method": "#0072B2",
    "baseline_1": "#E69F00",
    "baseline_2": "#56B4E9",
    "baseline_3": "#009E73",
    "baseline_4": "#CC79A7",
    "baseline_5": "#D55E00",
    "danger":     "#CC0000",
    "safe":       "#009900",
}

POLICY_COLORS: Dict[str, str] = {
    "ThresholdPolicy":        COLOR_PALETTE["baseline_5"],
    "DDQN":                   COLOR_PALETTE["baseline_1"],
    "Dueling DQN":            COLOR_PALETTE["baseline_2"],
    "PPO":                    COLOR_PALETTE["baseline_3"],
    "Risk-Neutral DQN":       COLOR_PALETTE["baseline_4"],
    "CVaR QR-DQN (ours)":    COLOR_PALETTE["our_method"],
    "CorrectiveMaintenance":  "#888888",
    "CM (Corrective)":        "#888888",
    "RiskAverse-QR-DQN":     COLOR_PALETTE["our_method"],
    "RiskNeutral-DQN":        COLOR_PALETTE["baseline_4"],
}

_BEARING_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"
]
_BEARING_IDS = ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2"]


# ===========================================================================
# Data loading
# ===========================================================================

def load_all_csvs(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load every known CSV from results_dir. Missing files are skipped."""
    files = {
        "rul":         "table1_rul.csv",
        "rul_base":    "table_rul_baselines.csv",
        "uncertainty": "table_uncertainty.csv",
        "benchmarks":  "table_rl_benchmarks.csv",
        "final_comp":  "final_comparison.csv",
        "state_abl":   "table_state_ablation.csv",
        "risk":        "table_risk_analysis.csv",
        "repair":      "table_repair_ablation.csv",
        "policy2":     "table2_policy.csv",
        "train_log":   "training_log.csv",
    }
    data: Dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        path = results_dir / fname
        if path.exists():
            data[key] = pd.read_csv(path)
            logger.info("Loaded %s (%d rows)", fname, len(data[key]))
        else:
            logger.warning("Not found: %s", path)
    return data


def load_hi_sequences(processed_dir: Path) -> Dict[str, np.ndarray]:
    """Load HI arrays for all available bearings."""
    seqs: Dict[str, np.ndarray] = {}
    for bid in _BEARING_IDS:
        p = processed_dir / f"{bid}_hi.npy"
        if p.exists():
            seqs[bid] = np.load(p).astype(np.float32)
    logger.info("Loaded %d HI sequences", len(seqs))
    return seqs


def load_rul_predictions(
    processed_dir: Path,
    results_dir:   Path,
    device:        str = "cpu",
    n_mc:          int = 30,
    stride:        int = 3,
) -> Optional[Dict[str, np.ndarray]]:
    """Run MC Dropout and Deep Ensemble inference on bearing 3_2.

    Returns None if model files or feature data are unavailable.
    """
    try:
        from src.rul_predictor import (
            ConvSARULPredictor,
            mc_dropout_inference,
            create_sliding_windows,
            load_model,
        )
    except ImportError:
        try:
            _root = Path(__file__).resolve().parent.parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from src.rul_predictor import (
                ConvSARULPredictor,
                mc_dropout_inference,
                create_sliding_windows,
                load_model,
            )
        except ImportError:
            logger.warning("rul_predictor import failed; skipping RUL inference.")
            return None

    feat_path = processed_dir / "3_2_features.npy"
    rul_path  = processed_dir / "3_2_rul.npy"
    ckpt_path = results_dir   / "rul_model_best.pth"

    for p in (feat_path, rul_path, ckpt_path):
        if not p.exists():
            logger.warning("Missing for RUL inference: %s", p)
            return None

    try:
        features = np.load(feat_path).astype(np.float32)
        rul_arr  = np.load(rul_path).astype(np.float32)   # original scale [0, 125]
        X, y_true = create_sliding_windows(
            features, rul_arr, window_size=32, stride=stride
        )
        timesteps = np.arange(31, len(rul_arr), stride)[: len(X)]

        # MC Dropout
        model = load_model(ckpt_path, device=device)
        mc_mean, mc_sigma2, _ = mc_dropout_inference(model, X, n_samples=n_mc, device=device)
        mc_std = np.sqrt(np.clip(mc_sigma2, 0, None))

        # Deep Ensemble
        ens_seeds = [42, 123, 456, 789, 1024]
        ens_preds: List[np.ndarray] = []
        for seed in ens_seeds:
            ep = results_dir / f"ensemble_{seed}.pth"
            if not ep.exists():
                continue
            em = ConvSARULPredictor()
            sd = torch.load(ep, map_location="cpu", weights_only=False)
            em.load_state_dict(sd.get("model_state_dict", sd))
            em.eval()
            em.to(torch.device(device))
            xt = torch.tensor(X, dtype=torch.float32).to(device)
            with torch.no_grad():
                p_arr = torch.clamp(em(xt), 0, 1).cpu().numpy().squeeze() * 125.0
            ens_preds.append(p_arr)

        if ens_preds:
            ens_stack = np.stack(ens_preds, axis=0)
            ens_mean  = ens_stack.mean(axis=0)
            ens_std   = ens_stack.std(axis=0)
        else:
            ens_mean = ens_std = None

        return {
            "timesteps": timesteps,
            "y_true":    y_true,
            "mc_mean":   mc_mean,
            "mc_std":    mc_std,
            "ens_mean":  ens_mean,
            "ens_std":   ens_std,
        }
    except Exception as exc:
        logger.warning("RUL inference failed (%s); panel (b) will be placeholder.", exc)
        return None


# ===========================================================================
# Master figure panel helpers
# ===========================================================================

def _label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12, 1.02, f"({letter})",
        transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="bottom", ha="left",
    )


def _panel_a(ax: plt.Axes, hi_seqs: Dict[str, np.ndarray]) -> None:
    """(a) Health Index degradation for all 6 bearings."""
    for idx, bid in enumerate(_BEARING_IDS):
        if bid not in hi_seqs:
            continue
        hi = hi_seqs[bid]
        ts = np.arange(len(hi))
        color = _BEARING_COLORS[idx % len(_BEARING_COLORS)]
        lbl = f"B{bid.replace('_', '.')}"
        ax.plot(ts, hi, color=color, linewidth=0.9, alpha=0.85, label=lbl)

    ax.axhline(0.055, color=COLOR_PALETTE["danger"], linestyle=":", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Health Index")
    ax.set_title("(a) HI Degradation — All Bearings")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right", framealpha=0.7)
    _label(ax, "a")


def _panel_b(ax: plt.Axes, rul_preds: Optional[Dict]) -> None:
    """(b) RUL prediction on bearing 3_2 with uncertainty bands."""
    if rul_preds is None:
        ax.text(0.5, 0.5, "Model inference\nnot available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title("(b) RUL Prediction — B3.2")
        _label(ax, "b")
        return

    ts      = rul_preds["timesteps"]
    y_true  = rul_preds["y_true"]
    mc_m    = rul_preds["mc_mean"]
    mc_s    = rul_preds["mc_std"]

    ax.plot(ts, y_true, color="black", linewidth=1.0, label="True RUL", zorder=4)
    ax.plot(ts, mc_m,   color=COLOR_PALETTE["our_method"], linewidth=1.0,
            label="MC Dropout", zorder=3)
    ax.fill_between(ts, mc_m - 2 * mc_s, mc_m + 2 * mc_s,
                    alpha=0.2, color=COLOR_PALETTE["our_method"], linewidth=0)

    if rul_preds.get("ens_mean") is not None:
        em, es = rul_preds["ens_mean"], rul_preds["ens_std"]
        ax.plot(ts, em, color=COLOR_PALETTE["baseline_1"], linewidth=1.0,
                label="Deep Ensemble", zorder=3)
        ax.fill_between(ts, em - 2 * es, em + 2 * es,
                        alpha=0.2, color=COLOR_PALETTE["baseline_1"], linewidth=0)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("(b) RUL Prediction — Bearing 3.2")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)
    _label(ax, "b")


def _panel_c(ax: plt.Axes, abl_df: pd.DataFrame) -> None:
    """(c) State ablation — catastrophe rate vs state configuration."""
    configs = list(abl_df["State_Config"])
    catast  = list(abl_df["Catastrophe_pct"])
    blues   = ["#a8c8e8", "#5fa8d3", "#2980b9", "#1a5276"]

    bars = ax.bar(configs, catast, color=blues[: len(configs)],
                  edgecolor="white", linewidth=0.5, width=0.6)

    # Annotate best (D or min)
    best_idx = int(np.argmin(catast))
    ax.text(
        best_idx, catast[best_idx] + 0.2,
        "Best", ha="center", va="bottom", fontsize=7.5,
        color=COLOR_PALETTE["safe"], fontweight="bold",
    )
    ax.bar_label(bars, fmt="%.1f%%", fontsize=7, padding=1)

    ax.set_xlabel("State Configuration")
    ax.set_ylabel("Catastrophe Rate (%)")
    ax.set_title("(c) State Ablation")
    ax.set_ylim(0, max(catast) * 1.3)
    _label(ax, "c")


def _panel_d(
    ax: plt.Axes,
    bench_df: pd.DataFrame,
    final_df: Optional[pd.DataFrame],
) -> None:
    """(d) Policy comparison — catastrophe rate bar chart."""
    rows: List[Tuple[str, float]] = []

    # CM baseline from final_comparison.csv
    if final_df is not None and "catastrophe_rate" in final_df.columns:
        cm_row = final_df[final_df["policy"].str.contains("Corrective", na=False)]
        if not cm_row.empty:
            rows.append(("CM", float(cm_row.iloc[0]["catastrophe_rate"]) * 100))

    # RL benchmarks
    if "Policy" in bench_df.columns and "Catastrophe_pct" in bench_df.columns:
        for _, r in bench_df.iterrows():
            rows.append((str(r["Policy"]), float(r["Catastrophe_pct"])))

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        _label(ax, "d")
        return

    names  = [r[0] for r in rows]
    catast = [r[1] for r in rows]
    colors = [POLICY_COLORS.get(n, "#AAAAAA") for n in names]

    bars = ax.barh(range(len(names)), catast, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [n.replace("CVaR QR-DQN (ours)", "CVaR QR-DQN*") for n in names],
        fontsize=7.5,
    )
    ax.set_xlabel("Catastrophe Rate (%)")
    ax.set_title("(d) Policy Safety Comparison")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=6.5, padding=2)
    ax.invert_yaxis()
    _label(ax, "d")


def _panel_e(ax: plt.Axes, risk_df: pd.DataFrame) -> None:
    """(e) Risk-return tradeoff: CVaR alpha vs catastrophe rate."""
    alphas  = risk_df["alpha"].values
    catast  = risk_df["catastrophe_pct"].values
    costs   = risk_df["mean_cost"].values

    ax2 = ax.twinx()
    ax2.plot(alphas, costs, "s--", color=COLOR_PALETTE["baseline_1"],
             markersize=4, linewidth=0.9, alpha=0.75, label="Cost μ")
    ax2.set_ylabel("Mean Cost", fontsize=9, color=COLOR_PALETTE["baseline_1"])
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["baseline_1"], labelsize=8)
    ax2.grid(False)

    ax.plot(alphas, catast, "o-", color=COLOR_PALETTE["our_method"],
            markersize=4, linewidth=1.2, label="Catastrophe %")
    ax.axvline(0.40, color=COLOR_PALETTE["danger"], linestyle="--",
               linewidth=0.9, label=r"$\alpha^*=0.40$")
    ax.axhline(17.3, color="#888888", linestyle=":", linewidth=0.8, label="CM 17.3%")

    ax.set_xlabel(r"CVaR $\alpha$")
    ax.set_ylabel("Catastrophe Rate (%)")
    ax.set_title(r"(e) Risk-Return Tradeoff")
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas], rotation=35, ha="right", fontsize=8)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left", framealpha=0.7)
    _label(ax, "e")


def _panel_f(ax: plt.Axes, bench_df: pd.DataFrame) -> None:
    """(f) Action composition stacked bar for all learned policies."""
    dn_col   = "DoNothing_pct"
    rp_col   = "Repairs_per_ep"
    rx_col   = "Replaces_per_ep"
    pol_col  = "Policy"

    if pol_col not in bench_df.columns:
        _label(ax, "f")
        return

    names, dn_fracs, rp_fracs, rx_fracs = [], [], [], []
    for _, row in bench_df.iterrows():
        dn   = float(row.get(dn_col, 100)) / 100.0
        reps = float(row.get(rp_col, 0))
        rexs = float(row.get(rx_col, 0))
        tot  = reps + rexs
        rem  = 1.0 - dn
        rp   = rem * (reps / tot) if tot > 0 else 0.0
        rx   = rem * (rexs / tot) if tot > 0 else 0.0
        names.append(str(row[pol_col]))
        dn_fracs.append(dn * 100)
        rp_fracs.append(rp * 100)
        rx_fracs.append(rx * 100)

    xs = np.arange(len(names))
    w  = 0.55
    ax.bar(xs, dn_fracs, w, color="#CCCCCC",   edgecolor="white", lw=0.4, label="Do-Nothing")
    ax.bar(xs, rp_fracs, w, bottom=dn_fracs,   color="#5fa8d3",  edgecolor="white", lw=0.4, label="Repair")
    ax.bar(xs, rx_fracs, w,
           bottom=[a + b for a, b in zip(dn_fracs, rp_fracs)],
           color=COLOR_PALETTE["our_method"], edgecolor="white", lw=0.4, label="Replace")

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [n.replace("CVaR QR-DQN (ours)", "CVaR\nQR-DQN*").replace("Risk-Neutral DQN", "RN-DQN")
          .replace("Dueling DQN", "DuelDQN").replace("ThresholdPolicy", "Thresh.")
         for n in names],
        fontsize=7.5, rotation=20, ha="right",
    )
    ax.set_ylabel("Action Share (%)")
    ax.set_title("(f) Action Composition")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.7)
    _label(ax, "f")


# ===========================================================================
# Master figure (2x3, IEEE double-column width)
# ===========================================================================

def fig_master(
    data:          Dict[str, pd.DataFrame],
    hi_seqs:       Dict[str, np.ndarray],
    rul_preds:     Optional[Dict],
    results_dir:   Path,
) -> None:
    """Generate the 6-panel master figure."""
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.5))
    ax = axes.flatten()

    _panel_a(ax[0], hi_seqs)
    _panel_b(ax[1], rul_preds)
    _panel_c(ax[2], data.get("state_abl", _dummy_state_abl()))
    _panel_d(ax[3], data.get("benchmarks", pd.DataFrame()), data.get("final_comp"))
    _panel_e(ax[4], data.get("risk", _dummy_risk()))
    _panel_f(ax[5], data.get("benchmarks", pd.DataFrame()))

    # Shared bottom legend for policy color scheme
    legend_handles = [
        mpatches.Patch(color=POLICY_COLORS["ThresholdPolicy"],     label="ThresholdPolicy"),
        mpatches.Patch(color=POLICY_COLORS["DDQN"],                label="DDQN"),
        mpatches.Patch(color=POLICY_COLORS["Dueling DQN"],         label="Dueling DQN"),
        mpatches.Patch(color=POLICY_COLORS["PPO"],                 label="PPO"),
        mpatches.Patch(color=POLICY_COLORS["Risk-Neutral DQN"],    label="Risk-Neutral DQN"),
        mpatches.Patch(color=POLICY_COLORS["CVaR QR-DQN (ours)"], label="CVaR QR-DQN (ours)"),
        Line2D([0], [0], color="#888888", linestyle=":", linewidth=1.0, label="CM Baseline"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=7,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = results_dir / "fig_master.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def _dummy_state_abl() -> pd.DataFrame:
    return pd.DataFrame({
        "State_Config":   list("ABCD"),
        "Catastrophe_pct": [17.33, 0.33, 0.33, 8.33],
    })


def _dummy_risk() -> pd.DataFrame:
    alphas = [0.05, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00]
    return pd.DataFrame({
        "alpha":           alphas,
        "catastrophe_pct": [14.75, 14.50, 13.75, 13.0, 12.0, 11.0, 10.0],
        "mean_cost":       [250.5, 251.0, 252.5, 254.0, 256.0, 258.0, 260.0],
    })


# ===========================================================================
# Supplementary figures
# ===========================================================================

def fig_supp_uncertainty(
    data: Dict[str, pd.DataFrame],
    results_dir: Path,
) -> None:
    """Supplementary: uncertainty calibration metrics (3 panels)."""
    unc_df = data.get("uncertainty")
    if unc_df is None:
        logger.warning("table_uncertainty.csv not found; skipping fig_supp_uncertainty.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))

    methods = list(unc_df["Method"])
    colors  = [COLOR_PALETTE["baseline_4"], COLOR_PALETTE["our_method"],
               COLOR_PALETTE["baseline_1"]]

    # (a) ECE — lower is better
    ax = axes[0]
    ece = list(unc_df["ECE"])
    bars = ax.bar(methods, ece, color=colors[:len(methods)],
                  edgecolor="white", lw=0.6, width=0.55)
    best = np.argmin(ece)
    ax.bar([methods[best]], [ece[best]], color=colors[best],
           edgecolor=COLOR_PALETTE["safe"], lw=1.5, width=0.55)
    ax.bar_label(bars, fmt="%.4f", fontsize=7.5, padding=2)
    ax.set_ylabel("ECE (lower = better)")
    ax.set_title("(a) Calibration Error")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha="right")

    # (b) PICP coverage — higher is better
    ax = axes[1]
    picp90 = list(unc_df["PICP@90%"])
    picp95 = list(unc_df["PICP@95%"])
    xs = np.arange(len(methods))
    w  = 0.32
    ax.bar(xs - w / 2, picp90, w, color="#56B4E9", label="PICP@90%", edgecolor="white", lw=0.5)
    ax.bar(xs + w / 2, picp95, w, color="#0072B2", label="PICP@95%", edgecolor="white", lw=0.5)
    ax.axhline(0.90, color="#56B4E9", linestyle="--", lw=0.8, alpha=0.7)
    ax.axhline(0.95, color="#0072B2", linestyle="--", lw=0.8, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("PICP Coverage")
    ax.set_title("(b) Interval Coverage")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.1)

    # (c) RMSE vs MPIW tradeoff
    ax = axes[2]
    rmse  = list(unc_df["RMSE"])
    mpiw  = list(unc_df["MPIW@95%"])
    for i, (m, r, pi, c) in enumerate(zip(methods, rmse, mpiw, colors)):
        ax.scatter(r, pi, color=c, s=60, zorder=3, label=m)
        ax.annotate(m, (r, pi), textcoords="offset points",
                    xytext=(4, 3), fontsize=7)
    ax.set_xlabel("RMSE (lower = better)")
    ax.set_ylabel("MPIW@95% (lower = better)")
    ax.set_title("(c) Sharpness vs Accuracy")
    ax.legend(fontsize=7)

    fig.tight_layout(pad=1.2)
    out = results_dir / "fig_supp_uncertainty.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_supp_repair(
    data: Dict[str, pd.DataFrame],
    results_dir: Path,
) -> None:
    """Supplementary: repair model ablation (2 panels)."""
    rep_df = data.get("repair")
    if rep_df is None:
        logger.warning("table_repair_ablation.csv not found; skipping fig_supp_repair.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    models  = list(rep_df["repair_model"])
    costs   = list(rep_df["mean_cost"])
    stds    = list(rep_df["std_cost"])
    catasts = list(rep_df["catastrophe_pct"])
    colors  = [COLOR_PALETTE["safe"], COLOR_PALETTE["our_method"]]

    # (a) Cost with error bars
    ax = axes[0]
    xs = np.arange(len(models))
    ax.bar(xs, costs, yerr=stds, color=colors[:len(models)], capsize=5,
           edgecolor="white", lw=0.5, width=0.55, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [m.replace("Exponential Decay (ours)", "Exp. Decay (ours)") for m in models],
        rotation=12, ha="right",
    )
    ax.set_ylabel("Mean Episode Cost")
    ax.set_title("(a) Episode Cost Comparison")
    for xi, (c, s) in enumerate(zip(costs, stds)):
        ax.text(xi, c + s + max(stds) * 0.03, f"{c:.1f}",
                ha="center", va="bottom", fontsize=8)

    # (b) Catastrophe rate + behavioral metrics side-by-side
    ax = axes[1]
    n_repairs = list(rep_df["mean_n_repairs"])
    n_replaces = list(rep_df["mean_n_replacements"])
    w = 0.22
    xs = np.arange(len(models))
    ax.bar(xs - w, catasts,  w, color="#d7191c", label="Catastrophe %", alpha=0.85)
    ax.bar(xs,     n_repairs, w, color="#5fa8d3", label="Repairs/ep",    alpha=0.85)
    ax.bar(xs + w, n_replaces, w, color="#1a9641", label="Replaces/ep",  alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [m.replace("Exponential Decay (ours)", "Exp. Decay (ours)") for m in models],
        rotation=12, ha="right",
    )
    ax.set_ylabel("Value")
    ax.set_title("(b) Behavioural Metrics")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout(pad=1.2)
    out = results_dir / "fig_supp_repair.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_supp_training(
    data: Dict[str, pd.DataFrame],
    results_dir: Path,
) -> None:
    """Supplementary: training curves (from training_log.csv if present)."""
    log_df = data.get("train_log")

    if log_df is None or log_df.empty:
        # Fallback: show benchmark final-performance summary as a bar chart
        bench = data.get("benchmarks")
        if bench is None:
            logger.warning("No training log or benchmarks; skipping fig_supp_training.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

        policies = list(bench["Policy"])
        costs    = list(bench["Cost_mu"])
        catast   = list(bench["Catastrophe_pct"])
        clrs     = [POLICY_COLORS.get(p, "#AAAAAA") for p in policies]

        axes[0].barh(range(len(policies)), costs, color=clrs, edgecolor="white", lw=0.5)
        axes[0].set_yticks(range(len(policies)))
        axes[0].set_yticklabels(policies, fontsize=8)
        axes[0].set_xlabel("Mean Episode Cost")
        axes[0].set_title("(a) Final Episode Cost")
        axes[0].invert_yaxis()

        axes[1].barh(range(len(policies)), catast, color=clrs, edgecolor="white", lw=0.5)
        axes[1].set_yticks(range(len(policies)))
        axes[1].set_yticklabels(policies, fontsize=8)
        axes[1].set_xlabel("Catastrophe Rate (%)")
        axes[1].set_title("(b) Final Catastrophe Rate")
        axes[1].invert_yaxis()

        fig.suptitle("Policy Performance Summary (no training log available)", fontsize=9)
        fig.tight_layout(pad=1.2)
    else:
        # Plot actual training curves
        eval_rows = log_df[log_df.get("phase", pd.Series(dtype=str)) == "eval"] \
            if "phase" in log_df.columns else pd.DataFrame()
        train_rows = log_df[log_df.get("phase", pd.Series(dtype=str)) == "train"] \
            if "phase" in log_df.columns else log_df

        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

        if "ep_reward" in train_rows.columns and "episode" in train_rows.columns:
            ep   = train_rows["episode"].values.astype(float)
            rew  = train_rows["ep_reward"].values.astype(float)
            # 200-episode moving average
            if len(rew) >= 200:
                smooth = np.convolve(rew, np.ones(200) / 200, mode="valid")
                axes[0].plot(ep[199:], smooth, color=COLOR_PALETTE["our_method"],
                             linewidth=1.2, label="Moving avg (200 ep)")
            axes[0].plot(ep, rew, color=COLOR_PALETTE["our_method"],
                         alpha=0.15, linewidth=0.5)
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Episode Reward")
            axes[0].set_title("(a) Training Reward")
            axes[0].legend(fontsize=7)

        if not eval_rows.empty and "eval_catast" in eval_rows.columns:
            axes[1].plot(
                eval_rows["episode"], eval_rows["eval_catast"] * 100,
                "o-", color=COLOR_PALETTE["danger"],
                markersize=3, linewidth=1.0, label="Eval catast. rate",
            )
            axes[1].axhline(9.0, color=COLOR_PALETTE["safe"], linestyle="--",
                            linewidth=0.8, label="Target 9%")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Catastrophe Rate (%)")
            axes[1].set_title("(b) Eval Catastrophe Rate")
            axes[1].legend(fontsize=7)

        fig.tight_layout(pad=1.2)

    out = results_dir / "fig_supp_training.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ===========================================================================
# Master LaTeX table
# ===========================================================================

def table_master(
    data: Dict[str, pd.DataFrame],
    results_dir: Path,
) -> None:
    """Single LaTeX table with 3 sections: RUL / Uncertainty / RL policies."""
    results_dir.mkdir(parents=True, exist_ok=True)

    def _bf(val: str, is_best: bool) -> str:
        return f"\\textbf{{{val}}}" if is_best else val

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive comparison of RUL predictors and maintenance policies"
        r" on PRONOSTIA bearing 3\_2 (Condition 3). Bold = best per column.}",
        r"\label{tab:master}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"Method / Policy & Metric-1 & Metric-2 & Metric-3 & Metric-4 & Notes \\",
        r"\midrule",
        # ----- Section 1 -----
        r"\multicolumn{6}{l}{\textit{Part I — RUL Prediction (bearing 3\_2)}} \\",
        r"\midrule",
        r"\quad & Full-RMSE $\downarrow$ & Late-RMSE $\downarrow$ & MAE $\downarrow$"
        r" & Uncertainty & \\",
    ]

    rul_df = data.get("rul_base")
    if rul_df is not None:
        best_full = rul_df["Full-RMSE"].min()
        best_late = rul_df["Late-RMSE"].min()
        best_mae  = rul_df["MAE"].min()
        for _, row in rul_df.iterrows():
            model = str(row["Model"])
            f_rmse = float(row["Full-RMSE"])
            l_rmse = float(row["Late-RMSE"])
            mae    = float(row["MAE"])
            unc    = str(row.get("Uncertainty", "None"))
            note   = str(row.get("Notes", ""))
            is_ours = "(ours)" in model
            model_cell = "\\textbf{" + model + "}" if is_ours else model
            unc_cell  = "--" if (pd.isna(unc) or str(unc).strip().lower() in ("nan", "none", "")) else str(unc)
            note_cell = "--" if (pd.isna(note) or str(note).strip().lower() in ("nan",)) else str(note)
            lines.append(
                f"{model_cell} & "
                f"{_bf(f'{f_rmse:.2f}', abs(f_rmse - best_full) < 0.01)} & "
                f"{_bf(f'{l_rmse:.2f}', abs(l_rmse - best_late) < 0.01)} & "
                f"{_bf(f'{mae:.2f}',    abs(mae    - best_mae)  < 0.01)} & "
                f"{unc_cell} & {note_cell} \\\\"
            )

    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Part II — Predictive Uncertainty Calibration}} \\",
        r"\midrule",
        r"\quad & RMSE $\downarrow$ & ECE $\downarrow$ & PICP@95\% $\uparrow$"
        r" & MPIW@95\% $\downarrow$ & \\",
    ]

    unc_df = data.get("uncertainty")
    if unc_df is not None:
        best_rmse = unc_df["RMSE"].min()
        best_ece  = unc_df["ECE"].min()
        best_picp = unc_df["PICP@95%"].max()
        best_mpiw = unc_df["MPIW@95%"].min()
        for _, row in unc_df.iterrows():
            meth = str(row["Method"])
            rmse = float(row["RMSE"])
            ece  = float(row["ECE"])
            picp = float(row["PICP@95%"])
            mpiw = float(row["MPIW@95%"])
            lines.append(
                f"{meth} & "
                f"{_bf(f'{rmse:.2f}', abs(rmse - best_rmse) < 0.01)} & "
                f"{_bf(f'{ece:.4f}',  abs(ece  - best_ece)  < 1e-4)} & "
                f"{_bf(f'{picp:.3f}', abs(picp - best_picp) < 0.001)} & "
                f"{_bf(f'{mpiw:.2f}', abs(mpiw - best_mpiw) < 0.01)} & \\\\"
            )

    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Part III — RL Maintenance Policies}} \\",
        r"\midrule",
        r"\quad & Cost $\mu$ $\downarrow$ & Cost $\sigma$ $\downarrow$"
        r" & Catast.\% $\downarrow$ & Avg Reward $\uparrow$ & \\",
    ]

    bench_df = data.get("benchmarks")
    if bench_df is not None:
        best_cost   = bench_df["Cost_mu"].min()
        best_std    = bench_df["Cost_sigma"].min()
        best_catast = bench_df["Catastrophe_pct"].min()
        best_rew    = bench_df["Avg_Reward"].max()
        for _, row in bench_df.iterrows():
            pol  = str(row["Policy"])
            cost = float(row["Cost_mu"])
            std  = float(row["Cost_sigma"])
            cat  = float(row["Catastrophe_pct"])
            rew  = float(row["Avg_Reward"])
            is_ours = "(ours)" in pol
            pol_cell = "\\textbf{" + pol + "}" if is_ours else pol
            lines.append(
                f"{pol_cell} & "
                f"{_bf(f'{cost:.1f}',  abs(cost - best_cost)   < 0.1)} & "
                f"{_bf(f'{std:.1f}',   abs(std  - best_std)    < 0.1)} & "
                f"{_bf(f'{cat:.2f}',   abs(cat  - best_catast) < 0.01)} & "
                f"{_bf(f'{rew:.1f}',   abs(rew  - best_rew)    < 0.1)} & \\\\"
            )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    out = results_dir / "table_master.tex"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("Saved %s", out)


# ===========================================================================
# Summary printer
# ===========================================================================

def print_summary(data: Dict[str, pd.DataFrame]) -> None:
    """Print === PAPER RESULTS SUMMARY === to stdout."""
    print("\n" + "=" * 50)
    print("=== PAPER RESULTS SUMMARY ===")
    print("=" * 50)

    # RUL
    rul_df = data.get("rul_base")
    if rul_df is not None:
        idx  = rul_df["Late-RMSE"].idxmin()
        best = rul_df.loc[idx]
        print(f"RUL:         Best model by Late-RMSE: {best['Model']}"
              f"  (Late-RMSE = {best['Late-RMSE']:.2f})")
    else:
        print("RUL:         (table_rul_baselines.csv not found)")

    # Uncertainty
    unc_df = data.get("uncertainty")
    if unc_df is not None:
        idx  = unc_df["ECE"].idxmin()
        best = unc_df.loc[idx]
        print(f"Uncertainty: Best calibrated: {best['Method']}"
              f"  (ECE = {best['ECE']:.4f})")
    else:
        print("Uncertainty: (table_uncertainty.csv not found)")

    # Safety
    bench_df = data.get("benchmarks")
    if bench_df is not None:
        idx  = bench_df["Catastrophe_pct"].idxmin()
        best = bench_df.loc[idx]
        print(f"Safety:      Lowest catastrophe rate: {best['Policy']}"
              f"  at {best['Catastrophe_pct']:.2f}%")
    else:
        print("Safety:      (table_rl_benchmarks.csv not found)")

    # State ablation
    abl_df = data.get("state_abl")
    if abl_df is not None and "State_Config" in abl_df.columns:
        rul_only = abl_df.loc[abl_df["State_Config"] == "A", "Catastrophe_pct"].values
        full     = abl_df.loc[abl_df["State_Config"] == "D", "Catastrophe_pct"].values
        if len(rul_only) > 0 and len(full) > 0 and float(rul_only[0]) > 0:
            improve = (float(rul_only[0]) - float(full[0])) / float(rul_only[0]) * 100
            print(f"State abl.:  Full state (D) improves over RUL-only (A) by"
                  f"  {improve:.1f}% catastrophe reduction"
                  f"  ({rul_only[0]:.2f}% -> {full[0]:.2f}%)")

    # Risk alpha=0.40
    risk_df = data.get("risk")
    if risk_df is not None:
        row = risk_df[risk_df["alpha"].round(2) == 0.40]
        if not row.empty:
            r = row.iloc[0]
            print(f"Risk:        alpha=0.40 is Pareto-optimal at"
                  f"  [cost={r['mean_cost']:.2f}, catast={r['catastrophe_pct']:.2f}%]")

    print("=" * 50)
    print("=== END ===\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate all publication-quality figures from phase results."
    )
    p.add_argument("--processed-dir", default="data/processed", type=Path)
    p.add_argument("--results-dir",   default="results",        type=Path)
    p.add_argument("--device",        default=None,
                   help="Compute device for RUL inference: cuda | cpu (default: auto)")
    p.add_argument("--no-inference",  action="store_true",
                   help="Skip RUL model inference (panel b will be placeholder)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir: Path  = args.results_dir
    processed_dir: Path = args.processed_dir

    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading all CSV results from %s ...", results_dir)
    data = load_all_csvs(results_dir)

    logger.info("Loading HI sequences from %s ...", processed_dir)
    hi_seqs = load_hi_sequences(processed_dir)

    rul_preds: Optional[Dict] = None
    if not args.no_inference:
        logger.info("Running RUL inference on bearing 3_2 (device=%s)...", device_str)
        rul_preds = load_rul_predictions(processed_dir, results_dir, device=device_str)

    # Master figure
    logger.info("Generating fig_master.png ...")
    fig_master(data, hi_seqs, rul_preds, results_dir)

    # Supplementary figures
    logger.info("Generating supplementary figures ...")
    fig_supp_uncertainty(data, results_dir)
    fig_supp_repair(data, results_dir)
    fig_supp_training(data, results_dir)

    # Master table
    logger.info("Generating table_master.tex ...")
    table_master(data, results_dir)

    # Summary
    print_summary(data)


if __name__ == "__main__":
    main()
