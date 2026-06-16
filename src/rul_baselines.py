"""
rul_baselines.py
================
Three additional RUL predictors compared against Conv-SA in a shared evaluation.

Models
------
1. XGBoostRUL       -- flattened-window XGBRegressor (1024-D input)
2. GRURULPredictor  -- 2-layer GRU with linear head
3. TCNRULPredictor  -- 4-block dilated causal TCN

All three are point-estimate models (no uncertainty quantification).
Existing Conv-SA, CNN-baseline, and LSTM-baseline metrics are loaded from
results/table1_rul.csv when available.

Usage
-----
    python src/rul_baselines.py
"""

from __future__ import annotations

import copy
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from src.device import get_device
    from src.rul_predictor import MAX_RUL, N_FEATURES, create_sliding_windows
except ImportError:
    from device import get_device                           # type: ignore[no-redef]
    from rul_predictor import MAX_RUL, N_FEATURES, create_sliding_windows  # type: ignore[no-redef]

try:
    from xgboost import XGBRegressor as _XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGBRegressor  = None                                  # type: ignore[assignment, misc]
    _XGB_AVAILABLE = False

logger = logging.getLogger(__name__)

TRAIN_BEARINGS: List[str] = ["1_1", "1_2", "2_1", "2_2", "3_1"]
TEST_BEARING:   str       = "3_2"
WINDOW_SIZE:    int       = 32


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rul_model(
    predictions: np.ndarray,
    true_rul:    np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE / MAE on test-bearing windows.

    Parameters
    ----------
    predictions, true_rul : (M,) arrays in [0, 125]

    Returns
    -------
    dict with keys ``rmse_full``, ``rmse_late``, ``mae_full``
    """
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(true_rul,    dtype=np.float64)
    M    = len(true)
    late = int(M * 0.80)
    err  = pred - true
    return {
        "rmse_full": float(np.sqrt(np.mean(err ** 2))),
        "rmse_late": float(np.sqrt(np.mean(err[late:] ** 2))),
        "mae_full":  float(np.mean(np.abs(err))),
    }


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
            logger.warning("Bearing %s files missing — skipped.", bid)
            continue
        feat = np.load(fp).astype(np.float32)
        rul  = np.load(rp).astype(np.float32)
        X, y = create_sliding_windows(feat, rul, window_size=WINDOW_SIZE, stride=1)
        Xs.append(X)
        ys.append(y)
    if not Xs:
        raise FileNotFoundError(f"No feature/rul files found in {proc_dir}")
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _train_val_split(
    X:        np.ndarray,
    y:        np.ndarray,
    val_frac: float = 0.10,
    seed:     int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X))
    split = int(len(X) * (1.0 - val_frac))
    tr, va = idx[:split], idx[split:]
    return X[tr], y[tr], X[va], y[va]


# ---------------------------------------------------------------------------
# Model 1 — XGBoost
# ---------------------------------------------------------------------------

class XGBoostRUL:
    """Flatten each (32 x 32) window to 1024-D and fit an XGBRegressor."""

    def __init__(self) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        self.model = _XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRUL":
        """X: (M, 32, 32)  y: (M,) in [0, 125]."""
        self.model.fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(
            self.model.predict(X.reshape(len(X), -1)), 0.0, MAX_RUL
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Model 2 — GRU
# ---------------------------------------------------------------------------

class GRURULPredictor(nn.Module):
    """2-layer GRU -> Linear(128,64) -> ReLU -> Linear(64,1) -> Sigmoid * 125."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=N_FEATURES, hidden_size=128,
            num_layers=2, batch_first=True, dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),   nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=32, F=32)
        _, h_n = self.gru(x)                              # h_n: (2, B, 128)
        return self.head(h_n[-1]).squeeze(-1) * MAX_RUL   # (B,)


# ---------------------------------------------------------------------------
# Model 3 — Temporal Convolutional Network (TCN)
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Dilated causal residual block."""

    def __init__(
        self,
        in_ch:       int,
        out_ch:      int   = 64,
        kernel_size: int   = 3,
        dilation:    int   = 1,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self._left_pad = (kernel_size - 1) * dilation   # causal left-only padding

        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        res = self.skip(x)
        h   = F.pad(x, (self._left_pad, 0))
        h   = self.drop1(F.relu(self.bn1(self.conv1(h))))
        h   = F.pad(h, (self._left_pad, 0))
        h   = self.drop2(F.relu(self.bn2(self.conv2(h))))
        return h + res


class TCNRULPredictor(nn.Module):
    """4 dilated causal TCN blocks -> global avg pool -> Linear(64,1) -> Sigmoid * 125."""

    def __init__(self) -> None:
        super().__init__()
        in_ch:  int              = N_FEATURES
        blocks: List[nn.Module]  = []
        for d in [1, 2, 4, 8]:
            blocks.append(_TCNBlock(in_ch, out_ch=64, kernel_size=3, dilation=d))
            in_ch = 64
        self.tcn  = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=32, F=32) -- permute to channels-first for Conv1d
        h = self.tcn(x.permute(0, 2, 1))                 # (B, 64, 32)
        return self.head(h.mean(dim=-1)).squeeze(-1) * MAX_RUL  # (B,)


# ---------------------------------------------------------------------------
# Generic PyTorch trainer
# ---------------------------------------------------------------------------

def _train_torch(
    model:        nn.Module,
    X_tr:         np.ndarray,
    y_tr:         np.ndarray,
    X_va:         np.ndarray,
    y_va:         np.ndarray,
    epochs:       int   = 100,
    patience:     int   = 20,
    batch_size:   int   = 64,
    lr:           float = 3e-4,
    weight_decay: float = 1e-4,
    name:         str   = "model",
    device:       str   = "cpu",
) -> nn.Module:
    """Train with AdamW + MSE + early stopping. Returns best model on cpu."""
    dev     = torch.device(device)
    model   = model.to(dev)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr / MAX_RUL, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True,
    )
    Xv_t = torch.tensor(X_va, dtype=torch.float32).to(dev)

    best_rmse = float("inf")
    best_wts  = copy.deepcopy(model.state_dict())
    no_imp    = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb) / MAX_RUL, yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t).cpu().numpy()
        val_rmse = float(np.sqrt(np.mean((val_pred - y_va) ** 2)))

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_wts  = copy.deepcopy(model.state_dict())
            no_imp    = 0
        else:
            no_imp += 1

        if ep % 25 == 0:
            print(f"    {name:3s} ep {ep:3d}/{epochs}  val={val_rmse:.2f}  best={best_rmse:.2f}")
        if no_imp >= patience:
            print(f"    {name:3s} early-stop ep {ep}  best={best_rmse:.2f}")
            break

    model.load_state_dict(best_wts)
    return model.cpu()


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _isnan(v: float) -> bool:
    return math.isnan(v)


_MODEL_ORDER: List[str] = [
    "XGBoost", "GRU", "TCN", "CNN-baseline", "LSTM-baseline", "Conv-SA (ours)",
]
_UNCERTAINTY: Dict[str, str] = {
    "XGBoost":        "None",
    "GRU":            "None",
    "TCN":            "None",
    "CNN-baseline":   "None",
    "LSTM-baseline":  "None",
    "Conv-SA (ours)": "MC Dropout",
}
_NOTES: Dict[str, str] = {
    "XGBoost":        "Point estimate",
    "GRU":            "Point estimate",
    "TCN":            "Point estimate",
    "CNN-baseline":   "Point estimate",
    "LSTM-baseline":  "Point estimate",
    "Conv-SA (ours)": "Epistemic uncertainty",
}


def _load_table1(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load CNN/LSTM/Conv-SA rows from results/table1_rul.csv if available."""
    p   = results_dir / "table1_rul.csv"
    out: Dict[str, Dict[str, float]] = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("method", "").strip()
            try:
                out[name] = {
                    "rmse_full": float(row.get("Full-RMSE", "nan") or "nan"),
                    "rmse_late": float(row.get("Late-RMSE", "nan") or "nan"),
                    "mae_full":  float(row.get("MAE",       "nan") or "nan"),
                }
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_table(
    new_results:  Dict[str, Dict[str, Any]],
    results_dir:  Path,
) -> Dict[str, Dict[str, Any]]:
    """Merge XGBoost/GRU/TCN results with table1_rul.csv; write CSV + LaTeX.

    Returns merged dict keyed by model name (all 6 models).
    """
    existing = _load_table1(results_dir)
    _nan_row = {"rmse_full": float("nan"), "rmse_late": float("nan"), "mae_full": float("nan")}

    all_rows: Dict[str, Dict[str, Any]] = {}
    for m in _MODEL_ORDER:
        if m in new_results:
            all_rows[m] = new_results[m]
        elif m in existing:
            all_rows[m] = existing[m]
        else:
            all_rows[m] = dict(_nan_row)

    # Find best (lowest) value per metric for bold formatting in LaTeX
    def _best(key: str) -> float:
        vals = [all_rows[m][key] for m in all_rows if not _isnan(all_rows[m][key])]
        return min(vals) if vals else float("nan")

    best = {k: _best(k) for k in ("rmse_full", "rmse_late", "mae_full")}

    def _cell(v: float, b: float) -> str:
        if _isnan(v):
            return "---"
        s = f"{v:.2f}"
        return f"\\textbf{{{s}}}" if abs(v - b) < 0.005 else s

    # CSV
    csv_path = results_dir / "table_rul_baselines.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f, fieldnames=["Model", "Full-RMSE", "Late-RMSE", "MAE", "Uncertainty", "Notes"],
        )
        wr.writeheader()
        for name, m in all_rows.items():
            wr.writerow({
                "Model":       name,
                "Full-RMSE":   "" if _isnan(m["rmse_full"]) else f"{m['rmse_full']:.2f}",
                "Late-RMSE":   "" if _isnan(m["rmse_late"]) else f"{m['rmse_late']:.2f}",
                "MAE":         "" if _isnan(m["mae_full"])  else f"{m['mae_full']:.2f}",
                "Uncertainty": _UNCERTAINTY.get(name, "None"),
                "Notes":       _NOTES.get(name, ""),
            })

    # LaTeX
    tex_path = results_dir / "table_rul_baselines.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{RUL prediction comparison on PRONOSTIA Bearing 3\\_2}\n")
        f.write("\\label{tab:rul_baselines}\n")
        f.write("\\begin{tabular}{lcccll}\n\\toprule\n")
        f.write("Model & Full-RMSE & Late-RMSE & MAE & Uncertainty & Notes \\\\\n\\midrule\n")
        for name, m in all_rows.items():
            row_cells = [
                _cell(m["rmse_full"], best["rmse_full"]),
                _cell(m["rmse_late"], best["rmse_late"]),
                _cell(m["mae_full"],  best["mae_full"]),
                _UNCERTAINTY.get(name, "None"),
                _NOTES.get(name, ""),
            ]
            f.write(f"{name} & " + " & ".join(row_cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  Table -> {csv_path.name}  {tex_path.name}")
    return all_rows


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

_COLORS: Dict[str, str] = {
    "XGBoost":        "#E69F00",
    "GRU":            "#56B4E9",
    "TCN":            "#009E73",
    "CNN-baseline":   "#CC79A7",
    "LSTM-baseline":  "#D55E00",
    "Conv-SA (ours)": "#0072B2",
}


def generate_figure(
    all_rows:    Dict[str, Dict[str, Any]],
    results_dir: Path,
) -> None:
    """IEEE-style bar charts: Full-RMSE (left) and Late-RMSE (right)."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         10,
        "axes.linewidth":    0.8,
        "lines.linewidth":   1.2,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,
        "axes.spines.right": False,
        "axes.spines.top":   False,
    })

    names:      List[str] = _MODEL_ORDER
    short:      List[str] = ["XGBoost", "GRU", "TCN", "CNN", "LSTM", "Conv-SA"]
    colors:     List[str] = [_COLORS[n] for n in names]
    x:          np.ndarray = np.arange(len(names))
    convsa_idx: int        = names.index("Conv-SA (ours)")

    full_vals = [all_rows[n]["rmse_full"] for n in names]
    late_vals = [all_rows[n]["rmse_late"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    def _panel(
        ax:          plt.Axes,
        values:      List[float],
        ylabel:      str,
        panel_label: str,
    ) -> None:
        safe  = [0.0 if _isnan(v) else v for v in values]
        valid = [v for v in values if not _isnan(v)]
        ymax  = max(valid) if valid else 1.0
        offset = ymax * 0.025

        bars = ax.bar(x, safe, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

        for bar, v in zip(bars, values):
            if not _isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + offset,
                    f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7,
                )

        # Red star above Conv-SA bar
        if not _isnan(values[convsa_idx]):
            cv = values[convsa_idx]
            ax.text(
                x[convsa_idx],
                cv + offset * 5,
                "*", color="red", ha="center", va="bottom",
                fontsize=14, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(0, ymax * 1.35)
        ax.text(
            0.03, 0.97, panel_label,
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left",
        )

    _panel(ax1, full_vals, "Full-Sequence RMSE (cycles)", "(a)")
    _panel(ax2, late_vals, "Late-Stage RMSE (cycles)",    "(b)")

    fig.tight_layout()
    out = results_dir / "fig_rul_baselines.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {out.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    _proc_dir    = _PROJ / "data" / "processed"
    _results_dir = _PROJ / "results"
    _results_dir.mkdir(parents=True, exist_ok=True)

    _device = get_device(verbose=True)

    # ---- Load data ----------------------------------------------------------
    print("\nLoading training windows...")
    X_all, y_all = _load_bearing_windows(_proc_dir, TRAIN_BEARINGS)
    X_tr, y_tr, X_va, y_va = _train_val_split(X_all, y_all)
    print(f"  train={len(X_tr):,}  val={len(X_va):,}  "
          f"(from {len(X_all):,} total train windows)")

    print("Loading test windows...")
    X_test, y_test = _load_bearing_windows(_proc_dir, [TEST_BEARING])
    print(f"  test={len(X_test):,}  rul_range=[{y_test.min():.0f}, {y_test.max():.0f}]")

    new_results: Dict[str, Dict[str, Any]] = {}

    # ---- Model 1: XGBoost ---------------------------------------------------
    if _XGB_AVAILABLE:
        print("\n[1/3] XGBoost -- fitting on all train windows...")
        xgb = XGBoostRUL()
        xgb.fit(X_all, y_all)
        xgb_pred = xgb.predict(X_test)
        new_results["XGBoost"] = evaluate_rul_model(xgb_pred, y_test)
        r = new_results["XGBoost"]
        print(f"  XGBoost: full={r['rmse_full']:.2f}  "
              f"late={r['rmse_late']:.2f}  mae={r['mae_full']:.2f}")
    else:
        print("\n[1/3] XGBoost: SKIPPED (pip install xgboost to enable)")
        new_results["XGBoost"] = {
            "rmse_full": float("nan"),
            "rmse_late": float("nan"),
            "mae_full":  float("nan"),
        }

    # ---- Model 2: GRU -------------------------------------------------------
    print("\n[2/3] GRU -- training (epochs=100, patience=20)...")
    gru_model = _train_torch(
        GRURULPredictor(), X_tr, y_tr, X_va, y_va,
        epochs=100, patience=20, batch_size=64, lr=3e-4, weight_decay=1e-4,
        name="GRU", device=_device,
    )
    gru_model.eval()
    with torch.no_grad():
        gru_pred = gru_model(
            torch.tensor(X_test, dtype=torch.float32)
        ).numpy()
    new_results["GRU"] = evaluate_rul_model(gru_pred, y_test)
    r = new_results["GRU"]
    print(f"  GRU:     full={r['rmse_full']:.2f}  "
          f"late={r['rmse_late']:.2f}  mae={r['mae_full']:.2f}")

    # ---- Model 3: TCN -------------------------------------------------------
    print("\n[3/3] TCN -- training (epochs=100, patience=20)...")
    tcn_model = _train_torch(
        TCNRULPredictor(), X_tr, y_tr, X_va, y_va,
        epochs=100, patience=20, batch_size=64, lr=3e-4, weight_decay=1e-4,
        name="TCN", device=_device,
    )
    tcn_model.eval()
    with torch.no_grad():
        tcn_pred = tcn_model(
            torch.tensor(X_test, dtype=torch.float32)
        ).numpy()
    new_results["TCN"] = evaluate_rul_model(tcn_pred, y_test)
    r = new_results["TCN"]
    print(f"  TCN:     full={r['rmse_full']:.2f}  "
          f"late={r['rmse_late']:.2f}  mae={r['mae_full']:.2f}")

    # ---- Table + Figure -----------------------------------------------------
    print("\nGenerating outputs...")
    all_rows = generate_table(new_results, _results_dir)
    generate_figure(all_rows, _results_dir)

    # ---- Summary ------------------------------------------------------------
    _W = 22
    print(f"\n{'='*65}")
    print(f"  {'Model':<{_W}} {'Full-RMSE':>10} {'Late-RMSE':>10} {'MAE':>8}")
    print(f"  {'-'*52}")
    for _name in _MODEL_ORDER:
        _m  = all_rows[_name]
        _fr = "  N/A    " if _isnan(_m["rmse_full"]) else f"{_m['rmse_full']:10.2f}"
        _lr = "  N/A    " if _isnan(_m["rmse_late"]) else f"{_m['rmse_late']:10.2f}"
        _ma = "  N/A " if _isnan(_m["mae_full"])  else f"{_m['mae_full']:8.2f}"
        print(f"  {_name:<{_W}} {_fr} {_lr} {_ma}")
    print(f"{'='*65}")
    print("  * Conv-SA uses MC Dropout for epistemic uncertainty estimation")
    print(f"  Results -> {_results_dir}")
