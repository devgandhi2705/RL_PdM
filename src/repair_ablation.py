"""
repair_ablation.py
==================
Compares two repair models by training a QR-DQN agent in each environment
variant and analysing the resulting maintenance behaviour.

  Variant 1 — PerfectRepairEnv : efficacy = 0.40 (constant, no decay)
  Variant 2 — PdMBearingEnv    : efficacy = max(0.1, 0.35 * exp(-0.4 * n))

Usage
-----
    python -m src.repair_ablation                    # from project root
    python -m src.repair_ablation --eval-only        # skip training
    python -m src.repair_ablation --force-retrain    # retrain both agents

Outputs
-------
    results/repair_perfect.pth           perfect-repair agent checkpoint
    results/repair_decay.pth             decay-repair agent checkpoint (if retrained)
    results/table_repair_ablation.csv    metrics table
    results/table_repair_ablation.tex    LaTeX table
    results/fig_repair_ablation.png      2x2 behavioural plot
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import shutil
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer
    from src.rl_environment import (
        PdMBearingEnv,
        make_env_from_processed,
        ACTION_DO_NOTHING,
        ACTION_REPAIR,
        ACTION_REPLACE,
    )
except ImportError:
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer
    from src.rl_environment import (
        PdMBearingEnv,
        make_env_from_processed,
        ACTION_DO_NOTHING,
        ACTION_REPAIR,
        ACTION_REPLACE,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CVAR_ALPHA: float = 0.40

# Training HPs matching train.py _RL_DEFAULTS exactly
_HP: Dict[str, Any] = {
    "N_quantiles":            51,
    "lr":                     5e-4,
    "gamma":                  0.99,
    "epsilon_start":          1.0,
    "epsilon_end":            0.10,
    "epsilon_decay_episodes": 3000,
    "target_update_freq":     200,
    "batch_size":             128,
    "replay_buffer_size":     50_000,
    "total_episodes":         5000,
    "warmup_episodes":        100,
    "eval_interval":          200,
    "n_eval_episodes":        50,
}

_VARIANT_LABELS: Dict[str, str] = {
    "perfect": "Perfect Repair",
    "decay":   "Exponential Decay (ours)",
}


# ===========================================================================
# Environment variant 1 — Perfect Repair
# ===========================================================================

class PerfectRepairEnv(PdMBearingEnv):
    """PdMBearingEnv with constant repair efficacy = 0.40 (no exponential decay).

    Only the repair action is overridden; do-nothing and replace delegate
    to the parent unchanged.  The cost formula (4.0 + 1.5 * n_repairs) is
    intentionally kept identical so comparisons reflect pure repair-efficacy
    differences.
    """

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Delegate do-nothing and replace to parent (no change needed)
        if action != ACTION_REPAIR:
            return super().step(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}.")

        t     = self._t
        hi_t  = self._safe_hi(t)
        rul_t = self._safe_rul(t)

        # Perfect repair: constant efficacy = 0.40, no n_repairs penalty
        hi_gain = 0.40 * (1.0 - hi_t)
        self._hi_copy[t:] = np.minimum(
            self._hi_copy[t:] + hi_gain,
            hi_t + hi_gain,
        )
        self._n_repairs += 1
        step_cost = 4.0 + 1.5 * self._n_repairs
        reward    = -step_cost

        # Advance timestep — mirrors parent exactly
        self._t                   += 1
        self._episode_steps       += 1
        self._steps_since_replace += 1

        tn     = self._t
        hi_nxt = float(self._hi_copy[tn]) if tn < len(self._hi_copy) else 0.0
        terminated = hi_nxt < 0.055
        truncated  = (
            tn >= min(len(self._hi_copy) - 1, self._max_steps)
            or self._episode_steps >= self._max_steps
        )

        is_failure = False
        if terminated:
            reward    -= 100.0
            step_cost += 100.0
            is_failure = True
            self._n_failures += 1

        self._total_cost += step_cost

        obs  = self._build_obs()
        info: Dict[str, Any] = {
            "cost":                step_cost,
            "action_name":         "repair",
            "hi_t":                hi_t,
            "rul_t":               rul_t,
            "mean_rul":            rul_t * 125.0,
            "n_repairs":           self._n_repairs,
            "n_replacements":      self._n_replacements,
            "steps_since_replace": self._steps_since_replace,
            "is_failure":          is_failure,
            "failure":             is_failure,
            "total_cost":          self._total_cost,
            "bearing_id":          self._current_bid,
        }
        return obs, reward, terminated, truncated, info


def make_perfect_repair_env(
    processed_dir: Path,
    seed: Optional[int] = None,
) -> PerfectRepairEnv:
    """Load HI/RUL arrays and return a PerfectRepairEnv instance."""
    processed_dir = Path(processed_dir)
    hi_seqs:  Dict[str, np.ndarray] = {}
    rul_seqs: Dict[str, np.ndarray] = {}

    for hi_path in sorted(processed_dir.glob("*_hi.npy")):
        bid      = hi_path.stem.replace("_hi", "")
        rul_path = processed_dir / f"{bid}_rul.npy"
        if not rul_path.exists():
            continue
        hi_seqs[bid]  = np.load(hi_path).astype(np.float32)
        rul_seqs[bid] = (np.load(rul_path) / 125.0).astype(np.float32)

    if not hi_seqs:
        raise FileNotFoundError(f"No *_hi.npy files found in {processed_dir}")

    return PerfectRepairEnv(hi_seqs, rul_seqs, seed=seed)


# ===========================================================================
# Shared training utilities
# ===========================================================================

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warmup(
    env: PdMBearingEnv,
    buffer: ReplayBuffer,
    n_episodes: int,
    seed: int,
) -> None:
    action_cycle = cycle([0, 0, 0, 1, 1, 2])
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = next(action_cycle)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs


def _make_agent(device_str: str) -> QRDQNAgent:
    return QRDQNAgent(
        state_dim=5,
        n_actions=3,
        N_quantiles=_HP["N_quantiles"],
        lr=_HP["lr"],
        gamma=_HP["gamma"],
        epsilon_start=_HP["epsilon_start"],
        epsilon_end=_HP["epsilon_end"],
        epsilon_decay_episodes=_HP["epsilon_decay_episodes"],
        target_update_freq=_HP["target_update_freq"],
        batch_size=_HP["batch_size"],
        risk_mode="cvar",
        cvar_alpha=CVAR_ALPHA,
        device=device_str,
    )


def _quick_eval(
    agent: QRDQNAgent,
    env: PdMBearingEnv,
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    costs:   List[float] = []
    catasts: List[int]   = []
    dn_count = 0
    total_actions = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i, force_degraded=False)
        done = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            if action == ACTION_DO_NOTHING:
                dn_count += 1
            total_actions += 1
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        costs.append(float(info.get("total_cost", 0.0)))
        catasts.append(1 if info.get("is_failure", False) else 0)
    return {
        "mean_cost":        float(np.mean(costs)),
        "catastrophe_rate": float(np.mean(catasts)),
        "dn_fraction":      dn_count / max(total_actions, 1),
    }


def _train_agent(
    env: PdMBearingEnv,
    ckpt_path: Path,
    seed: int,
    device_str: str,
    force_retrain: bool = False,
    label: str = "",
) -> QRDQNAgent:
    """Train a QR-DQN agent in the given env. Skips if checkpoint exists."""
    agent = _make_agent(device_str)

    if ckpt_path.exists() and not force_retrain:
        logger.info("[%s] Loading checkpoint: %s", label, ckpt_path)
        agent.load_checkpoint(ckpt_path)
        return agent

    logger.info("[%s] Training %d episodes (seed=%d)...", label, _HP["total_episodes"], seed)
    _set_seed(seed)

    buffer       = ReplayBuffer(maxlen=_HP["replay_buffer_size"])
    best_catast  = float("inf")

    _warmup(env, buffer, n_episodes=_HP["warmup_episodes"], seed=seed)

    for ep in range(1, _HP["total_episodes"] + 1):
        obs, _ = env.reset(seed=seed + ep, force_degraded=(random.random() < 0.40))
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, float(done))
            if buffer.is_ready(agent.batch_size):
                agent.update(buffer.sample(agent.batch_size))
            obs = next_obs

        agent.step_episode()

        if ep % _HP["eval_interval"] == 0:
            eval_m     = _quick_eval(
                agent, env,
                n_episodes=_HP["n_eval_episodes"],
                seed=seed + 1_000_000 + ep,
            )
            catast     = eval_m["catastrophe_rate"]
            cost       = eval_m["mean_cost"]
            collapsed  = eval_m["dn_fraction"] >= 0.95

            logger.info(
                "  [%s] ep %4d | catast=%.1f%% cost=%.2f eps=%.3f",
                label, ep, catast * 100, cost, agent.epsilon,
            )

            if catast < 0.15 and not collapsed:
                if catast < best_catast:
                    best_catast = catast
                    agent.save_checkpoint(ckpt_path)
                    logger.info("  -> New best catast=%.1f%%  saved.", catast * 100)
            elif ep >= 500 and not collapsed and catast < best_catast:
                best_catast = catast
                agent.save_checkpoint(ckpt_path)
                logger.info("  -> New best (non-coll)  saved.")

    if not ckpt_path.exists():
        agent.save_checkpoint(ckpt_path)
    else:
        agent.load_checkpoint(ckpt_path)

    return agent


# ===========================================================================
# Full evaluation with behavioural analysis
# ===========================================================================

def _is_better_representative(
    n_repairs: int,
    n_replacements: int,
    best: Optional[Dict],
) -> bool:
    """Pick the episode with the most repairs that also has ≥1 replacement."""
    if best is None:
        return True
    has_repl   = n_replacements > 0
    best_repl  = best["n_replacements"] > 0
    if has_repl and not best_repl:
        return True
    if has_repl and best_repl and n_repairs > best["n_repairs"]:
        return True
    if not has_repl and not best_repl and n_repairs > best["n_repairs"]:
        return True
    return False


def evaluate_variant(
    agent: QRDQNAgent,
    env: PdMBearingEnv,
    n_episodes: int = 300,
    seed: int = 3_000_000,
) -> Dict[str, Any]:
    """Run greedy evaluation and collect aggregate + per-episode behavioural data.

    Returns
    -------
    mean_cost, std_cost, catastrophe_rate, mean_n_repairs, mean_n_replacements,
    mean_repairs_before_replace, costs (array), repairs_before_replace (array),
    rep_hi_traj, rep_repair_ts, rep_replace_ts  (representative episode data)
    """
    costs:                  List[float] = []
    catasts:                List[int]   = []
    n_repairs_per_ep:       List[int]   = []
    n_replacements_per_ep:  List[int]   = []
    repairs_before_replace: List[int]   = []

    representative: Optional[Dict] = None   # best episode for trajectory plot

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i, force_degraded=False)
        done = False

        ep_repairs       = 0
        ep_replacements  = 0
        repairs_in_cycle = 0

        hi_traj:    List[float] = [float(obs[0])]
        repair_ts:  List[int]   = []
        replace_ts: List[int]   = []
        t = 0

        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            t   += 1

            hi_traj.append(float(info.get("hi_t", float(obs[0]))))

            if action == ACTION_REPAIR:
                ep_repairs       += 1
                repairs_in_cycle += 1
                repair_ts.append(t)
            elif action == ACTION_REPLACE:
                ep_replacements += 1
                repairs_before_replace.append(repairs_in_cycle)
                repairs_in_cycle = 0
                replace_ts.append(t)

        costs.append(float(info.get("total_cost", 0.0)))
        catasts.append(1 if info.get("is_failure", False) else 0)
        n_repairs_per_ep.append(ep_repairs)
        n_replacements_per_ep.append(ep_replacements)

        if _is_better_representative(ep_repairs, ep_replacements, representative):
            representative = {
                "hi_traj":    hi_traj,
                "repair_ts":  repair_ts,
                "replace_ts": replace_ts,
                "n_repairs":       ep_repairs,
                "n_replacements":  ep_replacements,
            }

    rpr_arr = np.array(repairs_before_replace, dtype=float) if repairs_before_replace else np.array([0.0])

    rep = representative or {
        "hi_traj": [0.0], "repair_ts": [], "replace_ts": [],
        "n_repairs": 0, "n_replacements": 0,
    }

    return {
        "mean_cost":                 float(np.mean(costs)),
        "std_cost":                  float(np.std(costs)),
        "catastrophe_rate":          float(np.mean(catasts)),
        "mean_n_repairs":            float(np.mean(n_repairs_per_ep)),
        "mean_n_replacements":       float(np.mean(n_replacements_per_ep)),
        "mean_repairs_before_replace": float(np.mean(rpr_arr)),
        "costs":                     np.asarray(costs, dtype=float),
        "repairs_before_replace":    rpr_arr,
        "rep_hi_traj":               rep["hi_traj"],
        "rep_repair_ts":             rep["repair_ts"],
        "rep_replace_ts":            rep["replace_ts"],
    }


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_ablation(
    processed_dir: Path,
    results_dir:   Path,
    seed:          int  = 42,
    device_str:    Optional[str] = None,
    force_retrain: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Train (or load) both variants and evaluate them. Returns results dict."""
    results_dir.mkdir(parents=True, exist_ok=True)

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    perfect_ckpt = results_dir / "repair_perfect.pth"
    decay_ckpt   = results_dir / "repair_decay.pth"
    fallback_ckpt = results_dir.parent / "00_primary_cvar_qrdqn" / "qrdqn_best.pth"

    # ---- Perfect Repair env + agent ----------------------------------------
    perfect_env   = make_perfect_repair_env(processed_dir, seed=seed)
    perfect_agent = _train_agent(
        env=perfect_env,
        ckpt_path=perfect_ckpt,
        seed=seed,
        device_str=device_str,
        force_retrain=force_retrain,
        label="PerfectRepair",
    )

    # ---- Decay Repair env + agent ------------------------------------------
    decay_env = make_env_from_processed(processed_dir, seed=seed)

    if decay_ckpt.exists() and not force_retrain:
        decay_agent = _make_agent(device_str)
        logger.info("[DecayRepair] Loading checkpoint: %s", decay_ckpt)
        decay_agent.load_checkpoint(decay_ckpt)
    elif fallback_ckpt.exists() and not force_retrain:
        # Reuse existing best checkpoint (trained with decay env by train.py)
        decay_agent = _make_agent(device_str)
        logger.info("[DecayRepair] Loading fallback qrdqn_best.pth: %s", fallback_ckpt)
        decay_agent.load_checkpoint(fallback_ckpt)
        # Mirror checkpoint to canonical path for reproducibility
        shutil.copy2(fallback_ckpt, decay_ckpt)
    else:
        decay_agent = _train_agent(
            env=decay_env,
            ckpt_path=decay_ckpt,
            seed=seed,
            device_str=device_str,
            force_retrain=force_retrain,
            label="DecayRepair",
        )

    # ---- Evaluate both variants --------------------------------------------
    logger.info("Evaluating Perfect Repair (300 episodes)...")
    perfect_results = evaluate_variant(perfect_agent, perfect_env,
                                       n_episodes=300, seed=3_000_000)

    logger.info("Evaluating Exponential Decay (300 episodes)...")
    decay_results   = evaluate_variant(decay_agent, decay_env,
                                       n_episodes=300, seed=3_000_000)

    for label, m in [("Perfect Repair", perfect_results), ("Decay Repair", decay_results)]:
        logger.info(
            "  %s | cost=%.2f±%.2f | catast=%.1f%% | repairs/ep=%.2f | "
            "replaces/ep=%.2f | repairs-before-replace=%.2f",
            label,
            m["mean_cost"], m["std_cost"],
            m["catastrophe_rate"] * 100,
            m["mean_n_repairs"], m["mean_n_replacements"],
            m["mean_repairs_before_replace"],
        )

    return {"perfect": perfect_results, "decay": decay_results}


# ===========================================================================
# Table generation
# ===========================================================================

def generate_repair_table(
    results: Dict[str, Dict[str, Any]],
    results_dir: Path,
) -> None:
    """Write CSV and LaTeX table comparing the two repair variants."""
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "repair_model":             _VARIANT_LABELS["perfect"],
            "mean_cost":                results["perfect"]["mean_cost"],
            "std_cost":                 results["perfect"]["std_cost"],
            "catastrophe_pct":          results["perfect"]["catastrophe_rate"] * 100,
            "mean_n_repairs":           results["perfect"]["mean_n_repairs"],
            "mean_n_replacements":      results["perfect"]["mean_n_replacements"],
            "mean_repairs_before_repl": results["perfect"]["mean_repairs_before_replace"],
        },
        {
            "repair_model":             _VARIANT_LABELS["decay"],
            "mean_cost":                results["decay"]["mean_cost"],
            "std_cost":                 results["decay"]["std_cost"],
            "catastrophe_pct":          results["decay"]["catastrophe_rate"] * 100,
            "mean_n_repairs":           results["decay"]["mean_n_repairs"],
            "mean_n_replacements":      results["decay"]["mean_n_replacements"],
            "mean_repairs_before_repl": results["decay"]["mean_repairs_before_replace"],
        },
    ]

    # CSV
    csv_path = results_dir / "table_repair_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved %s", csv_path)

    # LaTeX — bold lower cost / lower catastrophe rate
    best_cost   = min(r["mean_cost"]       for r in rows)
    best_std    = min(r["std_cost"]        for r in rows)
    best_catast = min(r["catastrophe_pct"] for r in rows)

    def _b(val: float, best: float, fmt: str, tol: float = 0.01) -> str:
        s = format(val, fmt)
        return f"\\textbf{{{s}}}" if abs(val - best) <= tol else s

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Effect of repair model on trained policy behaviour.",
        r"Bold = best per cost/safety column.}",
        r"\label{tab:repair_ablation}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lSSSSSS}",   # l + 6 S columns
        r"\toprule",
        (
            r"Repair Model & {Cost $\mu$} & {Cost $\sigma$}"
            r" & {Catast.~\%} & {Repairs/ep} & {Reps/ep}"
            r" & {Repairs-before-replace} \\"
        ),
        r"\midrule",
    ]

    for r in rows:
        cols = [
            r["repair_model"].replace("(ours)", "(ours)"),
            _b(r["mean_cost"],        best_cost,   ".2f"),
            _b(r["std_cost"],         best_std,    ".2f"),
            _b(r["catastrophe_pct"],  best_catast, ".1f", tol=0.05),
            f"{r['mean_n_repairs']:.2f}",
            f"{r['mean_n_replacements']:.2f}",
            f"{r['mean_repairs_before_repl']:.2f}",
        ]
        tex_lines.append(" & ".join(cols) + r" \\")

    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = results_dir / "table_repair_ablation.tex"
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex_lines) + "\n")
    logger.info("Saved %s", tex_path)


# ===========================================================================
# Plot generation
# ===========================================================================

def _plot_hi_trajectory(
    ax: plt.Axes,
    hi_traj:    List[float],
    repair_ts:  List[int],
    replace_ts: List[int],
    title:      str,
    variant:    str,
) -> None:
    """Draw one HI trajectory panel with repair/replace event markers."""
    ts  = list(range(len(hi_traj)))
    ax.plot(ts, hi_traj, color="#2166ac", linewidth=1.4, zorder=2)
    ax.fill_between(ts, 0, hi_traj, alpha=0.08, color="#2166ac")

    # Repair markers: upward triangle at the post-repair HI value
    for rt in repair_ts:
        if rt < len(hi_traj):
            ax.scatter(
                rt, hi_traj[rt],
                marker="^", s=60, color="#1a9641", zorder=4,
                linewidths=0.8, edgecolors="white",
            )

    # Replace markers: red star
    for rt in replace_ts:
        if rt < len(hi_traj):
            ax.scatter(
                rt, hi_traj[rt],
                marker="*", s=100, color="#d7191c", zorder=4,
                linewidths=0.6, edgecolors="white",
            )

    # Failure zone
    ax.axhline(y=0.055, color="#d7191c", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.text(
        len(hi_traj) * 0.02, 0.07, "failure threshold",
        fontsize=6.5, color="#d7191c", va="bottom",
    )

    # Custom legend proxies
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color="#2166ac", lw=1.4, label="HI"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1a9641",
               markersize=7, label="Repair"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#d7191c",
               markersize=9, label="Replace"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Health Index (HI)", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def generate_repair_plots(
    results: Dict[str, Dict[str, Any]],
    results_dir: Path,
) -> None:
    """2×2 IEEE-style behavioural comparison figure.

    (a) HI trajectory — Perfect Repair
    (b) HI trajectory — Exponential Decay
    (c) Episode cost distribution (overlapping histograms)
    (d) Repairs-before-replacement (side-by-side box plots)
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size":       8,
        "axes.titlesize":  8,
        "axes.labelsize":  8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    # ------------------------------------------------------------------
    # Panel (a): HI trajectory — Perfect Repair
    # ------------------------------------------------------------------
    _plot_hi_trajectory(
        axes[0, 0],
        hi_traj=results["perfect"]["rep_hi_traj"],
        repair_ts=results["perfect"]["rep_repair_ts"],
        replace_ts=results["perfect"]["rep_replace_ts"],
        title="(a) HI Trajectory — Perfect Repair",
        variant="perfect",
    )

    # ------------------------------------------------------------------
    # Panel (b): HI trajectory — Exponential Decay
    # ------------------------------------------------------------------
    _plot_hi_trajectory(
        axes[0, 1],
        hi_traj=results["decay"]["rep_hi_traj"],
        repair_ts=results["decay"]["rep_repair_ts"],
        replace_ts=results["decay"]["rep_replace_ts"],
        title="(b) HI Trajectory — Exponential Decay",
        variant="decay",
    )

    # ------------------------------------------------------------------
    # Panel (c): Episode cost distributions (overlapping histograms)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    costs_p = results["perfect"]["costs"]
    costs_d = results["decay"]["costs"]

    all_costs = np.concatenate([costs_p, costs_d])
    bins = np.linspace(all_costs.min(), all_costs.max(), 31)

    ax.hist(costs_p, bins=bins, alpha=0.5, color="#1a9641",
            label=_VARIANT_LABELS["perfect"], edgecolor="white", linewidth=0.3)
    ax.hist(costs_d, bins=bins, alpha=0.5, color="#2166ac",
            label=_VARIANT_LABELS["decay"],   edgecolor="white", linewidth=0.3)

    ax.axvline(x=float(np.mean(costs_p)), color="#1a9641",
               linestyle="--", linewidth=1.2, label=f"Mean P: {np.mean(costs_p):.1f}")
    ax.axvline(x=float(np.mean(costs_d)), color="#2166ac",
               linestyle="--", linewidth=1.2, label=f"Mean D: {np.mean(costs_d):.1f}")

    ax.set_xlabel("Episode Cost")
    ax.set_ylabel("Frequency")
    ax.set_title("(c) Cost Distribution")
    ax.legend(fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Panel (d): Repairs before replacement (box plots)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    rpr_p = results["perfect"]["repairs_before_replace"]
    rpr_d = results["decay"]["repairs_before_replace"]

    bp = ax.boxplot(
        [rpr_p, rpr_d],
        labels=[_VARIANT_LABELS["perfect"], _VARIANT_LABELS["decay"]],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        widths=0.45,
    )
    bp["boxes"][0].set_facecolor("#1a9641")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#2166ac")
    bp["boxes"][1].set_alpha(0.6)

    ax.set_ylabel("Repairs before Replacement")
    ax.set_title("(d) Repair Cycles per Replacement")
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.tight_layout(pad=1.5)
    out_path = results_dir / "fig_repair_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Repair model ablation: Perfect vs Exponential Decay repair."
    )
    p.add_argument("--processed-dir", default="data/processed", type=Path)
    p.add_argument("--results-dir",   default="results/07_repair_ablation", type=Path)
    p.add_argument("--device",        default=None)
    p.add_argument("--seed",          default=42, type=int)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--eval-only",     action="store_true",
                   help="Skip training; load existing checkpoints and re-evaluate.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        # Load both checkpoints and evaluate without training
        perfect_env   = make_perfect_repair_env(args.processed_dir, seed=args.seed)
        decay_env     = make_env_from_processed(args.processed_dir, seed=args.seed)

        perfect_agent = _make_agent(device_str)
        decay_agent   = _make_agent(device_str)

        perfect_ckpt = results_dir / "repair_perfect.pth"
        decay_ckpt   = results_dir / "repair_decay.pth"
        fallback_ckpt = results_dir.parent / "00_primary_cvar_qrdqn" / "qrdqn_best.pth"

        if perfect_ckpt.exists():
            perfect_agent.load_checkpoint(perfect_ckpt)
            logger.info("Loaded %s", perfect_ckpt)
        else:
            logger.warning("No perfect repair checkpoint found at %s", perfect_ckpt)

        if decay_ckpt.exists():
            decay_agent.load_checkpoint(decay_ckpt)
            logger.info("Loaded %s", decay_ckpt)
        elif fallback_ckpt.exists():
            decay_agent.load_checkpoint(fallback_ckpt)
            logger.info("Loaded fallback %s", fallback_ckpt)
        else:
            logger.warning("No decay repair checkpoint found.")

        perfect_results = evaluate_variant(perfect_agent, perfect_env, 300, 3_000_000)
        decay_results   = evaluate_variant(decay_agent,   decay_env,   300, 3_000_000)
        ablation_results = {"perfect": perfect_results, "decay": decay_results}
    else:
        ablation_results = run_ablation(
            processed_dir=args.processed_dir,
            results_dir=results_dir,
            seed=args.seed,
            device_str=device_str,
            force_retrain=args.force_retrain,
        )

    generate_repair_table(ablation_results, results_dir)
    generate_repair_plots(ablation_results, results_dir)
    logger.info("Done. Outputs written to %s", results_dir)


if __name__ == "__main__":
    main()
