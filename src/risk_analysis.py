"""
risk_analysis.py
================
Sweeps CVaR alpha in {0.05, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00}, training one
QR-DQN agent per alpha and generating a risk-return tradeoff analysis.

Usage
-----
    python -m src.risk_analysis                     # from project root
    python -m src.risk_analysis --eval-only         # skip training, load checkpoints
    python -m src.risk_analysis --force-retrain     # retrain even if checkpoint exists

Outputs
-------
    results/cvar_alpha_{alpha:.2f}.pth      trained agent checkpoint per alpha
    results/table_risk_analysis.csv         metrics per alpha
    results/table_risk_analysis.tex         LaTeX table (dagger on alpha=0.40)
    results/fig_risk_return.png             1x2 IEEE risk-return tradeoff plot
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer
    from src.rl_environment import PdMBearingEnv, make_env_from_processed
except ImportError:
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer
    from src.rl_environment import PdMBearingEnv, make_env_from_processed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHAS: List[float] = [0.05, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00]
PAPER_ALPHA: float  = 0.40
CM_BASELINE: float  = 0.17333   # CorrectiveMaintenance from final_comparison.csv

# Training hyperparameters matching train.py _RL_DEFAULTS exactly
_HP: Dict = {
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


# ---------------------------------------------------------------------------
# Seed / warmup helpers
# ---------------------------------------------------------------------------

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
    """Fill replay buffer with cyclic [0,0,0,1,1,2] actions (no network updates)."""
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


# ---------------------------------------------------------------------------
# Greedy evaluation helpers
# ---------------------------------------------------------------------------

def _quick_action_dist(
    agent: QRDQNAgent,
    env: PdMBearingEnv,
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    """Returns do_nothing fraction over n_episodes greedy rollouts."""
    counts = [0, 0, 0]
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i, force_degraded=False)
        done = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            counts[action] += 1
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    total = max(sum(counts), 1)
    return {
        "do_nothing": counts[0] / total,
        "repair":     counts[1] / total,
        "replace":    counts[2] / total,
    }


def _greedy_eval_fast(
    agent: QRDQNAgent,
    env: PdMBearingEnv,
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    """Lightweight greedy eval used during training to gate checkpoint saving."""
    costs:   List[float] = []
    catasts: List[int]   = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i, force_degraded=False)
        done = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        costs.append(float(info.get("total_cost", 0.0)))
        catasts.append(1 if info.get("is_failure", False) else 0)
    return {
        "mean_cost":        float(np.mean(costs)),
        "catastrophe_rate": float(np.mean(catasts)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_agent(alpha: float, device_str: str) -> QRDQNAgent:
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
        cvar_alpha=alpha,
        device=device_str,
    )


def _train_alpha_agent(
    alpha: float,
    env: PdMBearingEnv,
    results_dir: Path,
    seed: int,
    device_str: str,
    force_retrain: bool = False,
) -> QRDQNAgent:
    """Train (or load) one QR-DQN agent with the given CVaR alpha.

    Checkpoint path: results/cvar_alpha_{alpha:.2f}.pth
    Skips training if checkpoint exists and force_retrain is False.
    """
    ckpt_path = results_dir / f"cvar_alpha_{alpha:.2f}.pth"
    agent = _make_agent(alpha, device_str)

    if ckpt_path.exists() and not force_retrain:
        logger.info("Loading existing checkpoint: %s", ckpt_path)
        agent.load_checkpoint(ckpt_path)
        return agent

    logger.info(
        "Training CVaR alpha=%.2f for %d episodes (seed=%d)...",
        alpha, _HP["total_episodes"], seed,
    )
    _set_seed(seed)

    buffer = ReplayBuffer(maxlen=_HP["replay_buffer_size"])
    _warmup(env, buffer, n_episodes=_HP["warmup_episodes"], seed=seed)

    best_catast = float("inf")

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
            eval_m = _greedy_eval_fast(
                agent, env,
                n_episodes=_HP["n_eval_episodes"],
                seed=seed + 1_000_000 + ep,
            )
            catast = eval_m["catastrophe_rate"]
            cost   = eval_m["mean_cost"]

            act_d = _quick_action_dist(
                agent, env,
                n_episodes=10,
                seed=seed + 2_000_000 + ep,
            )
            not_collapsed = act_d["do_nothing"] < 0.95

            logger.info(
                "  alpha=%.2f ep %4d | catast=%.1f%% cost=%.2f eps=%.3f dn=%.0f%%",
                alpha, ep, catast * 100, cost, agent.epsilon,
                act_d["do_nothing"] * 100,
            )

            # Checkpoint logic mirrors train.py
            if catast < 0.15 and not_collapsed:
                if catast < best_catast:
                    best_catast = catast
                    agent.save_checkpoint(ckpt_path)
                    logger.info("  -> New best catast=%.1f%%, saved.", catast * 100)
            elif ep >= 500 and not_collapsed and catast < best_catast:
                best_catast = catast
                agent.save_checkpoint(ckpt_path)
                logger.info(
                    "  -> New best (non-coll) catast=%.1f%%, saved.", catast * 100
                )

    # If no checkpoint was saved (rare edge case), save the final policy
    if not ckpt_path.exists():
        agent.save_checkpoint(ckpt_path)
        logger.info("  -> No best found; saved final policy to %s", ckpt_path)
    else:
        agent.load_checkpoint(ckpt_path)

    return agent


# ---------------------------------------------------------------------------
# Full evaluation (300 episodes, extended metrics)
# ---------------------------------------------------------------------------

def evaluate_alpha(
    agent: QRDQNAgent,
    env: PdMBearingEnv,
    n_episodes: int = 300,
    seed: int = 3_000_000,
) -> Dict[str, float]:
    """Greedy evaluation with risk-oriented metrics.

    Returns
    -------
    mean_cost, std_cost, catastrophe_rate, mean_reward, cost_var,
    cvar_10_cost  (expected cost in worst 10% of episodes).
    """
    costs:   List[float] = []
    rewards: List[float] = []
    catasts: List[int]   = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i, force_degraded=False)
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        costs.append(float(info.get("total_cost", 0.0)))
        rewards.append(ep_reward)
        catasts.append(1 if info.get("is_failure", False) else 0)

    cost_arr = np.asarray(costs, dtype=np.float64)

    # CVaR_0.10 of costs: mean of worst 10% highest-cost episodes
    n_tail   = max(1, int(math.floor(0.10 * n_episodes)))
    cvar_10  = float(np.mean(np.sort(cost_arr)[::-1][:n_tail]))

    return {
        "mean_cost":        float(np.mean(cost_arr)),
        "std_cost":         float(np.std(cost_arr)),
        "catastrophe_rate": float(np.mean(catasts)),
        "mean_reward":      float(np.mean(rewards)),
        "cost_var":         float(np.var(cost_arr)),
        "cvar_10_cost":     cvar_10,
    }


# ---------------------------------------------------------------------------
# Alpha sweep orchestrator
# ---------------------------------------------------------------------------

def run_alpha_sweep(
    env: PdMBearingEnv,
    results_dir: Path,
    alphas: List[float] = ALPHAS,
    seed: int = 42,
    device_str: Optional[str] = None,
    force_retrain: bool = False,
) -> Dict[float, Dict[str, float]]:
    """Train + evaluate one QR-DQN agent per alpha value.

    Returns {alpha: metrics_dict}.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    sweep: Dict[float, Dict[str, float]] = {}

    for alpha in alphas:
        logger.info("=" * 60)
        logger.info("CVaR alpha = %.2f", alpha)
        logger.info("=" * 60)

        agent = _train_alpha_agent(
            alpha=alpha,
            env=env,
            results_dir=results_dir,
            seed=seed,
            device_str=device_str,
            force_retrain=force_retrain,
        )

        logger.info("Evaluating alpha=%.2f over %d episodes...", alpha, 300)
        metrics = evaluate_alpha(agent, env, n_episodes=300, seed=3_000_000)
        sweep[alpha] = metrics

        logger.info(
            "  alpha=%.2f | cost=%.2f+/-%.2f | catast=%.1f%% | reward=%.2f | "
            "cvar10=%.2f",
            alpha,
            metrics["mean_cost"], metrics["std_cost"],
            metrics["catastrophe_rate"] * 100,
            metrics["mean_reward"],
            metrics["cvar_10_cost"],
        )

    return sweep


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_risk_table(
    results: Dict[float, Dict[str, float]],
    alphas: List[float],
    results_dir: Path,
) -> None:
    """Write CSV and LaTeX comparison table. Alpha=PAPER_ALPHA row gets dagger."""
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "alpha":           a,
            "mean_cost":       results[a]["mean_cost"],
            "std_cost":        results[a]["std_cost"],
            "catastrophe_pct": results[a]["catastrophe_rate"] * 100,
            "mean_reward":     results[a]["mean_reward"],
            "cvar_10_cost":    results[a]["cvar_10_cost"],
        }
        for a in alphas
    ]

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    csv_path = results_dir / "table_risk_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved %s", csv_path)

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------
    # Best = minimum for cost columns, maximum for reward
    best_cost       = min(r["mean_cost"]       for r in rows)
    best_std        = min(r["std_cost"]        for r in rows)
    best_catast     = min(r["catastrophe_pct"] for r in rows)
    best_reward     = max(r["mean_reward"]     for r in rows)
    best_cvar10     = min(r["cvar_10_cost"]    for r in rows)

    def _cell(val: float, best: float, fmt: str, tol: float = 0.01) -> str:
        s = format(val, fmt)
        return f"\\textbf{{{s}}}" if abs(val - best) <= tol else s

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Risk-return tradeoff across CVaR $\alpha$ values.",
        r"$\dagger$ marks the proposed agent ($\alpha=0.40$). Bold = best per column.}",
        r"\label{tab:risk_analysis}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{cSSSSS}",
        r"\toprule",
        (
            r"CVaR $\alpha$ & {Cost $\mu$} & {Cost $\sigma$}"
            r" & {Catastrophe \%} & {Avg Reward} & {CVaR$_{0.1}$(cost)} \\"
        ),
        r"\midrule",
    ]

    for r in rows:
        a = r["alpha"]
        dagger = r"$^\dagger$" if abs(a - PAPER_ALPHA) < 1e-9 else ""
        alpha_col   = f"{a:.2f}{dagger}"
        cost_col    = _cell(r["mean_cost"],       best_cost,   ".2f")
        std_col     = _cell(r["std_cost"],        best_std,    ".2f")
        catast_col  = _cell(r["catastrophe_pct"], best_catast, ".1f", tol=0.05)
        reward_col  = _cell(r["mean_reward"],     best_reward, ".1f", tol=0.5)
        cvar10_col  = _cell(r["cvar_10_cost"],    best_cvar10, ".2f")
        tex_lines.append(
            f"{alpha_col} & {cost_col} & {std_col} & {catast_col} & {reward_col} & {cvar10_col} \\\\"
        )

    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = results_dir / "table_risk_analysis.tex"
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex_lines) + "\n")
    logger.info("Saved %s", tex_path)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_risk_plots(
    results: Dict[float, Dict[str, float]],
    alphas: List[float],
    results_dir: Path,
    paper_alpha: float = PAPER_ALPHA,
    cm_baseline: float = CM_BASELINE,
) -> None:
    """1x2 IEEE-style risk-return tradeoff figure.

    Panel (a): mean cost vs CVaR alpha, sigma shading, red vline at paper_alpha.
    Panel (b): catastrophe rate vs CVaR alpha, CM baseline, green safe-zone shade,
               annotation for proposed agent.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    alphas_arr   = np.asarray(alphas)
    mean_costs   = np.asarray([results[a]["mean_cost"]                for a in alphas])
    std_costs    = np.asarray([results[a]["std_cost"]                 for a in alphas])
    catast_pcts  = np.asarray([results[a]["catastrophe_rate"] * 100   for a in alphas])

    paper_idx    = alphas.index(paper_alpha)
    paper_catast = catast_pcts[paper_idx]
    paper_cost   = mean_costs[paper_idx]

    _BLUE   = "#0072B2"
    _RED    = "#D55E00"
    _GRAY   = "#555555"
    _GREEN  = "#009E73"

    plt.rcParams.update({
        "font.size":       9,
        "axes.titlesize":  9,
        "axes.labelsize":  9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ------------------------------------------------------------------
    # Panel (a): Mean cost vs CVaR alpha
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(
        alphas_arr, mean_costs,
        "o-", color=_BLUE, linewidth=1.8, markersize=5,
        markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=_BLUE,
    )
    ax.fill_between(
        alphas_arr,
        mean_costs - std_costs,
        mean_costs + std_costs,
        alpha=0.15, color=_BLUE, linewidth=0,
    )
    ax.axvline(x=paper_alpha, color=_RED, linestyle="--", linewidth=1.2, zorder=3)

    # Label the risk-neutral endpoint at alpha=1.00
    ax.annotate(
        "Risk-Neutral",
        xy=(1.00, mean_costs[-1]),
        xytext=(0.80, mean_costs[-1] + (mean_costs.max() - mean_costs.min()) * 0.15),
        fontsize=7, color=_GRAY,
        arrowprops=dict(arrowstyle="->", color=_GRAY, lw=0.8),
        ha="right",
    )

    ax.set_xlabel(r"CVaR $\alpha$")
    ax.set_ylabel("Mean Episode Cost")
    ax.set_title("(a) Cost vs Risk Level")
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Panel (b): Catastrophe rate vs CVaR alpha
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.plot(
        alphas_arr, catast_pcts,
        "o-", color=_RED, linewidth=1.8, markersize=5,
        markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=_RED,
    )

    # CM baseline
    ax.axhline(
        y=cm_baseline * 100, color=_GRAY, linestyle="--", linewidth=1.2,
        label=f"CM baseline ({cm_baseline * 100:.1f}%)",
    )
    ax.text(
        alphas_arr[0], cm_baseline * 100 + 0.4,
        f"CM baseline ({cm_baseline * 100:.1f}%)",
        fontsize=7, color=_GRAY, va="bottom",
    )

    # Light-green safe zone below the proposed agent's catastrophe rate
    ax.axhspan(0.0, paper_catast, alpha=0.12, color=_GREEN, linewidth=0)

    # Red vline at paper_alpha
    ax.axvline(x=paper_alpha, color=_RED, linestyle="--", linewidth=1.2, zorder=3)

    # Annotation for proposed agent
    y_range = float(catast_pcts.max() - catast_pcts.min())
    offset_y = max(1.5, y_range * 0.15)
    ax.annotate(
        f"Proposed ($\\alpha$={paper_alpha:.2f}): {paper_catast:.1f}%",
        xy=(paper_alpha, paper_catast),
        xytext=(paper_alpha + 0.12, paper_catast + offset_y),
        fontsize=7, color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9),
        ha="left",
    )

    ax.set_xlabel(r"CVaR $\alpha$")
    ax.set_ylabel("Catastrophe Rate (%)")
    ax.set_title("(b) Safety vs Risk Level")
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas], rotation=30, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.tight_layout(pad=1.2)
    out_path = results_dir / "fig_risk_return.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="CVaR alpha sweep for QR-DQN risk-return tradeoff analysis."
    )
    p.add_argument(
        "--processed-dir", default="data/processed", type=Path,
        help="Path to processed HI/RUL .npy files (default: data/processed)",
    )
    p.add_argument(
        "--results-dir", default="results", type=Path,
        help="Directory for checkpoints and outputs (default: results)",
    )
    p.add_argument("--device", default=None,
                   help="Compute device: cuda | mps | cpu (default: auto)")
    p.add_argument("--seed", default=42, type=int,
                   help="Global random seed (default: 42)")
    p.add_argument("--force-retrain", action="store_true",
                   help="Retrain agents even if checkpoint already exists")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load existing checkpoints and re-evaluate")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env_from_processed(args.processed_dir, seed=args.seed)
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        sweep: Dict[float, Dict[str, float]] = {}
        for alpha in ALPHAS:
            ckpt = results_dir / f"cvar_alpha_{alpha:.2f}.pth"
            agent = _make_agent(alpha, device_str)
            if ckpt.exists():
                agent.load_checkpoint(ckpt)
                logger.info("Loaded %s", ckpt)
            else:
                logger.warning(
                    "Checkpoint not found: %s — evaluating untrained agent", ckpt
                )
            sweep[alpha] = evaluate_alpha(agent, env, n_episodes=300, seed=3_000_000)
    else:
        sweep = run_alpha_sweep(
            env=env,
            results_dir=results_dir,
            alphas=ALPHAS,
            seed=args.seed,
            device_str=device_str,
            force_retrain=args.force_retrain,
        )

    generate_risk_table(sweep, ALPHAS, results_dir)
    generate_risk_plots(sweep, ALPHAS, results_dir)
    logger.info("Done. All outputs written to %s", results_dir)


if __name__ == "__main__":
    main()
