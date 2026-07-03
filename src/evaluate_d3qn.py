"""
evaluate_d3qn.py
================
Post-training evaluation comparing D3QN-CVaR against all prior agents.

ASSUMES results/d3qn_cvar_best.pth already exists.
Run full training first: python -m src.train_d3qn --episodes 5000

Outputs
-------
    results/fig_d3qn_learning_curve.png
    results/fig_d3qn_risk_return.png
    results/fig_d3qn_comparison.png
    results/table_d3qn_significance.csv
    results/table_d3qn_significance_footnote.txt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

try:
    from src.logging_config import setup_logger
    from src.device import get_device
    from src.dueling_qrdqn import D3QNAgent, ThreeDStateEnv
    from src.qrdqn_agent import QRDQNAgent
    from src.rl_benchmarks import DDQNAgent, DuelingDQNAgent, PPOAgent
    from src.rl_environment import make_env_from_processed
except ImportError:
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from src.logging_config import setup_logger
    from src.device import get_device
    from src.dueling_qrdqn import D3QNAgent, ThreeDStateEnv
    from src.qrdqn_agent import QRDQNAgent
    from src.rl_benchmarks import DDQNAgent, DuelingDQNAgent, PPOAgent
    from src.rl_environment import make_env_from_processed

logger = setup_logger(__name__)

# IEEE style
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

_N_EVAL = 300


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def _eval_agent(
    agent,
    env,
    n_episodes: int,
    seed: int,
    agent_name: str,
) -> Dict[str, float]:
    """Run n_episodes greedy (epsilon=0); return metrics dict."""
    saved_eps = getattr(agent, "epsilon", None)
    if saved_eps is not None:
        agent.epsilon = 0.0

    costs: List[float] = []
    catasts: List[int] = []
    action_counts = [0, 0, 0]

    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _  = env.reset(seed=ep_seed)
        done    = False
        while not done:
            try:
                action = agent.select_action(obs, greedy=True)
            except TypeError:
                action = agent.select_action(obs)
            action_counts[action] += 1
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        costs.append(float(info["total_cost"]))
        catasts.append(1 if info.get("is_failure", False) else 0)

    if saved_eps is not None:
        agent.epsilon = saved_eps

    total_ac = max(sum(action_counts), 1)
    return {
        "mean_cost":       float(np.mean(costs)),
        "std_cost":        float(np.std(costs)),
        "catastrophe_pct": float(np.mean(catasts)) * 100.0,
        "dn_pct":          action_counts[0] / total_ac * 100.0,
        "rp_pct":          action_counts[1] / total_ac * 100.0,
        "rx_pct":          action_counts[2] / total_ac * 100.0,
    }


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def _load_agents(
    results_dir: Path,
    processed_dir: Path,
    rul_ckpt: Path,
    device_str: str,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load all available agents and their matching environments.

    Returns
    -------
    agents : {name: agent_object}
    envs   : {name: env_object}

    Each agent is evaluated in the state space IT WAS TRAINED ON.
    D3QN-CVaR: 3D state (ThreeDStateEnv)
    All others: 5D state (PdMBearingEnv)
    """
    agents: Dict[str, Any] = {}
    envs:   Dict[str, Any] = {}

    # -- 3D environment (for D3QN-CVaR) ------------------------------------
    env_3d = ThreeDStateEnv.build(processed_dir, rul_ckpt, device_str=device_str, seed=seed)
    # -- 5D environment (for all prior agents) -----------------------------
    env_5d = make_env_from_processed(processed_dir, seed=seed)

    # ---- D3QN-CVaR (new) -------------------------------------------------
    d3qn_ckpt = results_dir / "d3qn_cvar_best.pth"
    d3qn = D3QNAgent(state_dim=3, n_actions=3, N_quantiles=51, device=device_str)
    d3qn.load_checkpoint(d3qn_ckpt)
    agents["D3QN-CVaR (ours)"] = d3qn
    envs["D3QN-CVaR (ours)"]   = env_3d

    # ---- QR-DQN variants (5D) --------------------------------------------
    qrdqn_ckpt = results_dir.parent / "00_primary_cvar_qrdqn" / "qrdqn_best.pth"
    if qrdqn_ckpt.exists():
        cvar_agent = QRDQNAgent(state_dim=5, n_actions=3, N_quantiles=51, device=device_str)
        cvar_agent.load_checkpoint(qrdqn_ckpt)
        agents["CVaR QR-DQN"] = cvar_agent
        envs["CVaR QR-DQN"]   = env_5d

        rn_agent = QRDQNAgent(state_dim=5, n_actions=3, N_quantiles=51,
                              risk_mode="mean", device=device_str)
        rn_agent.load_checkpoint(qrdqn_ckpt)
        rn_agent.risk_mode = "mean"  # override after load
        agents["Risk-Neutral QR-DQN"] = rn_agent
        envs["Risk-Neutral QR-DQN"]   = env_5d
    else:
        logger.warning("qrdqn_best.pth not found — skipping QR-DQN variants.")

    # ---- DDQN (5D) -------------------------------------------------------
    ddqn_ckpt = results_dir.parent / "04_rl_benchmarks_ddqn_dueling_ppo" / "ddqn_best.pth"
    if ddqn_ckpt.exists():
        try:
            ddqn = DDQNAgent(state_dim=5, n_actions=3, device=device_str)
            ddqn.load_checkpoint(ddqn_ckpt)
            agents["DDQN"] = ddqn
            envs["DDQN"]   = env_5d
        except Exception as exc:
            logger.warning("Could not load DDQN: %s", exc)
    else:
        logger.warning("ddqn_best.pth not found — skipping DDQN.")

    # ---- Dueling DQN (5D) ------------------------------------------------
    dueling_ckpt = results_dir.parent / "04_rl_benchmarks_ddqn_dueling_ppo" / "dueling_dqn_best.pth"
    if dueling_ckpt.exists():
        logger.info(
            "Evaluating Dueling DQN (originally trained on 5D state) -- "
            "using its existing checkpoint, action selection only, no retraining."
        )
        try:
            dueling = DuelingDQNAgent(state_dim=5, n_actions=3, device=device_str)
            dueling.load_checkpoint(dueling_ckpt)
            agents["Dueling DQN"] = dueling
            envs["Dueling DQN"]   = env_5d
        except Exception as exc:
            logger.warning("Could not load Dueling DQN: %s", exc)
    else:
        logger.warning("dueling_dqn_best.pth not found — skipping Dueling DQN.")

    # ---- PPO (5D) --------------------------------------------------------
    ppo_ckpt = results_dir.parent / "04_rl_benchmarks_ddqn_dueling_ppo" / "ppo_best.pth"
    if ppo_ckpt.exists():
        try:
            ppo = PPOAgent(state_dim=5, n_actions=3, device=device_str)
            ppo.load_checkpoint(ppo_ckpt)
            agents["PPO"] = ppo
            envs["PPO"]   = env_5d
        except Exception as exc:
            logger.warning("Could not load PPO: %s", exc)
    else:
        logger.warning("ppo_best.pth not found — skipping PPO.")

    logger.info(
        "Agents loaded for evaluation: %s  "
        "(IMPORTANT: D3QN-CVaR on 3D state; all others on 5D state — "
        "state-space differs, see State Dim column in output table)",
        list(agents.keys()),
    )
    return agents, envs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_STATE_DIM_MAP = {
    "D3QN-CVaR (ours)":   3,
    "CVaR QR-DQN":        5,
    "Risk-Neutral QR-DQN": 5,
    "DDQN":               5,
    "Dueling DQN":        5,
    "PPO":                5,
}

_COLORS = {
    "D3QN-CVaR (ours)":    "#0072B2",
    "CVaR QR-DQN":         "#56B4E9",
    "Risk-Neutral QR-DQN": "#CC79A7",
    "DDQN":                "#E69F00",
    "Dueling DQN":         "#009E73",
    "PPO":                 "#D55E00",
}


def _fig_learning_curve(results_dir: Path) -> None:
    """D3QN-CVaR and CVaR QR-DQN training rewards (smoothed, window=100)."""
    d3qn_log  = results_dir / "d3qn_training_log.csv"
    qrdqn_log = results_dir.parent / "00_primary_cvar_qrdqn" / "training_log.csv"

    if not d3qn_log.exists():
        logger.warning("d3qn_training_log.csv not found — skipping learning curve figure.")
        return

    def _load_train_rewards(path: Path) -> np.ndarray:
        df = pd.read_csv(path)
        return df.loc[df["phase"] == "train", "ep_reward"].to_numpy(dtype=float)

    def _smooth(arr: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(arr) < w:
            return np.arange(len(arr)), arr.copy()
        s = np.convolve(arr, np.ones(w) / w, mode="valid")
        return np.arange(w - 1, len(arr)), s

    fig, ax = plt.subplots(figsize=(7, 3.5))

    r_d3qn = _load_train_rewards(d3qn_log)
    xs, ys = _smooth(r_d3qn, 100)
    ax.plot(xs, ys, color=_COLORS["D3QN-CVaR (ours)"], linewidth=1.4,
            label="D3QN-CVaR (ours)")

    if qrdqn_log.exists():
        r_qr = _load_train_rewards(qrdqn_log)
        xs2, ys2 = _smooth(r_qr, 100)
        ax.plot(xs2, ys2, color=_COLORS["CVaR QR-DQN"], linewidth=1.4,
                linestyle="--", label="CVaR QR-DQN")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (smoothed, w=100)")
    ax.set_title("Learning curves — D3QN-CVaR vs CVaR QR-DQN")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = results_dir / "fig_d3qn_learning_curve.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def _fig_risk_return(
    results:     Dict[str, Dict[str, float]],
    results_dir: Path,
) -> None:
    """Scatter: catastrophe% (x) vs cost (y). 3 labeled points."""
    fig, ax = plt.subplots(figsize=(5, 4))

    plot_agents = [n for n in ["D3QN-CVaR (ours)", "CVaR QR-DQN", "Risk-Neutral QR-DQN"]
                   if n in results]

    for name in plot_agents:
        m = results[name]
        color = _COLORS.get(name, "gray")
        marker = "★" if "D3QN" in name else "o"
        ax.scatter(
            m["catastrophe_pct"], m["mean_cost"],
            color=color, s=80, zorder=5,
            marker="*" if "D3QN" in name else "o",
        )
        ax.annotate(
            name,
            xy=(m["catastrophe_pct"], m["mean_cost"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Catastrophe rate (%)")
    ax.set_ylabel("Mean cost")
    ax.set_title("Risk-return tradeoff")
    ax.grid(True, alpha=0.3)

    out = results_dir / "fig_d3qn_risk_return.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def _fig_comparison(
    results:     Dict[str, Dict[str, float]],
    results_dir: Path,
) -> None:
    """Grouped bar chart: cost (left axis) + catastrophe% (right axis)."""
    agents = list(results.keys())
    x      = np.arange(len(agents))
    costs  = [results[n]["mean_cost"]       for n in agents]
    catast = [results[n]["catastrophe_pct"] for n in agents]
    colors = [_COLORS.get(n, "#888888")     for n in agents]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    w = 0.35
    bars1 = ax1.bar(x - w / 2, costs,  w, color=colors, alpha=0.85, label="Cost")
    bars2 = ax2.bar(x + w / 2, catast, w, color=colors, alpha=0.45, hatch="//",
                    edgecolor="gray", label="Catastrophe %")

    # Highlight D3QN bar with black border + asterisk
    for i, name in enumerate(agents):
        if "D3QN" in name:
            bars1[i].set_edgecolor("black")
            bars1[i].set_linewidth(1.8)
            bars2[i].set_edgecolor("black")
            bars2[i].set_linewidth(1.8)
            ax1.text(x[i] - w / 2, costs[i] + max(costs) * 0.01, "*",
                     ha="center", fontsize=12, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Mean cost")
    ax2.set_ylabel("Catastrophe rate (%)")
    ax1.set_title("Policy comparison — all agents")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.tight_layout()
    out = results_dir / "fig_d3qn_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Significance table (honest: n=1 for D3QN, no statistical test)
# ---------------------------------------------------------------------------

def _save_significance_table(
    results:     Dict[str, Dict[str, float]],
    results_dir: Path,
) -> None:
    csv_path  = results_dir / "table_d3qn_significance.csv"
    note_path = results_dir / "table_d3qn_significance_footnote.txt"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Agent", "Cost_mean", "Cost_std", "Catastrophe_pct",
                    "State_Dim", "N_seeds_evaluated"])
        for name, m in results.items():
            n_seeds = 1 if "D3QN" in name else "see table_significance.csv"
            w.writerow([
                name,
                round(m["mean_cost"], 3),
                round(m["std_cost"], 3),
                round(m["catastrophe_pct"], 2),
                _STATE_DIM_MAP.get(name, "?"),
                n_seeds,
            ])

    footnote = (
        "Significance testing not applicable: D3QN-CVaR evaluated on single "
        "seed/checkpoint due to time constraints. Comparisons are point estimates, "
        "not statistically validated. See Phase 3 audit (table_significance.csv) "
        "for the seed-sensitivity analysis that motivates this caveat."
    )
    with open(note_path, "w", encoding="utf-8") as fh:
        fh.write(footnote + "\n")

    logger.info("Saved %s", csv_path)
    logger.info("Saved %s", note_path)


# ---------------------------------------------------------------------------
# Verdict printer
# ---------------------------------------------------------------------------

def _print_verdict(results: Dict[str, Dict[str, float]]) -> None:
    d3qn_m = results.get("D3QN-CVaR (ours)")
    if d3qn_m is None:
        logger.error("D3QN-CVaR not in results — cannot print verdict.")
        return

    d3qn_cost   = d3qn_m["mean_cost"]
    d3qn_catast = d3qn_m["catastrophe_pct"]

    others = {k: v for k, v in results.items() if k != "D3QN-CVaR (ours)"}
    if others:
        best_catast_name = min(others, key=lambda n: others[n]["catastrophe_pct"])
        best_cost_name   = min(others, key=lambda n: others[n]["mean_cost"])
    else:
        best_catast_name = best_cost_name = "N/A"

    def _delta(new: float, old: float, unit: str) -> str:
        diff = new - old
        pct  = diff / max(abs(old), 1e-8) * 100
        direction = "better" if diff < 0 else "worse"
        if unit == "pp":
            return f"{direction} by {abs(diff):.1f} pp"
        return f"{direction} by {abs(pct):.1f}%"

    cvar_m    = results.get("CVaR QR-DQN")
    dueling_m = results.get("Dueling DQN")

    # Undominated check: no prior agent strictly better on BOTH metrics
    dominated_by: List[str] = []
    for name, m in others.items():
        if m["mean_cost"] <= d3qn_cost and m["catastrophe_pct"] <= d3qn_catast:
            dominated_by.append(name)

    conclusion = "DOES" if not dominated_by else "DOES NOT"
    undominated_note = (
        "" if not dominated_by
        else f" (undominated by: {', '.join(dominated_by)})"
    )

    lines = [
        "",
        "=== DREAM ARCHITECTURE VERDICT ===",
        f"D3QN-CVaR cost: {d3qn_cost:.2f} | catastrophe: {d3qn_catast:.1f}%",
    ]
    if best_catast_name != "N/A":
        bc = others[best_catast_name]
        lines.append(
            f"Best prior agent (by catastrophe): {best_catast_name} "
            f"at {bc['catastrophe_pct']:.1f}%"
        )
        lines.append(
            f"Best prior agent (by cost):        {best_cost_name} "
            f"at {others[best_cost_name]['mean_cost']:.2f}"
        )
    if cvar_m:
        lines.append(
            f"D3QN-CVaR vs CVaR QR-DQN baseline: "
            f"cost {_delta(d3qn_cost, cvar_m['mean_cost'], '%')}, "
            f"catastrophe {_delta(d3qn_catast, cvar_m['catastrophe_pct'], 'pp')}"
        )
    if dueling_m:
        lines.append(
            f"D3QN-CVaR vs Dueling DQN baseline: "
            f"cost {_delta(d3qn_cost, dueling_m['mean_cost'], '%')}, "
            f"catastrophe {_delta(d3qn_catast, dueling_m['catastrophe_pct'], 'pp')}"
        )
    lines.append(
        f"CONCLUSION: combining dueling + distributional + CVaR {conclusion}"
        f" produce uniformly superior performance based on this single run."
        f"{undominated_note}"
    )
    lines.append("=== END VERDICT ===")

    verdict = "\n".join(lines)
    logger.info(verdict)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Evaluate D3QN-CVaR vs all prior agents.")
    p.add_argument("--processed-dir", default="data/processed", type=Path)
    p.add_argument("--results-dir",   default="results/10_dueling_distributional_d3qn_negative_result", type=Path)
    p.add_argument("--device",        default=None)
    p.add_argument("--seed",          default=42, type=int)
    args = p.parse_args()

    results_dir   = Path(args.results_dir)
    processed_dir = Path(args.processed_dir)
    rul_ckpt      = results_dir.parent / "01_rul_predictor" / "rul_model_best.pth"
    device_str    = args.device or get_device()
    seed          = args.seed

    d3qn_ckpt = results_dir / "d3qn_cvar_best.pth"
    if not d3qn_ckpt.exists():
        for alt in ["d3qn_cvar_final.pth", "d3qn_cvar_fallback_ep1000.pth"]:
            candidate = results_dir / alt
            if candidate.exists():
                logger.warning(
                    "d3qn_cvar_best.pth not found; using %s instead."
                    " This checkpoint may be a do-nothing collapse, not a learned policy.",
                    alt,
                )
                d3qn_ckpt = candidate
                break
        else:
            logger.error(
                "ERROR: results/d3qn_cvar_best.pth not found. "
                "Run 'python -m src.train_d3qn --episodes 5000' first, then re-run this script."
            )
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("D3QN-CVaR Evaluation | device=%s | n_eval=%d", device_str, _N_EVAL)
    logger.info("=" * 60)

    # ---- Load agents -------------------------------------------------------
    agents, envs = _load_agents(results_dir, processed_dir, rul_ckpt, device_str, seed)

    # ---- Evaluate each agent in its own state space ----------------------
    results: Dict[str, Dict[str, float]] = {}
    for name, agent in agents.items():
        env = envs[name]
        state_dim = _STATE_DIM_MAP.get(name, "?")
        logger.info("Evaluating %s (state_dim=%s) on %d episodes ...", name, state_dim, _N_EVAL)
        results[name] = _eval_agent(agent, env, _N_EVAL, seed + 500_000, name)
        logger.info(
            "  %s: cost=%.2f  catast=%.1f%%  act=[dn=%.0f%% rp=%.0f%% rx=%.0f%%]",
            name,
            results[name]["mean_cost"],
            results[name]["catastrophe_pct"],
            results[name]["dn_pct"],
            results[name]["rp_pct"],
            results[name]["rx_pct"],
        )

    # ---- Generate outputs -------------------------------------------------
    _fig_learning_curve(results_dir)
    _fig_risk_return(results, results_dir)
    _fig_comparison(results, results_dir)
    _save_significance_table(results, results_dir)
    _print_verdict(results)

    logger.info("evaluate_d3qn done. Outputs in %s", results_dir)


if __name__ == "__main__":
    main()
